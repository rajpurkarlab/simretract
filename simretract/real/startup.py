"""Startup-only observation quality; no robot, camera, or model dependencies."""
import numpy as np


class StartupTrackingError(RuntimeError):
    pass


class StartupReadiness:
    """Require several distinct usable observations, never a repeated stale one."""
    def __init__(self, min_tool_px=300):
        self.min_raw = max(100, int(min_tool_px))
        self.count = 0
        self.first_source = None
        self.last_source = self.last_result = None
        self.reason = 'waiting for target mask'
        self.raw_pixels = self.effective_pixels = 0

    def observe(self, source_time, result_time, mask, depth_mm, effective_mask=None):
        source_time, result_time = float(source_time), float(result_time)
        if not np.isfinite([source_time, result_time]).all():
            self.count, self.first_source = 0, None
            self.reason = 'invalid observation timestamp'
            self.last_source = self.last_result = None
            return
        if self.last_source is not None and source_time <= self.last_source:
            return  # Polling the same result cannot establish stable tracking.
        gap = None if self.last_source is None else source_time - self.last_source
        self.last_source, self.last_result = source_time, result_time
        raw = np.asarray(mask) > 0
        effective = raw if effective_mask is None else np.asarray(effective_mask) > 0
        depth = np.asarray(depth_mm)
        self.raw_pixels, self.effective_pixels = int(raw.sum()), int(effective.sum())
        valid = np.isfinite(depth) & (depth > 0)
        reason = None
        if result_time < source_time or result_time - source_time > .75:
            reason = 'waiting for a fresh tracker frame'
        elif self.raw_pixels < self.min_raw:
            reason = f'target mask too small ({self.raw_pixels}/{self.min_raw} px)'
        elif np.count_nonzero(raw & valid) < .5 * self.raw_pixels:
            reason = 'target depth is not usable'
        elif self.effective_pixels < 100:
            reason = 'metric crop has too little usable target'
        elif np.count_nonzero(effective & valid) < .5 * self.effective_pixels:
            reason = 'cropped target depth is not usable'
        if reason is not None:
            self.count, self.first_source, self.reason = 0, None, reason
            return
        if gap is not None and gap > .75:
            self.count, self.first_source = 0, None
        if self.first_source is None:
            self.first_source = source_time
        self.count += 1
        self.reason = 'confirming stable target'

    def snapshot(self, now):
        ready = False
        reason = self.reason
        if self.last_source is None:
            pass
        elif now - self.last_source > .75 or now - self.last_result > .75:
            reason = 'waiting for a fresh tracker frame'
        elif self.count >= 3 and self.last_source - self.first_source >= .25:
            ready, reason = True, 'target tracking ready'
        return {'ready': ready, 'reason': reason, 'raw_pixels': self.raw_pixels,
                'effective_pixels': self.effective_pixels, 'stable_frames': self.count,
                'source_time': self.last_source, 'result_time': self.last_result}
