"""Closed-loop take-over client for the real WidowX-AI follower arm.

The loop mirrors the simulation benchmark (:mod:`simretract.eval.sim_eval`):

* every ``exec_h`` ticks: capture a D405 RGB-D frame cropped to the
  simulated wrist camera's 58 degree vertical pinhole, take the SAM 3 tool
  mask from the background :class:`MaskStream`, look the gripper silhouette
  up in the opening-indexed LUT (``assets/gmask_lut.npz``), read the joint
  positions and query the policy server for a (50, 7) joint-target chunk;
* every tick (25 Hz): take the next chunk row, smooth it with an EMA
  against the previously executed action, clamp the per-tick motion and
  send it to the arm with a time-parameterised move.

Safety features that can drive real motors:

* dry-run by default: full perception, inference and logging with the arm
  read-only; pass ``--go`` to move and confirm at the ENTER gate;
* per-tick clamp: at most ``--max-step`` rad per joint and 5 mm on the
  gripper; ``--speed`` < 1 stretches every move's goal time;
* bad-observation hold: when the tool is (almost) invisible or its depth
  is mostly invalid the arm holds its pose instead of re-planning;
* gripper-close detection mirrors the simulation stall rule; on close the
  policy stops, the arm freezes and the gripper firm-closes;
* Ctrl-C returns the arm home while opening the gripper.

Every run is recorded under ``<eval>/real/<checkpoint>/run_<stamp>/``:
``settings.json``, ``q_NNN.png`` and ``q_NNN_raw.npz`` per query,
``log.npz``, ``result.json`` and ``review_view.mp4``.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from simretract.real import (FOLLOWER_IP_PLACEHOLDER, default_runs_root,
                             follower_ip_is_set, load_real_config, repo_root)
from simretract.serve.protocol import PolicyClient
from simretract.serve.metric import metric_plane_inputs
from simretract.real.startup import StartupReadiness, StartupTrackingError

log = logging.getLogger("simretract.real.client")

H = 50
"""Action-chunk length on the wire; shorter policy horizons are padded by the server."""
FOVY_SIM = 58.0
"""Vertical field of view (degrees) of the simulated wrist camera."""
PLANE_K = np.array([[433.0, 0, 320.0], [0, 433.0, 240.0], [0, 0, 1.0]])
"""Pinhole used for every plane fit (fx = fy = 240 / tan(29 deg)); matches training."""
PLANE_TOP_CUT_MAX_DEG = 20.0
"""A top-cut plane swinging more than this from the uncut fit is rejected."""
HOME_POSE = np.array([0.0] * 6 + [0.044])
"""Simulation start pose: joints at zero, gripper fully open; every episode starts here."""
GRIPPER_MAX = 0.044
GRIPPER_STEP = 0.005
"""Per-tick gripper clamp in metres."""
OPEN_RANGE = 0.036
"""Openings at or above this never count as closing."""
STALL_EPS, STALL_N = 8e-4, 12
"""Close rule: carriage moves < STALL_EPS m/tick for STALL_N consecutive ticks while closing."""
DEPTH_VIS_M = 1.5
"""Fixed metric depth scale used for visualisation (same as the model input)."""
CONTROL_HZ = 25.0


def env_default(cast: Callable[[str], Any], default: Any, *names: str) -> Any:
    """Default for a flag: the first set environment variable among ``names``, else ``default``."""
    for n in names:
        v = os.environ.get(n)
        if v is not None and v != "":
            return cast(v)
    return default


# ------------------------------------------------------------------ camera
def compute_crop(fy: float, ppx: float, ppy: float, fovy_deg: float = FOVY_SIM,
                 width: int = 640, height: int = 480) -> tuple[int, int, int, int]:
    """Centre crop (y0, x0, ch, cw) that gives a ``fovy_deg`` vertical FOV.

    The crop is centred on the measured principal point and clamped to the
    frame; resizing it back to ``width x height`` yields an effective focal
    length of ``fy * height / ch``.
    """
    ch = int(round(2 * fy * np.tan(np.radians(fovy_deg) / 2)))
    cw = int(round(ch * width / height))
    y0 = int(round(ppy - ch / 2))
    x0 = int(round(ppx - cw / 2))
    y0 = max(0, min(height - ch, y0))
    x0 = max(0, min(width - cw, x0))
    return y0, x0, ch, cw


class D405:
    """Aligned 640x480 RGB-D from a RealSense D405, cropped to the simulated
    camera's vertical FOV (centre crop about the principal point, resized back)."""

    def __init__(self, warmup_frames: int = 15):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        try:
            prof = self.pipe.start(cfg)
            self.align = rs.align(rs.stream.color)
            self.scale = prof.get_device().first_depth_sensor().get_depth_scale()
            i = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            fy_target = 240.0 / np.tan(np.radians(FOVY_SIM) / 2)
            self.crop = compute_crop(i.fy, i.ppx, i.ppy)
            y0, x0, ch, cw = self.crop
            log.info("camera depth_scale %.6f m/unit; fy %.1f -> crop %dx%d @(%d,%d) -> 640x480 "
                     "(fy_eff %.1f, target %.1f)", self.scale, i.fy, cw, ch, x0, y0,
                     i.fy * 480 / ch, fy_target)
            for _ in range(warmup_frames):  # auto exposure settles
                self.pipe.wait_for_frames()
        except BaseException:
            self.stop()
            raise

    def frame(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (rgb u8 (480,640,3), depth_mm u16 (480,640)) in the cropped view."""
        fr = self.align.process(self.pipe.wait_for_frames())
        rgb = np.asanyarray(fr.get_color_frame().get_data())
        dep = np.asanyarray(fr.get_depth_frame().get_data())
        y0, x0, ch, cw = self.crop
        rgb = cv2.resize(rgb[y0:y0 + ch, x0:x0 + cw], (640, 480), interpolation=cv2.INTER_AREA)
        dep_m = dep[y0:y0 + ch, x0:x0 + cw].astype(np.float32) * self.scale
        dep_mm = np.clip(dep_m * 1000.0, 0, 65535).astype(np.uint16)
        dep_mm = cv2.resize(dep_mm, (640, 480), interpolation=cv2.INTER_NEAREST)
        return rgb, dep_mm

    def stop(self) -> None:
        try:
            self.pipe.stop()
        except Exception:
            pass


# ------------------------------------------------------------------ arm
class Arm:
    """Raw ``trossen_arm`` driver: configure() is communication only, every
    move is explicit and time-parameterised.

    Motor gains are reset to the factory defaults on every connect because
    gain changes (for example a stiff hold) persist on the controller across
    sessions and a raised kp can trip an effort fault on the next home ramp.
    """

    def __init__(self, ip: str, model: str = "wxai_v0"):
        import trossen_arm
        self.ta = trossen_arm
        self.d = trossen_arm.TrossenArmDriver()
        try:
            self.d.configure(model=getattr(trossen_arm.Model, model),
                             end_effector=getattr(trossen_arm.StandardEndEffector, f"{model}_follower"),
                             serv_ip=ip, clear_error=True)
            try:
                self.d.set_motor_parameters(
                    getattr(trossen_arm.StandardMotorParameters, f"{model}_latest"))
                self._default_params = self.d.get_motor_parameters()
            except Exception as e:
                log.warning("could not reset motor gains (%r)", e)
                self._default_params = None
            self.present = np.array(self.d.get_all_positions(), float)
        except BaseException:
            self.cleanup()
            raise

    def enable_position_mode(self) -> None:
        """Switch all joints to position mode (holds the current pose)."""
        self.d.set_all_modes(self.ta.Mode.position)

    def read(self) -> np.ndarray:
        """Read the 7 joint positions (6 rad + gripper opening in metres)."""
        self.present = np.array(self.d.get_all_positions(), float)
        return self.present

    def goto(self, p: np.ndarray, goal_time: float) -> None:
        """Blocking move to ``p`` over ``goal_time`` seconds."""
        self.d.set_all_positions([float(x) for x in p], goal_time=float(goal_time), blocking=True)
        self.present = np.asarray(p, float)

    def cleanup(self) -> None:
        try:
            self.d.cleanup()
        except Exception:
            pass


def ramp_to(arm: Arm, target: np.ndarray, joint_step: float = 0.02, gripper_step: float = 0.002,
            goal_time: float = 0.1) -> int:
    """Move the arm to ``target`` in micro-steps bounded per joint/gripper; returns the step count."""
    cur = arm.read()
    n = max(1, int(np.ceil(max(np.abs(target[:6] - cur[:6]).max() / joint_step,
                               abs(target[6] - cur[6]) / gripper_step))))
    for k in range(1, n + 1):
        arm.goto(cur + (target - cur) * k / n, goal_time)
    return n


# ------------------------------------------------------------------ plane / overlay
def plane_top_cut(mask: np.ndarray, frac: float) -> tuple[np.ndarray, np.ndarray | None]:
    """Remove the top ``frac`` of the tool's pixel length before the plane fit.

    The length is the 1..99 percentile span along the mask's principal axis
    (oriented so that "top" is the image-up end). The result is guarded at
    the fit: a swing above :data:`PLANE_TOP_CUT_MAX_DEG` versus the uncut
    plane keeps the uncut plane. Returns ``(kept_mask, removed_mask)``;
    ``removed_mask`` is None when nothing was cut. Same algorithm as the
    policy server's per-query cut.
    """
    if frac <= 0:
        return mask, None
    v, u = np.nonzero(mask > 0)
    if len(v) < 300:
        return mask, None
    pts = np.stack([u, v], 1).astype(np.float64)
    c = pts.mean(0)
    _, sv, Vt = np.linalg.svd(pts - c, full_matrices=False)
    ax = Vt[0]
    if ax[1] > 0:
        ax = -ax
    t = (pts - c) @ ax
    t0, t1 = np.percentile(t, 1), np.percentile(t, 99)
    keep = t <= t1 - frac * (t1 - t0)
    if keep.sum() < 200:
        return mask, None
    out = np.zeros_like(mask)
    out[v[keep], u[keep]] = 255
    rem = np.zeros_like(mask)
    rem[v[~keep], u[~keep]] = 255
    return out, rem


def metric_model_preview(rgb, dep_mm, mask, gmask, result,
                         input_mode="masked_rgbd", nogrip=True):
    """Local full-resolution preview using the server's metric visibility."""
    effective = result["mask"] > 0
    grip = gmask > 0
    excluded = (mask > 0) & ~effective & ~grip
    out = rgb.copy()
    dep = np.clip(dep_mm.astype(np.float32) / 1500.0, 0, 1)
    out[excluded] = 0
    dep[excluded] = 0
    if input_mode != "full_rgbd":
        visible = effective | (grip if not nogrip else False)
        out[~visible] = 0
        dep *= visible
    return out, dep


def draw_plane(ov: np.ndarray, plane: dict[str, Any] | None, ext: float = 0.05) -> None:
    """Draw a fitted plane (cyan grid + red normal arrow) on a 640x480 BGR overlay."""
    if not plane or plane.get("n") is None or plane.get("c") is None:
        return
    n = np.asarray(plane["n"], float)
    ct = np.asarray(plane["c"], float)
    if ct[2] <= 0.02:
        return
    e1 = np.cross(n, [0, 0, 1.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.array([1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    for gi in np.linspace(-1, 1, 7):
        for (a0, b0, a1, b1) in ((gi * ext, -ext, gi * ext, ext),
                                 (-ext, gi * ext, ext, gi * ext)):
            p0 = ct + a0 * e1 + b0 * e2
            p1 = ct + a1 * e1 + b1 * e2
            if p0[2] <= 0.02 or p1[2] <= 0.02:
                continue
            q0 = PLANE_K @ (p0 / p0[2])
            q1 = PLANE_K @ (p1 / p1[2])
            cv2.line(ov, (int(q0[0]), int(q0[1])), (int(q1[0]), int(q1[1])), (255, 199, 39), 1)
    tip = ct + 0.045 * n
    q0 = PLANE_K @ (ct / ct[2])
    q1 = PLANE_K @ (tip / tip[2])
    cv2.arrowedLine(ov, (int(q0[0]), int(q0[1])), (int(q1[0]), int(q1[1])), (54, 65, 255), 3,
                    tipLength=0.25)


def fit_plane_local(dep_mm: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray] | None:
    """Overlay-side plane fit on a (depth, mask) pair from the same tracker triple.

    The policy uses the server's fit (refreshed only at queries); this local
    fit keeps the drawn plane consistent with the drawn mask between queries.
    Same algorithm and conventions as the served fit: 5x5 median blur, 50-900 mm
    band, 5-MAD depth trim (floor 1 cm), total least squares with a 3-sigma
    refit, normal signed toward the camera.
    """
    dep_mm = cv2.medianBlur(dep_mm.astype(np.float32), 5)
    m = (mask > 0) & (dep_mm > 50) & (dep_mm < 900)
    v, u = np.nonzero(m)
    if len(v) < 200:
        return None
    z = dep_mm[v, u].astype(np.float64) / 1000.0
    zm = np.median(z)
    mad = np.median(np.abs(z - zm))
    keep0 = np.abs(z - zm) <= max(5.0 * mad, 0.01)
    if keep0.sum() >= 200:
        u, v, z = u[keep0], v[keep0], z[keep0]
    fx, cx, cy = PLANE_K[0, 0], PLANE_K[0, 2], PLANE_K[1, 2]
    P = np.stack([(u - cx) / fx * z, (v - cy) / fx * z, z], 1)
    c = P.mean(0)
    _, s, Vt = np.linalg.svd(P - c, full_matrices=False)
    n = Vt[2]
    d = np.abs((P - c) @ n)
    keep = d < 3.0 * max(d.std(), 1e-6)
    if keep.sum() >= 100:
        c = P[keep].mean(0)
        _, s, Vt = np.linalg.svd(P[keep] - c, full_matrices=False)
        n = Vt[2]
    if n[2] > 0:
        n = -n
    return {"n": n, "c": c}


def depth_vis(dep_mm: np.ndarray, visible: np.ndarray | None = None) -> np.ndarray:
    """Colour-map depth on the fixed metric scale the model sees (optionally tool-only)."""
    dep = np.clip(dep_mm.astype(np.float32) / 1000.0 / DEPTH_VIS_M, 0, 1)
    if visible is not None:
        dep = dep * visible
    return cv2.applyColorMap((dep * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)


class MaskStream:
    """Background capture -> SAM 3 tracker -> latest synchronised (rgb, depth_mm, mask).

    Frames are captured at the camera rate and streamed to the mask server
    from a separate worker (~5 fps), so the tracker's temporal memory stays
    fresh between policy queries and the review recording runs at full rate.
    Each masked triple also gets the top-cut and the local plane fit, so the
    overlay is always consistent with the mask it shows.
    """

    def __init__(self, cam: D405, masker: PolicyClient, gmask_fn: Callable[[], np.ndarray],
                 run_dir: str | None = None, top_cut: float = 0.0,
                 top_cut_m: float = 0.0, input_mode: str = "masked_rgbd",
                 nogrip: bool = True, startup_min_tool_px: int = 300):
        self.cam, self.masker, self.gmask_fn = cam, masker, gmask_fn
        self.top_cut = float(top_cut)
        self.top_cut_m = float(top_cut_m)
        self.input_mode, self.nogrip = input_mode, bool(nogrip)
        self.startup = StartupReadiness(startup_min_tool_px)
        self.metric_inputs: dict[str, Any] | None = None
        self.lock = threading.Lock()
        self.latest: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.stamp = 0.0
        self.src_stamp = 0.0  # capture time of the frame the mask was run on
        self.local_plane: dict[str, Any] = {}
        self.top_removed: np.ndarray | None = None
        self.top_cut_guarded = False
        self.err: BaseException | None = None
        self.stop = False
        self.frames_dir: str | None = None
        self.n_frame = 0
        self.rec_t0: float | None = None
        self.rec_t1: float | None = None
        if run_dir is not None:
            self.frames_dir = os.path.join(run_dir, "_vframes")
            os.makedirs(self.frames_dir, exist_ok=True)
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _mask_worker(self, pending: dict, p_lock: threading.Lock) -> None:
        while not self.stop:
            with p_lock:
                item = pending.pop("frame", None)
            if item is None:
                time.sleep(0.005)
                continue
            rgb, dep_mm, cap_t = item
            try:
                mask = np.frombuffer(
                    self.masker.request({"rgb_bytes": rgb.tobytes(), "shape": rgb.shape,
                                         "gmask_bytes": self.gmask_fn().tobytes()}),
                    np.uint8).reshape(480, 640)
            except Exception as e:
                self.err = e
                return
            metric_result = None
            guarded = False
            if self.top_cut_m > 0:
                metric_result = metric_plane_inputs(mask, dep_mm, self.top_cut_m)
                rem, pl = metric_result["removed_mask"], metric_result["plane"]
            else:
                mask_fit, rem = plane_top_cut(mask, self.top_cut)
                pl = fit_plane_local(dep_mm, mask)
                if rem is not None:
                    pl_cut = fit_plane_local(dep_mm, mask_fit)
                    if pl is None:
                        pl = pl_cut
                    elif pl_cut is not None:
                        ang = np.degrees(np.arccos(np.clip(abs(float(pl["n"] @ pl_cut["n"])), 0, 1)))
                        if ang <= PLANE_TOP_CUT_MAX_DEG:
                            pl = pl_cut
                        else:
                            guarded = True
            with self.lock:
                self.latest = (rgb, dep_mm, mask)
                self.stamp = time.time()
                self.src_stamp = cap_t
                self.top_removed, self.top_cut_guarded = rem, guarded
                self.metric_inputs = metric_result
                self.startup.observe(cap_t, self.stamp, mask, dep_mm,
                                     None if metric_result is None else metric_result["mask"])
                if metric_result is not None:
                    self.local_plane = pl or {}
                elif pl is not None:
                    self.local_plane = pl

    def tracking_snapshot(self) -> dict[str, Any]:
        """Read a matched observation and its overlay metadata together."""
        with self.lock:
            return {"observation": self.latest, "source_time": self.src_stamp,
                    "result_time": self.stamp, "metric_inputs": self.metric_inputs,
                    "plane": self.local_plane, "removed_mask": self.top_removed,
                    "guarded": self.top_cut_guarded}

    def _record_frame(self, rgb: np.ndarray, dep_mm: np.ndarray, mask: np.ndarray,
                      overlay: dict[str, Any] | None = None) -> None:
        if overlay is not None and overlay["metric_inputs"] is not None:
            panel = _live_panel(rgb, dep_mm, mask, self, self.n_frame, self.top_cut, overlay)
            cv2.imwrite(os.path.join(self.frames_dir, f"f_{self.n_frame:06d}.png"), panel)
            self.n_frame += 1
            self.rec_t1 = time.time()
            if self.rec_t0 is None:
                self.rec_t0 = self.rec_t1
            return
        ov = rgb.copy()
        m = mask > 0
        ov[m] = (0.45 * ov[m] + 0.55 * np.array([255, 60, 60])).astype(np.uint8)
        # the gripper mask is deliberately not drawn: it never reaches the
        # policy, so showing it in the review video would be misleading
        dvis = depth_vis(dep_mm)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(dvis, cnts, -1, (60, 60, 255), 2)
        ovb = cv2.cvtColor(ov, cv2.COLOR_RGB2BGR)
        if self.top_removed is not None:
            rm = self.top_removed > 0
            ovb[rm] = (0.4 * ovb[rm] + 0.6 * np.array([40, 40, 255])).astype(np.uint8)
        draw_plane(ovb, self.local_plane)
        panel = np.hstack([ovb, dvis])
        cv2.imwrite(os.path.join(self.frames_dir, f"f_{self.n_frame:06d}.png"), panel)
        self.n_frame += 1
        self.rec_t1 = time.time()
        if self.rec_t0 is None:
            self.rec_t0 = self.rec_t1

    def _loop(self) -> None:
        pending: dict[str, Any] = {}
        p_lock = threading.Lock()
        mt = threading.Thread(target=self._mask_worker, args=(pending, p_lock), daemon=True)
        mt.start()
        try:
            while not self.stop:
                rgb, dep_mm = self.cam.frame()
                with p_lock:
                    pending["frame"] = (rgb, dep_mm, time.time())
                with self.lock:
                    mask = self.latest[2] if self.latest is not None else np.zeros((480, 640), np.uint8)
                if self.frames_dir is not None:
                    overlay = self.tracking_snapshot()
                    if self.top_cut_m > 0:
                        if overlay["observation"] is not None:
                            self._record_frame(*overlay["observation"], overlay=overlay)
                    else:
                        self._record_frame(rgb, dep_mm, mask)
        except Exception as e:
            self.err = e
            log.error("mask stream died: %r", e)

    def startup_status(self):
        if self.err is not None:
            raise StartupTrackingError(f'mask stream failed: {self.err!r}')
        with self.lock:
            status = self.startup.snapshot(time.time())
            status['observation'] = self.latest if status['ready'] else None
            return status

    def wait_startup_ready(self, after_ts=0., cancel_check=None, status_callback=None, timeout=30.):
        deadline = time.monotonic() + timeout
        reported, report_at = None, 0.
        while True:
            if cancel_check is not None:
                cancel_check()
            status = self.startup_status()
            if status['ready'] and status['source_time'] >= after_ts:
                return status['observation']
            now = time.monotonic()
            if status['reason'] != reported and now >= report_at:
                reported, report_at = status['reason'], now + 1.
                log.info("waiting for stable target: %s", reported)
                if status_callback is not None:
                    status_callback(reported)
            if now >= deadline:
                raise StartupTrackingError(f'No stable usable target within {timeout:g} seconds; '
                                           'approach cancelled. Restart the trial to select the target again.')
            time.sleep(.02)

    def get_startup(self, after_ts=0.):
        return self.wait_startup_ready(after_ts=after_ts)

    def get(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Latest triple; raises when the stream is dead, stale (> 2 s) or never started (15 s)."""
        t0 = time.time()
        while True:
            if self.err is not None:
                raise RuntimeError(f"mask stream dead: {self.err!r}")
            with self.lock:
                if self.latest is not None:
                    if time.time() - self.stamp > 2.0:
                        raise RuntimeError("mask stream stale >2s")
                    return self.latest
            if time.time() - t0 > 15:
                raise RuntimeError("mask stream never produced a frame")
            time.sleep(0.02)

    def get_settled(self, after_ts: float, timeout: float = 6.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Triple whose source frame was captured after ``after_ts`` (arm stopped and settled).

        Falls back to the freshest triple with a warning when the tracker
        cannot deliver one within ``timeout`` seconds.
        """
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.err is not None:
                raise RuntimeError(f"mask stream dead: {self.err!r}")
            with self.lock:
                if self.latest is not None and self.src_stamp >= after_ts:
                    return self.latest
            time.sleep(0.01)
        log.warning("no settled triple within %.1fs, using freshest", timeout)
        return self.get()

    def close(self) -> None:
        self.stop = True
        self.t.join(timeout=3)


def save_inspect(run_dir: str, k: int, rgb: np.ndarray, dep_mm: np.ndarray, mask: np.ndarray,
                 gmask: np.ndarray, metric_inputs: dict[str, Any] | None = None) -> None:
    """Write the query's 4-panel review row and the raw (depth, mask, rgb) arrays.

    The raw arrays allow plane-fit failures to be re-fit offline; the review
    row shows what the tool-only model sees (masked RGB and masked depth).
    """
    if metric_inputs is not None:
        shape = tuple(metric_inputs["shape"])
        effective = np.frombuffer(metric_inputs["mask"], np.uint8).reshape(shape)
        removed = np.frombuffer(metric_inputs["removed_mask"], np.uint8).reshape(shape)
        camera_shape = tuple(metric_inputs["camera_shape"])
        wrist = np.frombuffer(metric_inputs["wrist"], np.float32).reshape(camera_shape)
        aux = (None if metric_inputs["aux"] is None else
               np.frombuffer(metric_inputs["aux"], np.float32).reshape(camera_shape))
        # These are the actual server cameras after masking and letterbox,
        # before device/model normalization, including no-depth/no-mask modes.
        panels = [cv2.cvtColor((np.clip(wrist, 0, 1) * 255).astype(np.uint8),
                               cv2.COLOR_RGB2BGR)]
        if aux is not None:
            panels.append((np.clip(aux, 0, 1) * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(run_dir, f"q_{k:03d}.png"), np.hstack(panels))
        np.savez_compressed(
            os.path.join(run_dir, f"q_{k:03d}_raw.npz"),
            dep_mm=dep_mm.astype(np.uint16), mask=mask, rgb=rgb, gmask=gmask,
            effective_mask=effective, removed_mask=removed,
            metric_status=metric_inputs["status"],
            plane_top_cut_m=metric_inputs["length_m"],
            input_mode=metric_inputs["input_mode"],
            model_wrist=wrist, model_aux=(np.empty((0,), np.float32)
                                          if aux is None else aux))
        return
    visible = (mask > 0)
    gmask = np.zeros_like(mask)
    mrgb = np.zeros_like(rgb)
    mrgb[visible] = rgb[visible]
    dvis = depth_vis(dep_mm, visible)
    row = np.hstack([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.cvtColor(mrgb, cv2.COLOR_RGB2BGR), dvis,
                     cv2.cvtColor(np.dstack([mask, gmask, np.zeros_like(mask)]), cv2.COLOR_RGB2BGR)])
    cv2.imwrite(os.path.join(run_dir, f"q_{k:03d}.png"), row)
    np.savez_compressed(os.path.join(run_dir, f"q_{k:03d}_raw.npz"),
                        dep_mm=dep_mm.astype(np.uint16), mask=mask, rgb=rgb)


# ------------------------------------------------------------------ pure loop helpers
def exec_horizon(exec_h: int, exec_min: int, tool_depth_m: float | None, near_d: float,
                 far_d: float) -> int:
    """Actions to execute before the next query.

    Far from the tool (median depth >= ``far_d``) the full ``exec_h`` is used;
    at or inside ``near_d`` it drops to ``exec_min`` with linear interpolation
    in between. ``exec_min >= exec_h`` disables the adaptation.
    """
    if exec_min < exec_h and tool_depth_m is not None:
        fr = (tool_depth_m - near_d) / max(far_d - near_d, 1e-6)
        fr = float(np.clip(fr, 0.0, 1.0))
        return int(round(exec_min + fr * (exec_h - exec_min)))
    return exec_h


def clamp_target(act: np.ndarray, cur: np.ndarray, max_step: float) -> tuple[np.ndarray, np.ndarray]:
    """Bound the commanded per-tick motion: joints by ``max_step`` rad, gripper by 5 mm in [0, 0.044]."""
    tgt = act.copy()
    tgt[6] = np.clip(tgt[6], 0.0, GRIPPER_MAX)
    d6 = np.clip(tgt[:6] - cur[:6], -max_step, max_step)
    dg = np.clip(tgt[6] - cur[6], -GRIPPER_STEP, GRIPPER_STEP)
    return tgt, np.concatenate([cur[:6] + d6, [cur[6] + dg]])


class CloseDetector:
    """Gripper-close rule mirroring the simulation stall rule.

    The simulation senses pad contact; the real arm does not, so "closing"
    is: commanded opening below ``close_cmd`` while the measured opening has
    left the open range, and the carriage has stalled (< STALL_EPS per tick)
    for STALL_N consecutive ticks.
    """

    def __init__(self, close_cmd: float):
        self.close_cmd = close_cmd
        self.stall = 0
        self.prev_open: float | None = None

    def update(self, cmd_open: float, q_open: float) -> bool:
        closing = (cmd_open < self.close_cmd) and (q_open < OPEN_RANGE)
        if closing and self.prev_open is not None and abs(q_open - self.prev_open) < STALL_EPS:
            self.stall += 1
        else:
            self.stall = 0
        self.prev_open = q_open
        return self.stall >= STALL_N


def find_ffmpeg() -> str | None:
    """ffmpeg binary: the imageio-ffmpeg wheel's, else one on PATH, else None."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def encode_review_video(frames_dir: str, mp4: str, fps: int) -> bool:
    """Encode the recorded overlay frames to H.264 (2x nearest upscale keeps
    hard mask edges crisp under chroma subsampling); frames are removed on success."""
    ffbin = find_ffmpeg()
    if ffbin is None:
        log.warning("no ffmpeg available; frames kept in %s", frames_dir)
        return False
    try:
        subprocess.run([ffbin, "-y", "-loglevel", "error", "-framerate", str(fps), "-i",
                        os.path.join(frames_dir, "f_%06d.png"),
                        "-vf", "scale=iw*2:ih*2:flags=neighbor", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "18", "-an", mp4], check=True, timeout=600)
    except Exception as e:
        log.warning("encode failed (%r); frames kept in %s", e, frames_dir)
        return False
    shutil.rmtree(frames_dir)
    return True


def default_lut_path() -> Path:
    """Shipped gripper-silhouette LUT (``<assets>/gmask_lut.npz``)."""
    try:
        from simretract import config
        return Path(config.paths()["assets"]) / "gmask_lut.npz"
    except Exception:
        return repo_root() / "assets" / "gmask_lut.npz"


class LiveView:
    """Optional on-screen overlay; degrades to a one-time warning without a display."""

    def __init__(self, enabled: bool):
        self.enabled = bool(enabled)
        if self.enabled and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            log.warning("--view requested but no display is available; live view disabled")
            self.enabled = False

    def show(self, panel: np.ndarray) -> None:
        if not self.enabled:
            return
        try:
            cv2.imshow("take-over live", panel)
            cv2.waitKey(1)
        except cv2.error as e:
            log.warning("live view unavailable (%s); disabled", str(e).splitlines()[0])
            self.enabled = False

    def close(self) -> None:
        if self.enabled:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


def validate_server_capabilities(policy_ping: dict, mask_ping: dict, metric_length: float) -> None:
    """Reject incompatible services before connecting a camera or robot."""
    if not np.isfinite(metric_length) or metric_length < 0:
        raise ValueError("metric cut length must be finite and nonnegative")
    missing = [name for name in ("warm_reset", "locked_track_keepalive") if not mask_ping.get(name)]
    if missing:
        raise StartupTrackingError("restart the mask server; missing capabilities: " + ", ".join(missing))
    if metric_length > 0 and not policy_ping.get("metric_plane_addon"):
        raise RuntimeError("restart the policy server to enable metric preprocessing")


# ------------------------------------------------------------------ main loop
def run(a: argparse.Namespace) -> str:
    """Execute one take-over episode; returns the run directory."""
    with ExitStack() as preflight:
        with np.load(a.lut) as lut:
            lut_open, lut_masks = lut["openings"], lut["masks"]
        policy = PolicyClient(a.host, a.port)
        preflight.callback(policy.close)
        masker = PolicyClient(a.host, a.mask_port)
        preflight.callback(masker.close)
        pp = policy.request({"op": "ping"})
        # runs are grouped by the served checkpoint (its directory name; the
        # trainer's <run>/final layout is named after <run>)
        ck = Path(str(pp.get("ckpt", "")).rstrip("/"))
        model_name = (ck.parent.name if ck.name == "final" else ck.name) or "unknown_model"
        run_dir = a.out_dir or os.path.join(a.runs_root, model_name, time.strftime("run_%Y%m%d_%H%M%S"))
        os.makedirs(run_dir, exist_ok=True)
        log.info("recording -> %s", run_dir)
        with open(os.path.join(run_dir, "settings.json"), "w") as f:
            json.dump({"ckpt": pp.get("ckpt"), "model": model_name, "args": vars(a), "server_ping": pp,
                       "start": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=1, default=str)
        # the server reads nogrip from norm.json and zeros the gripper mask
        # itself; this is the client-side guarantee that it never leaves this
        # process for a tool-only checkpoint
        server_nogrip = bool(pp.get("nogrip", True))
        mp = masker.request({"op": "ping"})
        log.info("servers ok: %s (nogrip=%s) | mask prompt %r", pp.get("ckpt"), server_nogrip, mp.get("prompt"))
        validate_server_capabilities(pp, mp, a.plane_top_cut_m)
        cam = D405()
        preflight.callback(cam.stop)
        arm = Arm(a.follower_ip, a.arm_model)
        preflight.callback(arm.cleanup)
        q0 = arm.read()
        log.info("arm at %s", np.round(q0, 3))
        if a.go:
            arm.enable_position_mode()
            if not a.yes:
                input(f"*** LIVE on {a.follower_ip}: workspace clear, hand on e-stop? "
                      "ENTER to start, Ctrl-C to abort *** ")
            # every training episode starts at the simulation home with the
            # gripper fully open; a part-closed gripper is out of distribution
            log.info("ramping to sim home ...")
            n = ramp_to(arm, HOME_POSE)
            log.info("at home after %d micro-steps: %s", n, np.round(arm.read(), 3))
        else:
            log.info("DRY-RUN: perception + inference only, the arm will not move (pass --go to move)")

        dt = 1.0 / CONTROL_HZ
        goal_t = dt / max(a.speed, 1e-3)
        chunk, prev_a = None, None
        closed, aborted = False, False
        ci, cur_exec, n_query = 10 ** 9, a.exec_h, 0   # force a query at tick 0
        hold_t0: float | None = None
        closed_tick: int | None = None
        closer = CloseDetector(a.close_cmd)
        tgt = cur = q0.copy()
        rec: dict[str, list] = {"state": [], "action": [], "sent": [], "q_tick": [], "plane_n": [],
                                "plane_c": [], "tool_px": []}
        masker.request({"op": "reset"})  # fresh detection for this episode
        plane_ref: dict[str, Any] = {}

        def gmask_now() -> np.ndarray:
            return lut_masks[int(np.argmin(np.abs(lut_open - arm.present[6])))]

        if a.plane_top_cut_m > 0:
            log.info("metric crop on: %.1f cm (plane + model visibility)", 100 * a.plane_top_cut_m)
        elif a.plane_top_cut > 0:
            log.info("plane top-cut on: %.2f of the tool length (server + overlay)", a.plane_top_cut)
        view = LiveView(a.view)
        preflight.callback(view.close)
        stream = MaskStream(cam, masker, gmask_now, run_dir=run_dir, top_cut=a.plane_top_cut,
                            top_cut_m=a.plane_top_cut_m, input_mode=pp.get("input_mode", "masked_rgbd"),
                            nogrip=server_nogrip, startup_min_tool_px=a.min_tool_px)
        preflight.callback(stream.close)
        preflight.pop_all()  # The episode finally below now owns these resources.
    try:
        for tick in range(a.max_ticks):
            if ci >= cur_exec:
                t0 = time.time()
                if chunk is None:
                    rgb, dep_mm, mask = stream.get_startup(after_ts=t0 + a.settle)
                elif a.settle > 0:
                    # the arm holds its last commanded pose; the input triple's
                    # source frame must postdate the settle window
                    rgb, dep_mm, mask = stream.get_settled(t0 + a.settle)
                else:
                    rgb, dep_mm, mask = stream.get()
                # bad-observation guard: do not re-plan on an invisible tool
                # or invalid depth; hold and retry for at most hold_max seconds
                tm = mask > 0
                nv = int((tm & (dep_mm > 0)).sum())
                if (chunk is not None and a.min_tool_px > 0
                        and (int(tm.sum()) < a.min_tool_px or nv < 0.5 * max(int(tm.sum()), 1))
                        and (time.time() - hold_t0 if hold_t0 else 0.0) < a.hold_max):
                    if hold_t0 is None:
                        hold_t0 = time.time()
                        log.info("tick %d: tool not visible (px=%d, valid depth %d); holding",
                                 tick, int(tm.sum()), nv)
                    if a.go:
                        arm.goto(arm.present, goal_t)
                    else:
                        time.sleep(goal_t)
                    continue
                hold_t0 = None
                qpos = arm.read().astype(np.float32)
                gmask = lut_masks[int(np.argmin(np.abs(lut_open - qpos[6])))]
                metric_gmask = gmask
                if server_nogrip:
                    gmask = np.zeros_like(gmask)
                state14 = np.concatenate([qpos, np.zeros(7, np.float32)])
                resp = policy.request({"rgb": rgb, "depth_mm": dep_mm, "mask": mask, "gmask": gmask,
                                       "state": state14, "want_plane": True,
                                       "plane_top_cut": float(a.plane_top_cut),
                                       "plane_top_cut_m": float(a.plane_top_cut_m),
                                       **({"metric_gripper_mask": metric_gmask}
                                          if a.plane_top_cut_m > 0 else {})})
                metric_inputs = None
                if isinstance(resp, dict):
                    chunk = np.asarray(resp["chunk"], dtype=np.float32)
                    plane_ref["n"] = resp.get("plane_n")
                    plane_ref["c"] = resp.get("plane_c")
                    metric_inputs = resp.get("metric_inputs")
                else:
                    chunk = np.asarray(resp, dtype=np.float32)
                if a.plane_top_cut_m > 0 and metric_inputs is None:
                    raise RuntimeError("policy server did not return metric model inputs")
                if chunk.shape != (H, 7):
                    raise RuntimeError(f"bad chunk shape {chunk.shape}, expected {(H, 7)}")
                k = n_query
                n_query += 1
                ci = 0
                td = dep_mm[tm & (dep_mm > 0)]
                d_med = float(np.median(td)) / 1000.0 if len(td) else None
                cur_exec = exec_horizon(a.exec_h, a.exec_min, d_med, a.near_d, a.far_d)
                save_inspect(run_dir, k, rgb, dep_mm, mask, gmask, metric_inputs=metric_inputs)
                rec["q_tick"].append(tick)
                rec["plane_n"].append(np.asarray(plane_ref.get("n") or [np.nan] * 3, np.float32))
                rec["plane_c"].append(np.asarray(plane_ref.get("c") or [np.nan] * 3, np.float32))
                rec["tool_px"].append(int(tm.sum()))
                gm = gmask > 0
                gd = dep_mm[gm & (dep_mm > 0)]
                log.info("q%d: exec %d  settle+mask+infer %.2fs  tool_px=%d  open=%.4f  "
                         "tool_d=%.3fm(%.0f%%valid)  grip_d=%.3fm(%.0f%%valid)",
                         k, cur_exec, time.time() - t0, int(tm.sum()), qpos[6],
                         (np.median(td) / 1000 if len(td) else 0), 100 * len(td) / max(tm.sum(), 1),
                         (np.median(gd) / 1000 if len(gd) else 0), 100 * len(gd) / max(gm.sum(), 1))
            act = chunk[ci]
            ci += 1
            if a.smooth > 0:
                act = ((1 - a.smooth) * act + a.smooth * prev_a) if tick > 0 else act
            prev_a = act
            cur = arm.present
            tgt, sent = clamp_target(act, cur, a.max_step)
            rec["state"].append(arm.present.copy())
            rec["action"].append(act.copy())
            rec["sent"].append(sent.copy())
            if a.go:
                arm.goto(sent, goal_t)
            else:
                time.sleep(goal_t)
            if a.close_stop and not closed:
                q_open = float(arm.read()[6]) if a.go else float(sent[6])
                if closer.update(float(act[6]), q_open):
                    closed = True
                    closed_tick = tick
                    log.info("CLOSE detected at tick %d (opening %.4f, cmd %.4f); model stopped, arm frozen",
                             tick, q_open, act[6])
                    if a.go:
                        _firm_close_and_hold(arm, a.stiff_hold)
                    break
            if view.enabled and tick % 5 == 0:
                lrgb, ldep, lmask = stream.latest or (None, None, None)
                if lrgb is not None:
                    view.show(_live_panel(lrgb, ldep, lmask, stream, tick, a.plane_top_cut))
            if tick % 25 == 0:
                clip = float(np.abs(tgt[:6] - cur[:6]).max())
                log.info("tick %d/%d max|dtgt|=%.3frad%s", tick, a.max_ticks, clip,
                         " (clamped)" if clip > a.max_step else "")
    except StartupTrackingError as exc:
        aborted = True
        log.error("start cancelled: %s", exc)
        raise
    except KeyboardInterrupt:
        aborted = True
        log.info("ABORTED; returning home with the gripper opening")
    finally:
        stream.close()
        view.close()
        if stream.n_frame > 1 and stream.frames_dir is not None:
            fps = max(1, round(stream.n_frame / max((stream.rec_t1 or 0) - (stream.rec_t0 or 0), 1)))
            mp4 = os.path.join(run_dir, "review_view.mp4")
            if encode_review_video(stream.frames_dir, mp4, fps):
                log.info("review video -> %s (%d frames @%dfps)", mp4, stream.n_frame, fps)
        np.savez(os.path.join(run_dir, "log.npz"), **{k: np.array(v) for k, v in rec.items()})
        res = {"model": model_name, "closed": bool(closed),
               "closed_tick": int(closed_tick) if closed else None,
               "aborted": bool(aborted), "ticks": len(rec["state"]), "queries": int(n_query),
               "final_qpos": [round(float(v), 4) for v in arm.present],
               "end": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(os.path.join(run_dir, "result.json"), "w") as f:
            json.dump(res, f, indent=1)
        log.info("result: %s", res)
        cam.stop()
        if a.go:
            _return_home(arm, aborted, a.home_speed)
        arm.cleanup()
        policy.close()
        masker.close()
        log.info("done: %s", run_dir)
    return run_dir


def _firm_close_and_hold(arm: Arm, stiff_hold: float) -> None:
    """After a detected close: ramp the gripper to zero over ~1 s, optionally
    stiffen the arm, and hold until the operator presses ENTER."""
    hold = arm.read().copy()
    for k in range(1, 26):
        g = hold[6] * (1 - k / 25.0)
        arm.goto(np.concatenate([hold[:6], [g]]), 0.04)
    if stiff_hold > 1.0:
        # raise the position-loop gains so the pose resists the retraction
        # load; restored before any further motion
        try:
            params = arm.d.get_motor_parameters()
            for j in range(6):
                pp = params[j][arm.ta.Mode.position].position
                pp.kp = pp.kp * stiff_hold
                pp.kd = pp.kd * min(stiff_hold, 2.0)
            arm.d.set_motor_parameters(params)
            log.info("arm stiffened x%.2f for the hold", stiff_hold)
        except Exception as e:
            log.warning("stiffen failed (%r); holding at default gains", e)
    log.info("gripper firm-closed (%.4f); HOLDING pose. Press ENTER to return home "
             "(gripper stays closed), Ctrl-C to abort and open while returning home.", arm.read()[6])
    try:
        input()
    except EOFError:
        time.sleep(1e9)
    if stiff_hold > 1.0 and getattr(arm, "_default_params", None):
        try:
            arm.d.set_motor_parameters(arm._default_params)
            log.info("gains restored to default")
        except Exception as e:
            log.warning("gain restore failed (%r)", e)


def _return_home(arm: Arm, aborted: bool, home_speed: float) -> None:
    """Return the joints to zero. The gripper keeps its opening (a taken-over
    tool would otherwise be dropped) unless the run was aborted, in which
    case it opens while homing. Ctrl-C during the ramp holds the pose."""
    try:
        cur = arm.read()
        home = cur.copy()
        home[:6] = 0.0
        if aborted:
            home[6] = GRIPPER_MAX
        n = max(1, int(np.ceil(np.abs(home[:6] - cur[:6]).max() / (home_speed * 0.1))))
        log.info("returning arm to home (%d micro-steps at %.2f rad/s; gripper %s) ...",
                 n, home_speed, "OPENING" if aborted else "unchanged")
        for k in range(1, n + 1):
            arm.goto(cur + (home - cur) * k / n, 0.1)
        log.info("arm at home")
    except KeyboardInterrupt:
        log.info("home ramp aborted; holding pose")


def _live_panel(lrgb: np.ndarray, ldep: np.ndarray, lmask: np.ndarray, stream: MaskStream,
                tick: int, top_cut: float, overlay: dict[str, Any] | None = None) -> np.ndarray:
    if stream.top_cut_m > 0:
        overlay = overlay or stream.tracking_snapshot()
        if overlay["observation"] is not None:
            lrgb, ldep, lmask = overlay["observation"]
        result = overlay["metric_inputs"]
        if result is not None:
            ov = lrgb.copy()
            effective = result["mask"] > 0
            removed = result["removed_mask"] > 0
            rejected = (lmask > 0) & ~effective & ~removed
            for region, color in ((effective, [60, 255, 60]), (removed, [255, 230, 0]),
                                  (rejected, [255, 60, 60])):
                ov[region] = (0.45 * ov[region] + 0.55 * np.array(color)).astype(np.uint8)
            mrgb, dep = metric_model_preview(lrgb, ldep, lmask, stream.gmask_fn(), result,
                                            stream.input_mode, stream.nogrip)
            if stream.input_mode == "masked_rgb":
                dep = (effective | ((stream.gmask_fn() > 0) if not stream.nogrip else False)).astype(np.float32)
            if stream.input_mode == "masked_rgb_nodepth":
                dvis = cv2.cvtColor(mrgb, cv2.COLOR_RGB2BGR)
                preview_label = "model RGB visibility preview"
            else:
                dvis = cv2.applyColorMap((dep * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
                preview_label = ("model binary mask preview" if stream.input_mode == "masked_rgb"
                                 else "model depth visibility preview")
            ov = cv2.cvtColor(ov, cv2.COLOR_RGB2BGR)
            draw_plane(ov, overlay["plane"])
            age = max(0.0, time.time() - overlay["source_time"])
            cv2.putText(ov, f"metric {100 * stream.top_cut_m:.1f} cm: {result['status']}",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(ov, f"source age {age:.2f}s  tick {tick}", (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(dvis, preview_label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1)
            return np.hstack([ov, dvis])
    ov = lrgb.copy()
    m = lmask > 0
    ov[m] = (0.4 * ov[m] + 0.6 * np.array([255, 60, 60])).astype(np.uint8)
    rem = stream.top_removed
    if rem is not None:
        rm = rem > 0
        ov[rm] = (0.3 * ov[rm] + 0.7 * np.array([255, 230, 0])).astype(np.uint8)
    cv2.putText(ov, f"tool px {int(m.sum())}  tick {tick}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)
    if top_cut > 0:
        if stream.top_cut_guarded:
            st = "trim shown; plane GUARDED -> uncut fit"
        elif rem is not None:
            st = "CUT (yellow) applied to plane"
        else:
            st = "mask too small to trim"
        cv2.putText(ov, f"top-cut {top_cut:.2f}: {st}", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 0), 2)
    ov = cv2.cvtColor(ov, cv2.COLOR_RGB2BGR)
    draw_plane(ov, stream.local_plane)
    dvis = depth_vis(ldep, m)
    cv2.putText(dvis, "policy depth view", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return np.hstack([ov, dvis])


# ------------------------------------------------------------------ CLI
class PlaneArgumentParser(argparse.ArgumentParser):
    """An explicit legacy cut selects the legacy path unless metric is explicit."""

    def parse_known_args(self, args=None, namespace=None):
        tokens = list(sys.argv[1:] if args is None else args)
        parsed, remaining = super().parse_known_args(tokens, namespace)
        flags = {token.split("=", 1)[0] for token in tokens}
        if "--plane-top-cut" in flags and "--plane-top-cut-m" not in flags:
            parsed.plane_top_cut_m = 0.0
        if not np.isfinite(parsed.plane_top_cut_m) or parsed.plane_top_cut_m < 0:
            self.error("--plane-top-cut-m must be finite and nonnegative")
        return parsed, remaining


def build_parser() -> argparse.ArgumentParser:
    """Client flags. Defaults are the deployment regime; each flag falls back
    to ``SIMRETRACT_<FLAG>`` in the environment."""
    ap = PlaneArgumentParser(
        prog="python -m simretract.real.client",
        description="Closed-loop real-arm take-over client (dry-run unless --go).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--go", action="store_true", help="actually move the arm (default: dry-run)")
    ap.add_argument("--yes", action="store_true", help="skip the live confirmation gate")
    ap.add_argument("--follower-ip", dest="follower_ip", default=None,
                    help="follower controller IP; default from configs/real/default.yaml "
                         "robot.follower_ip (must not be CHANGE_ME)")
    ap.add_argument("--arm-model", dest="arm_model", default=env_default(str, "wxai_v0", "SIMRETRACT_ARM_MODEL"),
                    help="trossen_arm model name")
    ap.add_argument("--config", default=None, help="real-run YAML (default configs/real/default.yaml)")
    ap.add_argument("--host", default=env_default(str, "127.0.0.1", "SIMRETRACT_HOST"),
                    help="policy / mask server host")
    ap.add_argument("--port", type=int, default=env_default(int, 8765, "SIMRETRACT_PORT"),
                    help="policy server port")
    ap.add_argument("--mask-port", dest="mask_port", type=int,
                    default=env_default(int, 8766, "SIMRETRACT_MASK_PORT"), help="SAM 3 mask server port")
    ap.add_argument("--exec-h", dest="exec_h", type=int, default=env_default(int, 50, "SIMRETRACT_EXEC_H"),
                    help="actions executed per policy query when the tool is far (>= --far-d)")
    ap.add_argument("--exec-min", dest="exec_min", type=int,
                    default=env_default(int, 50, "SIMRETRACT_EXEC_MIN"),
                    help="actions per query when the tool is at/inside --near-d "
                         "(adaptive horizon; equal to --exec-h disables it)")
    ap.add_argument("--near-d", dest="near_d", type=float, default=env_default(float, 0.10, "SIMRETRACT_NEAR_D"),
                    help="median tool depth (m) at/below which --exec-min applies")
    ap.add_argument("--far-d", dest="far_d", type=float, default=env_default(float, 0.22, "SIMRETRACT_FAR_D"),
                    help="median tool depth (m) at/above which the full --exec-h applies")
    ap.add_argument("--settle", type=float, default=env_default(float, 0.0, "SIMRETRACT_SETTLE"),
                    help="seconds to pause after each chunk; the next query uses a post-stop frame (0 = off)")
    ap.add_argument("--smooth", type=float, default=env_default(float, 0.6, "SIMRETRACT_SMOOTH"),
                    help="EMA weight on the previous executed action: act = (1-s)*act + s*prev")
    ap.add_argument("--speed", type=float, default=env_default(float, 0.8, "SIMRETRACT_SPEED"),
                    help="1.0 = real-time 25 Hz; goal_time = (1/25 s) / speed")
    ap.add_argument("--max-step", dest="max_step", type=float,
                    default=env_default(float, 0.06, "SIMRETRACT_MAX_STEP"),
                    help="max rad per joint per tick (safety clamp; gripper clamp fixed at 5 mm)")
    ap.add_argument("--max-ticks", dest="max_ticks", type=int,
                    default=env_default(int, 451, "SIMRETRACT_MAX_TICKS"), help="episode cap in 25 Hz ticks")
    ap.add_argument("--view", action=argparse.BooleanOptionalAction, default=False,
                    help="live overlay window (needs a display; degrades to a warning without one)")
    ap.add_argument("--plane-top-cut", dest="plane_top_cut", type=float,
                    default=env_default(float, 0.0, "SIMRETRACT_PLANE_TOP_CUT", "PLANE_TOP_CUT"),
                    help="fraction of the tool's pixel length removed from the top of the mask before "
                         "the plane fit, applied identically on the server and the overlay (0 = off)")
    ap.add_argument("--plane-top-cut-m", type=float, nargs="?", const=0.01,
                    default=env_default(float, 0.01, "SIMRETRACT_PLANE_TOP_CUT_M", "PLANE_TOP_CUT_M"),
                    help="metric tip cut in metres for plane and model visibility (bare flag: 0.01; "
                         "0 selects the legacy path; overrides --plane-top-cut)")
    ap.add_argument("--min-tool-px", dest="min_tool_px", type=int,
                    default=env_default(int, 300, "SIMRETRACT_MIN_TOOL_PX"),
                    help="fewer tool pixels than this (or <50%% valid depth on them) at a query = bad "
                         "observation: hold pose and retry instead of re-planning (0 = off)")
    ap.add_argument("--hold-max", dest="hold_max", type=float,
                    default=env_default(float, 2.0, "SIMRETRACT_HOLD_MAX"),
                    help="max seconds to hold on bad observations before re-planning anyway")
    ap.add_argument("--stiff-hold", dest="stiff_hold", type=float,
                    default=env_default(float, 1.0, "SIMRETRACT_STIFF_HOLD"),
                    help="multiply the arm position-loop kp by this after the grasp (1.0 = off)")
    ap.add_argument("--close-stop", dest="close_stop", type=int,
                    default=env_default(int, 1, "SIMRETRACT_CLOSE_STOP"),
                    help="1: on gripper close stop the model, freeze the arm, firm-close and hold until ENTER")
    ap.add_argument("--close-cmd", dest="close_cmd", type=float,
                    default=env_default(float, 0.020, "SIMRETRACT_CLOSE_CMD"),
                    help="policy gripper command (m) below which the gripper counts as closing")
    ap.add_argument("--home-speed", dest="home_speed", type=float,
                    default=env_default(float, 0.6, "SIMRETRACT_HOME_SPEED"),
                    help="rad/s of the return-home ramp after the run or on Ctrl-C")
    ap.add_argument("--lut", default=None, help="gripper silhouette LUT (default assets/gmask_lut.npz)")
    ap.add_argument("--runs-root", dest="runs_root", default=None,
                    help="root for run recordings (default <eval>/real from configs/paths.yaml)")
    ap.add_argument("--out-dir", dest="out_dir", default=None,
                    help="explicit run directory (default <runs-root>/<checkpoint>/run_<stamp>)")
    return ap


def resolve_args(a: argparse.Namespace) -> argparse.Namespace:
    """Fill config-derived defaults (follower IP, LUT, run root) and validate the robot address."""
    if a.follower_ip is None:
        try:
            cfg = load_real_config(a.config)
        except FileNotFoundError:
            cfg = {}
        a.follower_ip = (cfg.get("robot") or {}).get("follower_ip")
        a.arm_model = (cfg.get("robot") or {}).get("model", a.arm_model)
    if not follower_ip_is_set(a.follower_ip):
        raise SystemExit(f"robot.follower_ip is {FOLLOWER_IP_PLACEHOLDER!r}: set it in "
                         "configs/real/default.yaml or pass --follower-ip")
    if a.lut is None:
        a.lut = str(default_lut_path())
    if a.runs_root is None:
        a.runs_root = str(default_runs_root())
    return a


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    a = resolve_args(build_parser().parse_args(argv))
    run(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
