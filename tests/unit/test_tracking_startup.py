"""Startup readiness from synthetic masks, depth, and observation timing."""
import unittest

import numpy as np

from simretract.real.startup import StartupReadiness


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.ones((30, 30), np.uint8)
        self.depth = np.full((30, 30), 200., np.float32)
        self.check = StartupReadiness()

    def feed(self, stamp, mask=None, depth=None, effective=None, lag=.1):
        self.check.observe(stamp, stamp + lag, self.mask if mask is None else mask,
                           self.depth if depth is None else depth, effective)
        return self.check.snapshot(stamp + lag)

    def test_three_distinct_fresh_frames_with_time_span_are_required(self):
        self.assertFalse(self.feed(1.)['ready'])
        self.assertFalse(self.feed(1.1)['ready'])
        self.assertFalse(self.feed(1.2)['ready'])
        self.assertTrue(self.feed(1.3)['ready'])
        self.assertFalse(self.check.snapshot(2.2)['ready'])

    def test_repolling_one_mask_does_not_make_it_ready(self):
        for _ in range(10):
            self.assertFalse(self.feed(1.)['ready'])
        self.assertEqual(self.check.count, 1)

    def test_first_warmup_result_cannot_be_accepted_when_stale(self):
        self.assertFalse(self.feed(1., lag=4.7)['ready'])
        self.assertEqual(self.check.count, 0)
        self.assertFalse(self.feed(5.8)['ready'])
        self.assertFalse(self.feed(5.94)['ready'])
        self.assertTrue(self.feed(6.08)['ready'])

    def test_loss_resets_stability_and_recovery_needs_new_evidence(self):
        for t in (1., 1.15, 1.30):
            self.feed(t)
        self.assertFalse(self.feed(1.45, mask=np.zeros_like(self.mask))['ready'])
        self.assertFalse(self.feed(1.6)['ready'])
        self.assertFalse(self.feed(1.75)['ready'])
        self.assertTrue(self.feed(1.9)['ready'])

    def test_metric_empty_or_bad_depth_cannot_start_but_plane_is_not_required(self):
        empty = np.zeros_like(self.mask)
        self.assertFalse(self.feed(1., effective=empty)['ready'])
        self.assertFalse(self.feed(1.2, depth=np.full_like(self.depth, np.nan))['ready'])
        self.assertFalse(self.feed(1.4, depth=np.zeros_like(self.depth))['ready'])
        for t in (1.6, 1.75, 1.9):
            status = self.feed(t)
        self.assertTrue(status['ready'])  # No plane normal is supplied or required.

    def test_invalid_timestamps_clear_previous_readiness(self):
        for t in (1., 1.15, 1.30):
            self.feed(t)
        self.check.observe(float("nan"), 2., self.mask, self.depth)
        self.assertFalse(self.check.snapshot(2.)["ready"])
        self.assertEqual(self.check.count, 0)

    def test_gap_requires_a_new_stability_window(self):
        for t in (1., 1.15, 1.30):
            self.feed(t)
        self.assertFalse(self.feed(2.1)["ready"])
        self.assertFalse(self.feed(2.25)["ready"])
        self.assertTrue(self.feed(2.4)["ready"])


if __name__ == '__main__':
    unittest.main()
