"""Known-geometry checks for metric cropping, hollow depth, and narrow views."""
import unittest
import numpy as np
from simretract.serve import metric

FY = 240.0 / np.tan(np.radians(58.0) / 2)

def metric_module():
    return metric

def render_shaft(z0=.25, tilt_deg=0., length=.10, width=.015, yaw_deg=0.):
    """Exact ray-plane intersection, uint16-mm sensor quantization."""
    a = np.radians(tilt_deg)
    axis = np.array([0., -np.cos(a), np.sin(a)])
    yaw = np.radians(yaw_deg)
    original_side = np.array([1., 0., 0.])
    side = original_side * np.cos(yaw) + np.cross(axis, original_side) * np.sin(yaw)
    normal = np.cross(side, axis)
    center = np.array([0., 0., z0])
    v, u = np.indices((480, 640))
    rays = np.stack([(u - 320.) / FY, (v - 240.) / FY, np.ones_like(u)], -1)
    z = (normal @ center) / (rays @ normal)
    points = rays * z[..., None]
    t = (points - center) @ axis
    w = (points - center) @ side
    active = (np.abs(t) <= length / 2) & (np.abs(w) <= width / 2)
    mask = active.astype(np.uint8) * 255
    depth = np.where(active, np.rint(z * 1000), 0).astype(np.uint16)
    return mask, depth, normal, t, w

def angle_deg(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.degrees(np.arccos(np.clip(abs(a @ b) /
                      (np.linalg.norm(a) * np.linalg.norm(b)), 0., 1.))))

class MetricGeometryTests(unittest.TestCase):
    def call(self, mask, depth, length=.01):
        return metric_module().metric_plane_inputs(mask, depth, length)

    def assert_result_consistent(self, result, original):
        kept = result['mask'] > 0
        removed = result['removed_mask'] > 0
        self.assertEqual(kept.shape, original.shape)
        self.assertEqual(removed.shape, original.shape)
        self.assertFalse(np.any(kept & ~(original > 0)))
        self.assertFalse(np.any(kept & removed))
        self.assertFalse(np.any(removed & ~(original > 0)))
        if result['plane'] is not None:
            for field in ('n', 'c'):
                self.assertTrue(np.isfinite(result['plane'][field]).all())
            self.assertAlmostEqual(np.linalg.norm(result['plane']['n']), 1., places=4)
            self.assertLessEqual(float(result['plane']['n'][2]), 0.)

    def test_one_centimetre_is_physical_at_several_distances_and_tilts(self):
        for z0, tilt in ((.15, 0), (.25, 0), (.40, 0), (.18, 30), (.25, 55), (.40, 40)):
            with self.subTest(z0=z0, tilt=tilt):
                mask, depth, normal, t, _ = render_shaft(z0, tilt)
                result = self.call(mask, depth)
                self.assert_result_consistent(result, mask)
                self.assertIsNotNone(result['plane'], result['status'])
                self.assertLess(angle_deg(result['plane']['n'], normal), 3.)
                cut = result['removed_mask'] > 0
                self.assertTrue(cut.any(), result['status'])
                kept = result['mask'] > 0
                measured = float(t[mask > 0].max() - t[kept].max())
                # One camera pixel plus depth quantization / robust-tip margin.
                self.assertAlmostEqual(measured, .01, delta=.0025)
                # Geometric top cut must be a contiguous terminal interval.
                self.assertGreaterEqual(float(t[cut].min()), float(t[kept].max()) - .002)

    def test_hollow_handle_background_does_not_become_the_shaft_plane(self):
        for tilt in (0., 35.):
            with self.subTest(tilt=tilt):
                mask, depth, normal, t, w = render_shaft(.25, tilt, width=.018)
                # Segmentation erroneously fills an interior hollow. Its
                # depth belongs to a far surface, not the foreground metal.
                hole = (mask > 0) & (abs(w) < .006) & (t > -.040) & (t < .025)
                depth[hole] = 650
                result = self.call(mask, depth)
                self.assert_result_consistent(result, mask)
                self.assertIsNotNone(result['plane'], result['status'])
                self.assertLess(angle_deg(result['plane']['n'], normal), 3.)
                self.assertLess(np.count_nonzero(hole & (result['mask'] > 0)), .03 * hole.sum())
                lower_metal = (mask > 0) & ~hole & (t < .030)
                self.assertGreater(np.count_nonzero(lower_metal & (result['mask'] > 0)),
                                   .60 * lower_metal.sum())

    def test_bent_top_is_removed_and_lower_normal_is_recovered(self):
        mask, depth, normal, t, _ = render_shaft(.25, 25.)
        tip = (mask > 0) & (t > .044)
        # A short bent terminal segment slopes away from the shaft plane.
        depth[tip] += np.rint((t[tip] - .044) / .006 * 4.).astype(np.uint16)
        result = self.call(mask, depth)
        self.assertIsNotNone(result['plane'], result['status'])
        self.assertLess(angle_deg(result['plane']['n'], normal), 3.)
        self.assertFalse(np.any(tip & (result['mask'] > 0)))

    def test_volkmann_rake_with_filled_washer_handle(self):
        for tilt, yaw in ((0., 0.), (35., 0.), (60., 0.), (35., 35.), (35., 75.)):
            with self.subTest(tilt=tilt, yaw=yaw):
                base, depth, normal, t, w = render_shaft(.28, tilt, length=.15, width=.05, yaw_deg=yaw)
                radius = np.hypot(w, t + .042)
                ring = (radius < .017) & (radius > .010)
                hollow = radius < .010
                stem = (abs(w) < .0045) & (t > -.030) & (t < .040)
                crossbar = (abs(w) < .014) & (t > .035) & (t < .041)
                teeth = np.zeros_like(base, bool)
                for x in (-.012, -.006, 0., .006, .012):
                    teeth |= (abs(w - x) < .0018) & (t > .038) & (t < .064)
                shape = (base > 0) & (ring | hollow | stem | crossbar | teeth)
                mask = shape.astype(np.uint8) * 255
                depth[~shape] = 0
                depth[hollow & shape] = 700
                tip = teeth & shape & (t > .058)
                depth[tip] += np.rint((t[tip] - .058) / .006 * 3.).astype(np.uint16)
                result = self.call(mask, depth)
                self.assert_result_consistent(result, mask)
                self.assertIsNotNone(result['plane'], result['status'])
                self.assertLess(angle_deg(result['plane']['n'], normal), 3.)
                self.assertFalse(np.any(tip & (result['mask'] > 0)))
                self.assertLess(np.count_nonzero(hollow & (result['mask'] > 0)),
                                .05 * np.count_nonzero(hollow & shape))
                actual_metal = shape & ~hollow & (t < .045)
                self.assertGreater(np.count_nonzero(actual_metal & (result['mask'] > 0)),
                                   .65 * actual_metal.sum())

    def test_invalid_depth_and_line_cloud_do_not_fabricate_normals(self):
        mask, depth, *_ = render_shaft()
        for invalid in (np.zeros_like(depth), np.full_like(depth, 1100),
                        np.full(depth.shape, np.nan, dtype=np.float32)):
            with self.subTest(invalid=str(invalid.dtype)):
                result = self.call(mask, invalid)
                self.assertIsNone(result['plane'])
                self.assert_result_consistent(result, mask)
        line_mask = np.zeros_like(mask)
        line_mask[100:380, 320] = 255
        line_depth = np.where(line_mask, 250, 0).astype(np.uint16)
        result = self.call(line_mask, line_depth)
        self.assertIsNone(result['plane'])
        self.assert_result_consistent(result, line_mask)
        # A line is not a plane, but foreground depth still locates its top.
        self.assertTrue(np.any(result['removed_mask'] > 0), result['status'])
        self.assertEqual(result['mask'][100, 320], 0)
        self.assertEqual(result['mask'][379, 320], 255)

    def test_side_on_thin_edge_never_invents_a_front_facing_plane(self):
        for yaw in (80., 85., 88.):
            with self.subTest(yaw=yaw):
                mask, depth, normal, *_ = render_shaft(.25, 20., width=.015, yaw_deg=yaw)
                result = self.call(mask, depth)
                self.assert_result_consistent(result, mask)
                if result['plane'] is not None:
                    self.assertLess(angle_deg(result['plane']['n'], normal), 6., result['status'])
                    self.assertLess(abs(float(result['plane']['n'][2])), .30)

    def test_thin_foreground_ring_must_not_select_its_filled_background(self):
        for thickness in (1, 2, 3, 6):
            with self.subTest(thickness=thickness):
                mask = np.zeros((480, 640), np.uint8)
                mask[100:380, 280:360] = 255
                depth = np.where(mask, 500, 0).astype(np.uint16)
                metal = mask > 0
                metal[100 + thickness:380 - thickness, 280 + thickness:360 - thickness] = False
                depth[metal] = 200
                result = self.call(mask, depth)
                self.assert_result_consistent(result, mask)
                if result['plane'] is not None:
                    self.assertLess(float(result['plane']['c'][2]), .25,
                                    'Far background must not become foreground plane')
                    self.assertLess(angle_deg(result['plane']['n'], [0., 0., -1.]), 3.)
                    self.assertLess(np.count_nonzero((result['support_mask'] > 0) & ~metal), 5)

    def test_disconnected_slanted_rails_keep_their_shared_plane(self):
        mask, depth, normal, t, w = render_shaft(.30, 35., width=.028, yaw_deg=60.)
        # The top bridge is inside the excluded centimetre; the visible
        # lower geometry is two rails with different absolute depths.
        hollow = (abs(w) < .008) & (t < .041)
        mask[hollow] = 0
        depth[mask == 0] = 0
        result = self.call(mask, depth)
        self.assertIsNotNone(result['plane'], result['status'])
        self.assertLess(angle_deg(result['plane']['n'], normal), 4.)
        support = result['support_mask'] > 0
        self.assertGreater(np.count_nonzero(support & (w < -.008)), 80)
        self.assertGreater(np.count_nonzero(support & (w > .008)), 80)

    def test_top_cut_excludes_a_detached_return_hook_but_preserves_original_islands(self):
        for connected in (True, False):
            with self.subTest(connected_before_crop=connected):
                base, depth, normal, t, w = render_shaft(.25, 20., length=.12, width=.055)
                shaft = (abs(w + .008) < .0045) & (t > -.05) & (t < .052)
                return_leg = (abs(w - .016) < .0045) & (t > .022) & (t < .052)
                bridge = (w > -.0125) & (w < .0205) & (t > .047) & (t < .052)
                shape = (base > 0) & (shaft | return_leg | (bridge if connected else False))
                mask = shape.astype(np.uint8) * 255
                depth[~shape] = 0
                # Small return-leg depth bend; its lower part lies below
                # the numeric 1 cm threshold but belongs to the hooked tip.
                depth[return_leg & shape] += 2
                result = self.call(mask, depth)
                self.assert_result_consistent(result, mask)
                self.assertIsNotNone(result['plane'], result['status'])
                self.assertLess(angle_deg(result['plane']['n'], normal), 4.)
                lower_shaft = shaft & shape & (t < .035)
                self.assertGreater(np.count_nonzero(lower_shaft & (result['mask'] > 0)),
                                   .8 * lower_shaft.sum())
                lower_return = return_leg & shape & (t < .035)
                self.assertGreater(lower_return.sum(), 30)
                if connected:
                    self.assertFalse(np.any(return_leg & shape & (result['mask'] > 0)))
                    self.assertTrue(np.all(result['removed_mask'][lower_return] > 0))
                    self.assertFalse(np.any(return_leg & (result['support_mask'] > 0)))
                else:
                    # An existing segmentation island must not be dropped
                    # merely because it is shorter than the other island.
                    self.assertGreater(np.count_nonzero(lower_return & (result['mask'] > 0)),
                                       .6 * lower_return.sum())

    def test_inputs_are_not_mutated_and_result_is_deterministic(self):
        mask, depth, *_ = render_shaft(.25, 30.)
        original_mask, original_depth = mask.copy(), depth.copy()
        a, b = self.call(mask, depth), self.call(mask, depth)
        np.testing.assert_array_equal(mask, original_mask)
        np.testing.assert_array_equal(depth, original_depth)
        for key in ('mask', 'removed_mask'):
            np.testing.assert_array_equal(a[key], b[key])
        self.assertIsNotNone(a['plane'])
        np.testing.assert_allclose(a['plane']['n'], b['plane']['n'], atol=1e-8)
