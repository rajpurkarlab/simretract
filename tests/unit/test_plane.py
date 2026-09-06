"""Both plane fits must reproduce the reference implementations bit for bit.

The reference code below is the exact fitting logic the shipped checkpoints
were trained (sidecar) and benchmarked (served) with, copied verbatim so
the test does not depend on any external tree.
"""
from __future__ import annotations

import numpy as np
import pytest

from simretract.serve import plane as P

cv2 = pytest.importorskip("cv2")

# ---------------------------------------------------------------- oracles
_FY = 240.0 / np.tan(np.radians(58.0) / 2)
_CX, _CY = 320.0, 240.0


def ref_sidecar(dep_i, msk_i):
    m = (msk_i > 0) & (dep_i > 0)
    v, u = np.nonzero(m)
    if len(v) < 200:
        return np.zeros(3, np.float32), np.float32(0)
    z = dep_i[v, u].astype(np.float64) / 1000.0
    Pm = np.stack([(u - _CX) / _FY * z, (v - _CY) / _FY * z, z], 1)
    c = Pm.mean(0)
    _, s, Vt = np.linalg.svd(Pm - c, full_matrices=False)
    n = Vt[2]
    d = np.abs((Pm - c) @ n)
    keep = d < 3.0 * max(d.std(), 1e-6)
    if keep.sum() >= 100:
        c2 = Pm[keep].mean(0)
        _, s, Vt = np.linalg.svd(Pm[keep] - c2, full_matrices=False)
        n = Vt[2]
    if n[2] > 0:
        n = -n
    out = np.zeros(3, np.float32)
    out[:] = n
    conf = np.zeros(1, np.float32)
    conf[0] = s[1] / max(s[2], 1e-9)
    return out, conf[0]


def ref_top_cut(mask, frac):
    if frac <= 0:
        return mask
    v, u = np.nonzero(mask > 0)
    if len(v) < 300:
        return mask
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
        return mask
    out = np.zeros_like(mask)
    out[v[keep], u[keep]] = 255
    return out


def ref_fit_nc(dep, mask_in, plane_filter):
    m = (mask_in > 0) & (dep > 50) & (dep < 900)
    v, u = np.nonzero(m)
    if len(v) < 200:
        return None
    z = dep[v, u].astype(np.float64) / 1000.0
    if plane_filter:
        zm = np.median(z)
        mad = np.median(np.abs(z - zm))
        keep0 = np.abs(z - zm) <= max(5.0 * mad, 0.01)
        if keep0.sum() >= 200:
            u, v, z = u[keep0], v[keep0], z[keep0]
    Pm = np.stack([(u - 320.0) / _FY * z, (v - 240.0) / _FY * z, z], 1)
    c = Pm.mean(0)
    _, _, Vt = np.linalg.svd(Pm - c, full_matrices=False)
    n = Vt[2]
    d = np.abs((Pm - c) @ n)
    keep = d < 3.0 * max(d.std(), 1e-6)
    if keep.sum() >= 100:
        c = Pm[keep].mean(0)
        _, _, Vt = np.linalg.svd(Pm[keep] - c, full_matrices=False)
        n = Vt[2]
    if n[2] > 0:
        n = -n
    n = (n / np.linalg.norm(n)).astype(np.float32)
    return n, c


def ref_served(depth_mm, mask, plane_filter, top_cut, max_deg):
    dep = np.asarray(depth_mm)
    mask_in = np.asarray(mask)
    if plane_filter:
        dep = cv2.medianBlur(dep.astype(np.float32), 5)
    res = ref_fit_nc(dep, mask_in, plane_filter)
    tc = float(top_cut)
    if tc > 0:
        res_cut = ref_fit_nc(dep, ref_top_cut(mask_in, tc), plane_filter)
        if res is None:
            res = res_cut
        elif res_cut is not None:
            ang = np.degrees(np.arccos(np.clip(abs(float(res[0] @ res_cut[0])), 0.0, 1.0)))
            if ang <= max_deg:
                res = res_cut
    return res


# --------------------------------------------------------------- fixtures
def tilted_plane_depth(normal, z0=0.30, rng=None, noise_mm=0.0, flyers=0, ring=False):
    """Render a plane with the given (camera-frame) normal through (0, 0, z0)
    into a 480x640 uint16 depth map + an elliptical tool mask."""
    n = np.asarray(normal, np.float64)
    n = n / np.linalg.norm(n)
    yy, xx = np.mgrid[:480, :640]
    a = (xx - _CX) / _FY
    b = (yy - _CY) / _FY
    # n . (a z, b z, z) = n . (0, 0, z0)  ->  z = n_z z0 / (n_x a + n_y b + n_z)
    z = n[2] * z0 / (n[0] * a + n[1] * b + n[2])
    dep = (z * 1000.0)
    if rng is not None and noise_mm > 0:
        dep = dep + rng.normal(0, noise_mm, dep.shape)
    dep = np.clip(dep, 1, 65535).astype(np.uint16)
    mask = ((((xx - 320) / 150.0) ** 2 + ((yy - 250) / 70.0) ** 2) < 1.0).astype(np.uint8) * 255
    if ring:
        hole = (((xx - 320) / 30.0) ** 2 + ((yy - 250) / 20.0) ** 2) < 1.0
        mask[hole] = 0
        dep[hole] = 1500
    if flyers and rng is not None:
        v, u = np.nonzero(mask)
        pick = rng.choice(len(v), flyers, replace=False)
        dep[v[pick], u[pick]] = rng.integers(20, 60, flyers).astype(np.uint16)
    return dep, mask


CASES = [
    dict(normal=(0.0, 0.0, -1.0)),
    dict(normal=(0.3, -0.2, -1.0), noise_mm=2.0),
    dict(normal=(-0.5, 0.4, -1.0), noise_mm=3.0, flyers=400, ring=True),
    dict(normal=(0.1, 0.9, -0.6), noise_mm=1.0, flyers=100),
    dict(normal=(0.6, 0.1, -0.5), noise_mm=4.0, ring=True),
]


@pytest.mark.parametrize("case", CASES)
def test_sidecar_matches_reference(case):
    rng = np.random.default_rng(11)
    dep, mask = tilted_plane_depth(rng=rng, **case)
    n_ref, c_ref = ref_sidecar(dep, mask)
    n_got, c_got = P.fit_plane_sidecar(dep, mask)
    assert n_got.dtype == np.float32
    assert np.array_equal(n_ref, n_got)
    assert np.float32(c_got) == c_ref
    # geometry: within 0.5 deg of the truth, oriented toward the camera
    truth = np.asarray(case["normal"], np.float64)
    truth /= np.linalg.norm(truth)
    ang = np.degrees(np.arccos(np.clip(abs(float(n_got @ truth)), 0, 1)))
    assert ang < 0.5, ang
    assert n_got[2] < 0


def test_sidecar_degenerate_frame():
    dep = np.full((480, 640), 300, np.uint16)
    mask = np.zeros((480, 640), np.uint8)
    mask[100:110, 100:115] = 255  # 150 px < 200
    n, c = P.fit_plane_sidecar(dep, mask)
    assert np.array_equal(n, np.zeros(3, np.float32)) and c == 0.0
    n_ref, c_ref = ref_sidecar(dep, mask)
    assert np.array_equal(n, n_ref) and c == c_ref


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("plane_filter", [True, False])
@pytest.mark.parametrize("top_cut", [0.0, 0.10, 0.30])
def test_served_matches_reference(case, plane_filter, top_cut):
    rng = np.random.default_rng(5)
    dep, mask = tilted_plane_depth(rng=rng, **case)
    ref = ref_served(dep, mask, plane_filter, top_cut, 20.0)
    got = P.fit_plane_served(dep, mask, plane_filter=plane_filter, top_cut=top_cut, top_cut_max_deg=20.0)
    assert (ref is None) == (got is None)
    if ref is not None:
        assert np.array_equal(ref[0], got[0])
        assert np.array_equal(ref[1], got[1])
        assert got[0].dtype == np.float32 and got[0][2] < 0


def test_served_top_cut_guard_and_masks():
    rng = np.random.default_rng(9)
    dep, mask = tilted_plane_depth((0.2, -0.3, -1.0), rng=rng, noise_mm=2.0)
    # the cut mask is identical to the reference for a range of fractions
    for frac in (0.0, 0.05, 0.10, 0.5, 0.95):
        assert np.array_equal(P.plane_top_cut(mask, frac), ref_top_cut(mask, frac))
    # a tiny max-deg guard forces the uncut plane whenever the cut deviates
    uncut = P.fit_plane_served(dep, mask, top_cut=0.0)
    guarded = P.fit_plane_served(dep, mask, top_cut=0.5, top_cut_max_deg=0.0)
    ref = ref_served(dep, mask, True, 0.5, 0.0)
    assert np.array_equal(guarded[0], ref[0])
    assert np.array_equal(guarded[0], uncut[0]) or np.allclose(guarded[0], uncut[0])


def test_served_depth_window_and_none():
    dep = np.full((480, 640), 950, np.uint16)  # outside the 50..900 mm window
    mask = np.zeros((480, 640), np.uint8)
    mask[200:300, 200:400] = 255
    assert P.fit_plane_served(dep, mask) is None
    assert ref_served(dep, mask, True, 0.0, 20.0) is None
    dep[:] = 400
    assert P.fit_plane_served(dep, mask) is not None
