"""Golden-output test for the augmentation module: a fixed generator seed
and a fixed fixture must reproduce the recorded sha256 digests.

The digests were computed once from the reference augmentation module.
``cv2.resize`` / ``GaussianBlur`` on float images differ at the bit level
between OpenCV releases, so the table is keyed by the OpenCV minor version;
an unknown version skips the float-path digests and still checks the mask
path, which is version-stable.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from simretract.data import seg_aug as sa  # noqa: E402

GOLDEN = {
    "4.13": {
        "corrupt_mask": "ef406b01a80ee4b3f2539d257cea431c3fb942fa93cff651c15e1f2979c2afb1",
        "random_fill": "97249a2fcc76901e51a8a121e668fa60249612a5f800e214cc3074c90d7faacb",
        "augment_depth": "97499d909bd38e394c8599c65604cefd2ad417940decea3a592836bd06e33046",
        "random_depth_fill": "81fbc6afb58f632272b3012a3d1ddae0a4c389408077986cbfafe5524d4a533b",
        "smooth_noise": "f8e5fa641efa0ebccaeb2194fdb3e9e7d97016f015ba8980f1173675d874473c",
    },
    "4.11": {
        "corrupt_mask": "ef406b01a80ee4b3f2539d257cea431c3fb942fa93cff651c15e1f2979c2afb1",
        "random_fill": "bb9c087ee8b459e1addfa008698a7786c9d7751f4fc777068c45261d398b7c59",
        "augment_depth": "97499d909bd38e394c8599c65604cefd2ad417940decea3a592836bd06e33046",
        "random_depth_fill": "81fbc6afb58f632272b3012a3d1ddae0a4c389408077986cbfafe5524d4a533b",
        "smooth_noise": "edf2736b28224944783640423e6b46dc5d470b705487dd4c333082302ade410d",
    },
}
VERSION_STABLE = ("corrupt_mask", "augment_depth", "random_depth_fill")


def fixture():
    h, w = 480, 640
    yy, xx = np.mgrid[:h, :w]
    ell = (((xx - 320) / 120.0) ** 2 + ((yy - 240) / 60.0) ** 2) < 1.0
    ring = (((xx - 320) / 30.0) ** 2 + ((yy - 240) / 20.0) ** 2) < 1.0
    mask = (ell & ~ring).astype(np.uint8)
    dep = np.zeros((h, w), np.float32)
    dep[ell] = np.clip(0.2 + 0.0004 * (xx[ell] - 200) + 0.0002 * (yy[ell] - 100), 0.05, 1.0)
    return mask, dep


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def outputs():
    mask, dep = fixture()
    out = {}
    rng = np.random.default_rng(1234)
    out["corrupt_mask"] = digest(np.stack([sa.corrupt_mask(mask, rng) for _ in range(6)]))
    rng = np.random.default_rng(1234)
    out["random_fill"] = digest(np.stack([sa.random_fill(480, 640, rng) for _ in range(3)]))
    rng = np.random.default_rng(1234)
    out["augment_depth"] = digest(np.stack([sa.augment_depth(dep, rng) for _ in range(3)]))
    rng = np.random.default_rng(1234)
    out["random_depth_fill"] = digest(np.stack([sa.random_depth_fill(480, 640, rng) for _ in range(3)]))
    rng = np.random.default_rng(7)
    out["smooth_noise"] = digest(sa.smooth_noise(120, 160, rng, 17.0))
    return out


def _golden():
    key = ".".join(cv2.__version__.split(".")[:2])
    return GOLDEN.get(key), key


def test_golden_outputs():
    got = outputs()
    table, key = _golden()
    if table is None:
        table = GOLDEN["4.13"]
        names = VERSION_STABLE
    else:
        names = tuple(table)
    bad = {k: (got[k], table[k]) for k in names if got[k] != table[k]}
    assert not bad, f"OpenCV {cv2.__version__}: digest mismatch {bad}"


def test_shapes_ranges_and_determinism():
    mask, dep = fixture()
    r1, r2 = np.random.default_rng(3), np.random.default_rng(3)
    m1, m2 = sa.corrupt_mask(mask, r1), sa.corrupt_mask(mask, r2)
    assert m1.dtype == np.uint8 and m1.shape == mask.shape and set(np.unique(m1)) <= {0, 1}
    assert np.array_equal(m1, m2)
    d = sa.augment_depth(dep, np.random.default_rng(4))
    assert d.shape == dep.shape and d.min() >= 0 and d.max() <= 1
    assert not d[dep <= 1e-6].any()  # invalid stays invalid
    f = sa.random_fill(480, 640, np.random.default_rng(5))
    assert f.dtype == np.uint8 and f.shape == (480, 640, 3)
    g = sa.random_depth_fill(480, 640, np.random.default_rng(6))
    assert g.dtype == np.float32 and g.shape == (480, 640) and 0 <= g.min() and g.max() <= 1


def test_clean_and_tiny_masks():
    mask, _ = fixture()
    tiny = np.zeros_like(mask)
    tiny[10:12, 10:15] = 1  # < 20 px: returned as is
    assert np.array_equal(sa.corrupt_mask(tiny, np.random.default_rng(0)), tiny)
    # over many draws about CLEAN_P of the outputs equal the input mask
    clean = sum(np.array_equal(sa.corrupt_mask(mask, np.random.default_rng(i)), mask) for i in range(200))
    assert 10 <= clean <= 60, clean
