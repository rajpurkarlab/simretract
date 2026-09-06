"""Seed-draw contract of simretract.sim.sampling.

The golden values were computed once from the reference implementation of
the sampling module for both hand styles (its legacy ``ERA4_HAND`` switch
unset -> ``test`` and ``=1`` -> ``train``). Set
``SIMRETRACT_REFERENCE_SAMPLING_PY`` to that module's path to additionally
compare full occluder specs live.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

from simretract.sim import sampling

TIP = [-0.00677, 0.0012, 0.12624]
SEEDS = (200005, 200350, 400851, 601001)

# per style: seed -> (tilt_deg, hand_z, n_hand, hand0_kind, hand0_pos, hand0_half,
#                     hand_rgba, n_tip, tip0_kind, tip0_pos)
GOLDEN = {
    "test": {
        200005: (67.43388, -0.060221305, 1, "box", [-0.014593576, -0.000801203, -0.060221305],
                 [0.02857592, 0.033161732, 0.023133399], [0.45940414, 0.619561472, 0.859598816],
                 4, "sphere", [-0.002261961, 0.005848412, 0.125466305]),
        200350: (72.613917, -0.056418481, 1, "ellipsoid", [0.005765436, -0.01225709, -0.056418481],
                 [0.017193165, 0.026995536, 0.009598983], [0.468347762, 0.68601656, 0.45435243],
                 4, "box", [-0.015307645, 0.000522154, 0.137255849]),
        400851: (23.782462, -0.066995645, 2, "sphere", [-0.008943171, -0.005205045, -0.066995645],
                 [0.011206888, 0.011206888, 0.011206888], [0.375020529, 0.698831054, 0.488263482],
                 2, "box", [-0.010633024, -0.003962529, 0.131170595]),
        601001: (27.091299, -0.072770274, 1, "box", [-0.002706239, 0.006382664, -0.072770274],
                 [0.02595029, 0.029602035, 0.037658392], [0.782588783, 0.626160604, 0.46004522],
                 3, "ellipsoid", [-0.013522767, 0.009741058, 0.120722209]),
    },
    "train": {
        200005: (67.43388, -0.073031026, 1, "box", [-0.014593576, -0.000801203, -0.073031026],
                 [0.02857592, 0.033161732, 0.023133399], [0.45940414, 0.619561472, 0.859598816],
                 4, "sphere", [-0.002261961, 0.005848412, 0.125466305]),
        200350: (72.613917, -0.070043092, 1, "ellipsoid", [0.005765436, -0.01225709, -0.070043092],
                 [0.03538633, 0.037944602, 0.023197967], [0.468347762, 0.68601656, 0.45435243],
                 4, "box", [-0.015307645, 0.000522154, 0.137255849]),
        400851: (23.782462, -0.078353721, 2, "sphere", [-0.008943171, -0.005205045, -0.078353721],
                 [0.020962584, 0.020962584, 0.020962584], [0.375020529, 0.698831054, 0.488263482],
                 2, "box", [-0.010633024, -0.003962529, 0.131170595]),
        601001: (27.091299, -0.082890929, 1, "box", [-0.002706239, 0.006382664, -0.082890929],
                 [0.02595029, 0.029602035, 0.037658392], [0.782588783, 0.626160604, 0.46004522],
                 3, "ellipsoid", [-0.013522767, 0.009741058, 0.120722209]),
    },
}


@pytest.mark.parametrize("style", ["train", "test"])
@pytest.mark.parametrize("seed", SEEDS)
def test_occluder_spec_matches_golden(style, seed):
    tilt, z, n_hand, hk, hpos, hhalf, hrgba, n_tip, tk, tpos = GOLDEN[style][seed]
    spec = sampling.sample_occluder_spec(seed, TIP, style)
    t_pred, z_pred = sampling.predict_seed_bins(seed, TIP, style)
    assert t_pred == pytest.approx(tilt, abs=1e-5)
    assert z_pred == pytest.approx(z, abs=1e-8)
    hand, tip = spec["hand"], spec["tip"]
    assert hand["z_along_tool"] == pytest.approx(z, abs=1e-8)
    assert len(hand["subs"]) == n_hand and hand["subs"][0]["kind"] == hk
    assert np.allclose(hand["subs"][0]["pos"], hpos, atol=1e-8)
    assert np.allclose(hand["subs"][0]["half"], hhalf, atol=1e-8)
    assert np.allclose(hand["rgba"], hrgba, atol=1e-8)
    assert len(tip["subs"]) == n_tip and tip["subs"][0]["kind"] == tk
    assert np.allclose(tip["subs"][0]["pos"], tpos, atol=1e-8)


def test_hand_style_is_explicit():
    with pytest.raises(TypeError):
        sampling.sample_occluder_spec(200005, TIP)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        sampling.sample_occluder_spec(200005, TIP, "era4")


def test_spec_is_deterministic_and_style_independent_where_expected():
    a = sampling.sample_occluder_spec(200350, TIP, "train")
    b = sampling.sample_occluder_spec(200350, TIP, "train")
    assert a == b
    c = sampling.sample_occluder_spec(200350, TIP, "test")
    # the tip anchor draws come after the hand draws in the same stream, so
    # the styles only differ in the hand size/height, not in the RNG order
    assert c["tip"]["subs"][0]["pos"] == a["tip"]["subs"][0]["pos"]
    assert c["hand"]["z_along_tool"] != a["hand"]["z_along_tool"]


def test_presented_pose_arrays_and_rotation():
    rng = np.random.default_rng(200005)
    pos, theta, phi, psi = sampling.sample_presented_pose_arrays(rng)
    assert 0.35 <= pos[0] <= 0.50 and -0.12 <= pos[1] <= 0.12 and 0.15 <= pos[2] <= 0.30
    assert 0.0 <= theta <= 1.4
    pos2, rot, tilt = sampling.sample_presented_pose(np.random.default_rng(200005))
    assert np.allclose(pos, pos2) and tilt == pytest.approx(np.degrees(theta))
    # tilt = angle between the tool axis and the vertical
    z_axis = rot.apply([0.0, 0.0, 1.0])
    assert np.degrees(np.arccos(abs(z_axis[2]))) == pytest.approx(tilt, abs=1e-6)
    p3, r3 = sampling.presented_pose_for_seed(200005)
    assert np.allclose(p3, pos) and np.allclose(r3.as_matrix(), rot.as_matrix())


def test_bin_blocking_helpers():
    assert sampling.bin_fully_blocked([(-0.06, -0.02)], -0.05, -0.025)
    assert not sampling.bin_fully_blocked([(-0.06, -0.03)], -0.05, -0.025)
    assert sampling.bin_fully_blocked([(-0.06, -0.035), (-0.04, 0.0)], -0.05, -0.025)
    assert not sampling.bin_fully_blocked([(-0.06, -0.045), (-0.04, 0.0)], -0.05, -0.025)
    assert not sampling.bin_fully_blocked([], -0.05, 0.05)
    spec = sampling.sample_occluder_spec(200005, TIP, "train")
    iv = sampling.hand_blocked_interval(spec, 0.01)
    assert len(iv) == len(spec["hand"]["subs"])
    r = max(spec["hand"]["subs"][0]["half"]) + 0.01
    assert iv[0] == pytest.approx((spec["hand"]["subs"][0]["pos"][2] - r,
                                   spec["hand"]["subs"][0]["pos"][2] + r))


@pytest.mark.skipif(not os.environ.get("SIMRETRACT_REFERENCE_SAMPLING_PY"),
                    reason="reference sampling module path not provided")
@pytest.mark.parametrize("style,era4", [("test", "0"), ("train", "1")])
def test_full_spec_matches_reference_module(style, era4, monkeypatch):
    path = os.environ["SIMRETRACT_REFERENCE_SAMPLING_PY"]
    spec_mod = importlib.util.spec_from_file_location("reference_sampling", path)
    mod = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(mod)
    monkeypatch.setenv("ERA4_HAND", era4)
    for seed in SEEDS:
        assert mod.sample_occluder_spec(seed, TIP) == sampling.sample_occluder_spec(seed, TIP, style)
        assert mod.predict_seed_bins(seed, TIP) == sampling.predict_seed_bins(seed, TIP, style)
