"""Per-seed scene draws (pure numpy, importable without the simulator).

This module is the single source of truth for everything a seed decides:
the presented tool pose, the hand and tissue occluder geometry, and the
launcher-side predictions used to pre-screen seeds before any simulation
runs. The ORDER of random draws is a contract shared by the recorders, the
sim benchmark and the assembly tools: a seed must reproduce the same scene
everywhere, so never reorder or insert draws.

Two occluder hand styles exist:

* ``"train"`` - the hand distribution used for every training set: fist
  probability 0.75 and the hand placed higher on the handle
  (z in U(-0.12, -0.065) along the tool axis).
* ``"test"`` - the distribution the frozen test seeds were drawn from: fist
  probability 0.5 and z in U(-0.12, -0.05).

The style is an explicit argument everywhere; nothing reads it from the
environment.
"""

from __future__ import annotations

from typing import Any

import numpy as np

HAND_STYLES = ("train", "test")

HAND_TONES = [(0.85, 0.72, 0.58), (0.80, 0.60, 0.50), (0.45, 0.62, 0.80),
              (0.40, 0.64, 0.52), (0.86, 0.86, 0.86)]
TISSUE_TONES = [(0.72, 0.38, 0.38), (0.82, 0.55, 0.55), (0.58, 0.26, 0.26),
                (0.42, 0.62, 0.52), (0.75, 0.68, 0.60)]

# RNG stream tags: the occluder and lighting streams are decoupled from the
# pose stream so changing one draw never shifts the others.
OCCLUDER_STREAM_TAG = 0x6F75636C
LIGHTING_STREAM_TAG = 0x4C494748
PERTURB_STREAM_TAG = 0x50455254


def check_hand_style(hand_style: str) -> str:
    """Validate a hand style string and return it."""
    if hand_style not in HAND_STYLES:
        raise ValueError(f"hand_style must be one of {HAND_STYLES}, got {hand_style!r}")
    return hand_style


def sample_presented_pose_arrays(rng: np.random.Generator):
    """Draw the presented pose as raw numbers: (pos(3), theta, phi, psi).

    Kept scipy-free so the module imports anywhere; ``presented_rotation``
    turns the angles into a rotation object.
    """
    pos = np.array([rng.uniform(0.35, 0.50), rng.uniform(-0.12, 0.12),
                    rng.uniform(0.15, 0.30)])
    theta = rng.uniform(0.0, 1.4)
    phi = rng.uniform(0.0, 2 * np.pi)
    psi = rng.uniform(-np.pi, np.pi)
    return pos, theta, phi, psi


def presented_rotation(theta: float, phi: float, psi: float):
    """Rotation of the presented tool: tilt ``theta`` about a random
    horizontal axis on top of a 180 degree x-flip and a spin ``psi``."""
    from scipy.spatial.transform import Rotation

    base = Rotation.from_euler("x", np.pi)
    return (Rotation.from_rotvec(theta * np.array([np.cos(phi), np.sin(phi), 0]))
            * base * Rotation.from_euler("z", psi))


def sample_presented_pose(rng: np.random.Generator):
    """Draw the presented pose: (pos(3), Rotation, tilt_deg).

    Consumes exactly the draws of ``sample_presented_pose_arrays`` so the
    launcher predictions and the simulator agree.
    """
    pos, theta, phi, psi = sample_presented_pose_arrays(rng)
    return pos, presented_rotation(theta, phi, psi), float(np.degrees(theta))


def presented_pose_for_seed(seed: int):
    """(pos, Rotation) of the presented tool for ``seed`` (first draws of the
    seed stream). Used by the assembly tools to express a stored world-frame
    grasp point in the tool frame."""
    pos, rot, _tilt = sample_presented_pose(np.random.default_rng(int(seed)))
    return pos, rot


def sample_occluder_spec(seed: int, tip_local, hand_style: str) -> dict[str, Any]:
    """Hand occluder + tissue tip anchor for ``seed`` (own RNG stream).

    The hand is bimodal (few-finger pinch vs fist) and sits on the handle
    below the tool center; the tip anchor rides the blade tip. Returns a
    spec dict of visual-only sub-shapes in the tool frame.
    """
    check_hand_style(hand_style)
    rng = np.random.default_rng(seed ^ OCCLUDER_STREAM_TAG)

    def _subshapes(rng, size, shape, base_pos, quat):
        subs = []
        if shape == "blob":
            n = int(rng.integers(2, 5))
            for i in range(n):
                kind = ["box", "ellipsoid", "sphere"][int(rng.integers(0, 3))]
                off = np.zeros(3) if i == 0 else rng.uniform(-0.45, 0.45, 3) * size
                q = rng.standard_normal(4)
                q /= np.linalg.norm(q)
                if kind == "sphere":
                    s = np.full(3, rng.uniform(0.35, 0.7) * np.mean(size))
                else:
                    s = rng.uniform(0.35, 0.85, 3) * size
                subs.append({"kind": kind, "pos": (base_pos + off).tolist(),
                             "quat_wxyz": q.tolist(), "half": s.tolist()})
        else:
            subs.append({"kind": shape, "pos": base_pos.tolist(),
                         "quat_wxyz": quat.tolist(), "half": size.tolist()})
        return subs

    train_hand = hand_style == "train"
    if rng.random() < (0.25 if train_hand else 0.5):  # few-fingers grip
        b = rng.uniform(0.012, 0.022)
        h_size = np.array([b, b * rng.uniform(0.9, 1.8),
                           rng.uniform(0.008, 0.018)])
    else:                    # fist grip
        b = rng.uniform(0.025, 0.045)
        h_size = np.array([b, b * rng.uniform(0.7, 1.2),
                           rng.uniform(0.020, 0.040)])
    _zlo, _zhi = (-0.12, -0.065) if train_hand else (-0.12, -0.05)
    h_pos = np.array([rng.uniform(-0.015, 0.015), rng.uniform(-0.015, 0.015),
                      rng.uniform(_zlo, _zhi)])
    h_shape = ["box", "ellipsoid", "blob"][int(rng.integers(0, 3))]
    tone = np.clip(np.array(HAND_TONES[int(rng.integers(0, len(HAND_TONES)))])
                   + rng.uniform(-0.07, 0.07, 3), 0, 1)
    h_quat = rng.standard_normal(4)
    h_quat /= np.linalg.norm(h_quat)
    hand = {"subs": _subshapes(rng, h_size, h_shape, h_pos, h_quat),
            "rgba": tone.tolist(), "z_along_tool": float(h_pos[2])}

    t = rng.uniform(0.010, 0.028)
    t_size = np.array([t, t * rng.uniform(0.7, 1.3), t * rng.uniform(0.6, 1.1)])
    t_off = np.array([rng.uniform(-0.010, 0.010), rng.uniform(-0.010, 0.010),
                      rng.uniform(-0.012, 0.014)])
    t_shape = ["ellipsoid", "blob"][int(rng.integers(0, 2))]
    t_tone = np.clip(
        np.array(TISSUE_TONES[int(rng.integers(0, len(TISSUE_TONES)))])
        + rng.uniform(-0.06, 0.06, 3), 0, 1)
    t_quat = rng.standard_normal(4)
    t_quat /= np.linalg.norm(t_quat)
    tip_anchor = {"subs": _subshapes(rng, t_size, t_shape,
                                     np.asarray(tip_local) + t_off, t_quat),
                  "rgba": t_tone.tolist()}
    return {"hand": hand, "tip": tip_anchor}


def predict_seed_bins(seed: int, tip_local, hand_style: str) -> tuple[float, float]:
    """Launcher-side pre-screen without the simulator: (tilt_deg, hand_z)."""
    rng = np.random.default_rng(seed)
    _pos, theta, _phi, _psi = sample_presented_pose_arrays(rng)
    spec = sample_occluder_spec(seed, tip_local, hand_style)
    return float(np.degrees(theta)), spec["hand"]["z_along_tool"]


def hand_blocked_interval(spec: dict[str, Any], gate_m: float) -> list[tuple[float, float]]:
    """Tool-axis z intervals a hand sub-shape blocks for grasp centers.

    The recorder rejects a candidate whose center is closer than
    ``max(half) + gate_m`` to any hand sub-shape position; projected on the
    tool axis (the x/y offsets of candidates are small) that is the
    interval ``[z - r, z + r]`` per sub-shape. Used to skip seeds whose hand
    covers a target grasp-height bin entirely.
    """
    out = []
    for s in spec["hand"]["subs"]:
        r = float(max(s["half"])) + float(gate_m)
        z = float(s["pos"][2])
        out.append((z - r, z + r))
    return out


def bin_fully_blocked(intervals: list[tuple[float, float]], z_lo: float, z_hi: float) -> bool:
    """True when the union of ``intervals`` covers ``[z_lo, z_hi]``."""
    cover = z_lo
    for a, b in sorted(intervals):
        if a > cover:
            break
        cover = max(cover, b)
        if cover >= z_hi:
            return True
    return cover >= z_hi
