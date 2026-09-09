"""Synthetic raw episodes for tests and smoke runs.

The frames are geometric (an elliptical "tool" that drifts across the image,
a fixed "gripper" blob, a tilted depth plane) and the trajectories follow the
recorder's conventions (7-D state, a linear gripper close, a stall segment
that the arc-length rule collapses), so the whole data pipeline - cache
builder, plane sidecar, alignment gate, trainer and policy server - can be
exercised without a simulator. These synthetic fixtures follow the
raw-episode contract in ``docs/DATA_FORMAT.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FRAME_H = 480
FRAME_W = 640
#: Frames of a synthetic strict episode and of a recovery draw.
T_STRICT = 60
T_RECOVERY = 90
#: Last frame of the synthetic perturbation window of a recovery draw.
PERTURB_END = 30


def trajectory(T: int, rng: np.random.Generator, close_at: int) -> tuple[np.ndarray, np.ndarray]:
    """Joints drift 3e-3 rad/frame (one quantum) with a stall in the middle;
    the carriage closes linearly until ``close_at`` and then holds."""
    t = np.arange(T, dtype=np.float32)
    joints = 0.1 * np.sin(0.05 * t)[:, None] + 3e-3 * t[:, None] * np.linspace(0.5, 1.5, 6)[None, :]
    joints[T // 3:T // 3 + 5] = joints[T // 3]  # a stall: skipped by the arc-length rule
    grip = np.where(t < close_at, 0.044 - (0.044 - 0.012) * t / close_at, 0.012).astype(np.float32)
    states = np.concatenate([joints.astype(np.float32), grip[:, None]], 1)
    actions = states + rng.normal(0, 1e-3, states.shape).astype(np.float32)
    return states, actions


def write_episode(root: Path, name: str, seed: int, T: int, rng: np.random.Generator, *,
                  info: dict | None = None, empty_mask: bool = False,
                  close_at: int | None = None) -> np.ndarray:
    """Write one synthetic episode directory ``root/name`` and return its states."""
    from PIL import Image

    H, W = FRAME_H, FRAME_W
    d = Path(root) / name
    d.mkdir(parents=True)
    states, actions = trajectory(T, rng, close_at or int(T * 0.6))
    yy, xx = np.mgrid[:H, :W]
    for i in range(T):
        cx = 320 + 2 * i
        tool = (((xx - cx) / 120.0) ** 2 + ((yy - 240) / 50.0) ** 2) < 1.0
        grip = (((xx - 320) / 30.0) ** 2 + ((yy - 400) / 40.0) ** 2) < 1.0
        rgb = np.zeros((H, W, 3), np.uint8)
        rgb[..., 0] = (xx // 3) % 256
        rgb[..., 1] = (yy // 2 + i) % 256
        rgb[tool] = (200, 40 + i, 90)
        dep = (300 + 0.2 * (xx - 320) + 0.1 * (yy - 240) + i).astype(np.uint16)
        dep[~tool & ~grip] = 900
        Image.fromarray(rgb).save(d / f"rgb_{i:04d}.png")
        Image.fromarray(dep).save(d / f"depth_{i:04d}.png")
        Image.fromarray(np.zeros((H, W), np.uint8) if empty_mask
                        else tool.astype(np.uint8) * 255).save(d / f"mask_{i:04d}.png")
        Image.fromarray(grip.astype(np.uint8) * 255).save(d / f"gmask_{i:04d}.png")
        Image.fromarray(np.zeros((H, W), np.uint8)).save(d / f"omask_{i:04d}.png")
    verdict = {"battery_skipped": True, "strict_pass": None}
    np.savez(d / "meta.npz", states=states, actions=actions,
             grasp_pose=np.array([0.5, -0.05, 0.2, 0.8, -0.5, 0.15, 0.26], np.float32),
             seed=np.int64(seed), verdict=json.dumps(verdict), info=json.dumps(info or {"gap_mm": 4.9}))
    return states


def write_synthetic_raw(root: Path, n_strict: int = 2, n_recovery: int = 1, *,
                        seed0: int = 200011, rng_seed: int = 0) -> dict[str, Path]:
    """Write ``root/strict`` (``n_strict`` seeds) and ``root/recovery``
    (one perturbation draw for each of the first ``n_recovery`` seeds).

    Seeds are ``seed0 + 11 * k`` (all above the cache builder's ``seed_min``
    and outside the shipped held-out pool). Returns ``{"strict": dir,
    "recovery": dir}``.
    """
    root = Path(root)
    rng = np.random.default_rng(rng_seed)
    strict, recovery = root / "strict", root / "recovery"
    seeds = [seed0 + 11 * k for k in range(n_strict)]
    for s in seeds:
        write_episode(strict, f"ep_{s}", s, T_STRICT, rng)
    for s in seeds[:n_recovery]:
        write_episode(recovery, f"ep_{s}_p0", s, T_RECOVERY, rng, close_at=70,
                      info={"perturb_win": {"start": 0, "end": PERTURB_END},
                            "perturb": {"frac": 0.3}, "gap_mm": 4.9})
    return {"strict": strict, "recovery": recovery}
