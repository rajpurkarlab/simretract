"""Bitwise alignment gate between a cache set and the trainer's dataset.

Run before training: it builds the trainer's ``DS`` on a set with
augmentation and dropout off and, for random kept positions in random
episodes, checks against the raw cache that

1. kept frame indices are strictly increasing and inside
   ``[episode start, close_cutoff)``;
2. every skipped frame is genuinely static (the accumulated motion between
   consecutive kept frames, excluding the triggering step, stays below one
   quantum);
3. the masked RGB stream equals ``letterbox(rgb * (tool | gripper))``;
4. the depth stream equals ``letterbox(clip(depth_m / 1.5) * visible)``
   (uint8 or float32 path, matching the trainer's depth mode);
5. the state equals the raw state, followed by the grasp pose (``full``) or
   the normalised plane normal (``qpos_plane``);
6. action chunk row ``j`` equals ``action[kept[p + j]]`` with the last frame
   repeated and ``action_is_pad`` set beyond the episode end.

The trainer symbols are imported lazily from ``simretract.train.train`` so
this module stays importable without torch.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

H = 50
"""Action chunk length checked (the production horizon)."""


def _trainer() -> Any:
    from simretract.train import train as trainer

    return trainer


def _make_dataset(trainer: Any, sets: list[Any], mode: str, state_mode: str, float_depth: bool, nogrip: bool) -> Any:
    """The trainer's DS with every augmentation and dropout switch off."""
    knobs = trainer.Knobs(float_depth=float_depth, nogrip=nogrip, no_aug=True, chunk_h=H,
                          drop_mask_p=0.0, drop_plane_p=0.0)
    ds = trainer.DS(sets, mode, train=True, resample=True, norm=None, state_mode=state_mode, knobs=knobs)
    ds.train = False  # deterministic: no augmentation, no dropout, clean mask
    return ds


def _expected_plane(n_cam: np.ndarray) -> np.ndarray:
    n = np.asarray(n_cam, np.float64)
    nn = np.linalg.norm(n)
    if nn < 0.5:
        return np.zeros(3, np.float32)
    return (n / nn).astype(np.float32)


def verify_alignment(
    cache: str | Path,
    set_names: list[str],
    *,
    n_episodes: int = 6,
    samples_per_episode: int = 5,
    seed: int = 3,
    mode: str = "masked_rgbd",
    state_mode: str = "full",
    float_depth: bool = False,
    nogrip: bool = False,
) -> bool:
    """Run the six checks on ``n_episodes`` random episodes of the first set.

    Returns True when every episode passes; details are logged per episode.
    """
    trainer = _trainer()
    cache = Path(cache)
    JQ, GQ = trainer.JQ, trainer.GQ
    sets = [trainer.Cached(cache, n) for n in set_names]
    ds = _make_dataset(trainer, sets, mode, state_mode, float_depth, nogrip)
    rng = np.random.default_rng(seed)
    s = sets[0]
    fails = 0
    all_eps = sorted({e for (si, e, _) in ds.index if si == 0})
    eps = rng.choice(all_eps, min(n_episodes, len(all_eps)), replace=False)
    for e in eps:
        arr = ds.kept[(0, e)]
        a0, a1 = s.episodes[e]
        cut = trainer.close_cutoff(s, a0, a1)
        ok_rng = bool(np.all(np.diff(arr) > 0) and arr[0] >= a0 and arr[-1] < cut)
        ok_skip = True
        for ga, gb in zip(arr[:-1], arr[1:]):
            if gb - ga <= 1:
                continue
            d = np.abs(np.diff(s.state[ga:gb], axis=0))
            acc = np.maximum(d[:, :6].max(1) / JQ, d[:, 6] / GQ).sum()
            if acc >= 1.0 + 1e-6:
                ok_skip = False
                break
        pos = {int(arr[p]): k for k, (si, ee, p) in enumerate(ds.index) if si == 0 and ee == e}
        ok_img = ok_dep = ok_st = ok_act = True
        for p in rng.choice(len(arr), min(samples_per_episode, len(arr)), replace=False):
            p = int(p)
            i = int(arr[p])
            smp = ds[pos[i]]
            gmk = np.zeros_like(s.gmk[i]) if nogrip else s.gmk[i]
            vis = (s.msk[i] > 0) | (gmk > 0)
            exp_rgb = trainer.letterbox(np.ascontiguousarray(s.rgb[i]) * vis[..., None])
            got_rgb = (smp["observation.images.wrist"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            ok_img &= np.array_equal(exp_rgb, got_rgb)
            if "observation.images.mask" in smp:
                dep = np.clip(s.dep[i].astype(np.float32) / 1000.0 / 1.5, 0, 1) * vis
                got = smp["observation.images.mask"].permute(1, 2, 0).numpy()
                if float_depth:
                    exp_dep = trainer.letterbox(np.repeat(dep.astype(np.float32)[:, :, None], 3, 2))
                    ok_dep &= np.array_equal(exp_dep, got.astype(np.float32))
                else:
                    exp_dep = trainer.letterbox(np.repeat((dep * 255).astype(np.uint8)[:, :, None], 3, 2))
                    ok_dep &= np.array_equal(exp_dep, (got * 255).round().astype(np.uint8))
            if state_mode == "qpos_plane":
                exp_st = np.concatenate([s.state[i].astype(np.float32)[:7], _expected_plane(s.plane[i])])
            elif state_mode == "qpos":
                exp_st = s.state[i].astype(np.float32)
            else:
                exp_st = np.concatenate([s.state[i], s.gpose[i]])
            ok_st &= np.allclose(smp["observation.state"].numpy(), exp_st)
            act = smp["action"].numpy()
            pad = smp["action_is_pad"].numpy()
            for j in range(H):
                gi = arr[p + j] if p + j < len(arr) else arr[-1]
                ok_act &= np.allclose(act[j], s.action[gi])
                ok_act &= bool(pad[j]) == (p + j >= len(arr))
        stat = all([ok_rng, ok_skip, ok_img, ok_dep, ok_st, ok_act])
        fails += 0 if stat else 1
        log.info(
            "ep %d: range=%s skip-static=%s rgb=%s depth=%s state=%s actions=%s",
            int(e), ok_rng, ok_skip, ok_img, ok_dep, ok_st, ok_act,
        )
    log.info("ALIGN %s", "PASS" if fails == 0 else f"FAIL ({fails} episodes)")
    return fails == 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bitwise alignment gate between a cache set and the trainer dataset.")
    p.add_argument("--cache", default=os.environ.get("SIMRETRACT_CACHE", os.environ.get("CACHE")),
                   required="SIMRETRACT_CACHE" not in os.environ and "CACHE" not in os.environ,
                   help="cache directory (env SIMRETRACT_CACHE or CACHE)")
    p.add_argument("--sets", default=os.environ.get("SETS", "strict"), help="comma list of set names; the first is checked")
    p.add_argument("--episodes", type=int, default=6, help="random episodes to check")
    p.add_argument("--mode", default="masked_rgbd", help="trainer input mode")
    p.add_argument("--state", default="qpos_plane", choices=("full", "qpos", "qpos_plane"), help="trainer state mode")
    p.add_argument("--float-depth", dest="float_depth", action="store_true", default=os.environ.get("FLOAT_DEPTH", "1") == "1",
                   help="float32 depth stream (production; env FLOAT_DEPTH)")
    p.add_argument("--no-float-depth", dest="float_depth", action="store_false", help="uint8 depth round-trip")
    p.add_argument("--nogrip", dest="nogrip", action="store_true", default=os.environ.get("NOGRIP", "1") == "1",
                   help="gripper mask zeroed (production; env NOGRIP)")
    p.add_argument("--with-grip", dest="nogrip", action="store_false", help="gripper mask visible")
    p.add_argument("--seed", type=int, default=3)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    a = build_parser().parse_args(argv)
    ok = verify_alignment(
        a.cache, a.sets.split(","), n_episodes=a.episodes, seed=a.seed, mode=a.mode,
        state_mode=a.state, float_depth=a.float_depth, nogrip=a.nogrip,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
