"""SimRetract trainer: one entry point for every policy row.

Data path (identical for every policy):
  frame cache (``simretract-build-cache``) -> ``Cached`` memmaps -> ``DS``
  (close-cutoff trim, arc-length kept frames, masked-input construction,
  ``seg_aug`` train-time augmentation, plane-normal state, late-trajectory
  dropout) -> z-scored state/action chunks -> policy.forward.

Policy switch (``--policy``): pi0 (default), smolvla, act, fastwam, pi05,
pi0_fast.  Ablation switch (``--ablation``) and data regime (``--data``)
map onto the input mode, state mode and cache sets exactly as the reference
rows; ``--preset`` bundles the two with the policy (see ``presets.py``).

Checkpoint contract (consumed by ``simretract-serve`` / ``simretract-eval-sim``):
  runs/<name>/norm.json                       normalization + regime record
  runs/<name>/final/{config.json, model.safetensors, norm.json[, context.pt]}
  runs/<name>/step<N>/...                     optional snapshots
  runs/<name>/{config.json, model.safetensors} symlinks into final/ so the
  run directory itself is loadable.

Argument parsing, presets and the dataset code import without lerobot;
lerobot and transformers are imported lazily by ``build_policy``.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

try:  # argument parsing / preset resolution do not need torch
    import torch
except ImportError:  # pragma: no cover - exercised only in torch-less envs
    torch = None  # type: ignore[assignment]

from simretract.train import presets as _presets

log = logging.getLogger("simretract.train")

TASK = "take over the tool from the operator and hold its pose"
RES = 224
JQ = 2.5e-3   # joint motion quantum, rad (~one normal moving step)
GQ = 2.5e-4   # gripper motion quantum, m
CLOSE_MARGIN = 25
VAL_FRAC = 0.05

POLICIES = ("pi0", "smolvla", "act", "fastwam", "pi05", "pi0_fast")
INPUT_MODES = ("full", "masked_rgb", "masked_rgbd", "full_rgbd",
               "masked_rgb_nodepth")
STATE_MODES = ("qpos", "qpos_plane", "full")
STATE_DIM = {"qpos": 7, "qpos_plane": 10, "full": 14}
DATA_MODES = ("full", "strict_only", "half")
ABLATIONS = ("none", "no_plane", "no_mask", "no_depth", "no_aug")
DEFAULT_SETS = ("strict", "recovery")

#: Production dropout / plane-deterioration knobs (docs/DROPOUT.md).
DROPOUT_DEFAULTS = {"drop_mask_p": 0.25, "drop_plane_p": 0.0, "drop_start": 0.5,
                    "plane_sig": 3.0, "plane_sig_end": 8.0,
                    "plane_glitch_p": 0.03, "plane_glitch_end": 0.25,
                    "plane_rand_frac": 0.5}

EPILOG = """\
reference hardware
  training: 1x NVIDIA H200 (141 GB).  Defaults are the H200 recipe:
    --batch 16 --accum 4 (64 effective) --steps 15000
    pi0 ~1.6-1.8 s/step (~7.5 h, ~107 GB), SmolVLA ~2.6 h, ACT ~1.1 h.
  smaller GPUs (e.g. RTX 5090, 32 GB): keep the effective batch with
    --batch 8 --accum 8 (pi0 needs the full fp32 master weights + AdamW
    state, ~52 GB static; it does not fit a 32 GB card - ACT and SmolVLA
    do).  --smoke runs 60 steps at --batch 4 --accum 1 for a plumbing check.

environment fallback
  every flag reads SIMRETRACT_<FLAG> and then the legacy name shown in its
  help (e.g. --cache <- SIMRETRACT_CACHE, CACHE) before its default.
  precedence: command line > --preset > environment > default.

examples
  simretract-train --preset baseline --cache data/cache --out runs/baseline
  simretract-train --policy act --data full --ablation none --smoke
  simretract-train --preset no_depth --batch 8 --accum 8

training is not bit-reproducible (unseeded sampler and augmentation).
"""


# --------------------------------------------------------------------------
# environment fallback helpers
# --------------------------------------------------------------------------
def env_value(name: str, legacy: str | None = None) -> str | None:
    """``SIMRETRACT_<name>`` or the legacy environment name, else ``None``."""
    for key in (f"SIMRETRACT_{name}", legacy or name):
        val = os.environ.get(key)
        if val not in (None, ""):
            return val
    return None


def env_bool(name: str, default: bool, legacy: str | None = None) -> bool:
    val = env_value(name, legacy)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def env_float(name: str, default: float | None,
              legacy: str | None = None) -> float | None:
    val = env_value(name, legacy)
    return default if val is None else float(val)


def env_int(name: str, default: int | None, legacy: str | None = None) -> int | None:
    val = env_value(name, legacy)
    return default if val is None else int(val)


# --------------------------------------------------------------------------
# repository configuration (paths / hub ids) with graceful fallbacks
# --------------------------------------------------------------------------
def _config_module():
    try:
        from simretract import config as srconfig
    except ImportError:
        return None
    return srconfig


def config_path(key: str, fallback: str) -> str:
    """A root from ``configs/paths.yaml`` (``data_cache``, ``runs``,
    ``checkpoints``, ``selections``) via ``simretract.config.paths()``;
    ``fallback`` when the config module or the key is unavailable."""
    cfg = _config_module()
    if cfg is None:
        return fallback
    try:
        roots = cfg.paths()
        return str(roots[key])
    except Exception:  # noqa: BLE001 - missing key/api -> repo-relative path
        return fallback


def default_strict_json() -> str:
    """``data/selections/<strict selection>`` from ``configs/data.yaml``
    (``sets.strict.selection``), falling back to ``strict_500.json``."""
    name = "strict_500.json"
    cfg = _config_module()
    if cfg is not None:
        try:
            name = str(cfg.data_config()["sets"]["strict"]["selection"])
        except Exception:  # noqa: BLE001
            pass
    sel = config_path("selections", os.path.join("data", "selections"))
    return os.path.join(sel, name)


def hub_id(key: str) -> str:
    """A base-weight / tokenizer id from ``configs/hub.yaml`` (section
    ``base_weights``) via ``simretract.config.hub()``.  Hub ids are never
    inlined in code, so a missing key is an error with the fix spelled out."""
    cfg = _config_module()
    table: dict[str, Any] = {}
    if cfg is not None:
        try:
            table = dict(cfg.hub())
        except Exception:  # noqa: BLE001
            table = {}
    for section in (table.get("base_weights") or {}, table):
        if isinstance(section, dict) and section.get(key):
            return str(section[key])
    raise SystemExit(f"configs/hub.yaml has no base_weights.{key}; add the hub "
                     f"id (see checkpoints/README.md) or pass --init")


# --------------------------------------------------------------------------
# dataset-side knobs (former environment variables)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Knobs:
    """Dataset/augmentation switches that used to be environment variables.

    ``drop_*`` and ``plane_*`` implement the late-trajectory input dropout and
    plane deterioration rule (docs/DROPOUT.md): with ``f`` the frame's
    position in the kept trajectory, the ramp ``w = max(0, (f - drop_start) /
    (1 - drop_start))`` scales the blind-frame probability (``drop_mask_p``,
    both image streams black and the plane zeroed), the plane-only drop
    (``drop_plane_p``), the Gaussian plane noise (``plane_sig`` ->
    ``plane_sig_end`` deg) and the glitch probability (``plane_glitch_p`` ->
    ``plane_glitch_end``); ``plane_rand_frac`` of the glitches on the ramp
    become a fully random normal.
    """
    float_depth: bool = True
    nogrip: bool = True
    no_aug: bool = False
    chunk_h: int = 50
    video_t: int = 0
    video_ratio: int = 6
    drop_mask_p: float = 0.25
    drop_plane_p: float = 0.0
    drop_start: float = 0.5
    plane_sig: float = 3.0
    plane_sig_end: float = 8.0
    plane_glitch_p: float = 0.03
    plane_glitch_end: float = 0.25
    plane_rand_frac: float = 0.5


def knobs_from_args(a: argparse.Namespace) -> Knobs:
    return Knobs(float_depth=a.float_depth, nogrip=a.nogrip, no_aug=a.no_aug,
                 chunk_h=a.chunk_h, video_t=a.video_t, video_ratio=a.video_ratio,
                 drop_mask_p=a.drop_mask_p, drop_plane_p=a.drop_plane_p,
                 drop_start=a.drop_start, plane_sig=a.plane_sig,
                 plane_sig_end=a.plane_sig_end, plane_glitch_p=a.plane_glitch_p,
                 plane_glitch_end=a.plane_glitch_end,
                 plane_rand_frac=a.plane_rand_frac)


# --------------------------------------------------------------------------
# cache access
# --------------------------------------------------------------------------
class Cached:
    """One cache set: memmapped image streams + in-memory state/action/plane."""

    def __init__(self, cache: str | os.PathLike, name: str):
        cache = str(cache)
        with open(f"{cache}/meta.json") as fh:
            meta = json.load(fh)[name]
        self.name = name
        self.episodes = meta["episodes"]
        self.prekept = bool(meta.get("prekept", False))  # already resampled
        self.rgb = np.load(f"{cache}/{name}_rgb.npy", mmap_mode="r")
        self.msk = np.load(f"{cache}/{name}_msk.npy", mmap_mode="r")
        self.gmk = np.load(f"{cache}/{name}_gmk.npy", mmap_mode="r")
        self.dep = np.load(f"{cache}/{name}_dep.npy", mmap_mode="r")
        self.state = np.load(f"{cache}/{name}_state.npy")
        self.action = np.load(f"{cache}/{name}_action.npy")
        self.gpose = np.load(f"{cache}/{name}_gpose.npy")
        plane_path = f"{cache}/{name}_plane.npy"
        self.plane = np.load(plane_path) if os.path.exists(plane_path) else None
        self.ep_end = np.zeros(len(self.state), np.int64)
        for a0, a1 in self.episodes:
            self.ep_end[a0:a1] = a1


def letterbox(img: np.ndarray, res: int = RES) -> np.ndarray:
    """Aspect-preserving resize onto a black ``res x res`` canvas."""
    import cv2  # local: keeps --help / preset resolution free of OpenCV

    h, w = img.shape[:2]
    s = res / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    out = np.zeros((res, res, 3), img.dtype)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out[(res - nh) // 2:(res - nh) // 2 + nh,
        (res - nw) // 2:(res - nw) // 2 + nw] = r
    return out


def _seg_aug():
    """``simretract.data.seg_aug`` (imported on first use so that argument
    parsing works in environments without the data package)."""
    from simretract.data import seg_aug
    return seg_aug


def masked_input(rgb: np.ndarray, msk: np.ndarray, gmk: np.ndarray,
                 m_seg: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Masked RGB stream: visible = (corrupted tool mask & true tool) | gripper;
    false-positive leak regions get a random texture fill; the rest is black."""
    tool_b = msk > 0
    grip_b = gmk > 0
    seg_b = m_seg > 0
    visible = (seg_b & tool_b) | grip_b
    leak = seg_b & ~tool_b & ~grip_b
    out = np.zeros_like(rgb)
    out[visible] = rgb[visible]
    if leak.any():
        fill = _seg_aug().random_fill(rgb.shape[0], rgb.shape[1], rng)
        out[leak] = fill[leak]
    return out


def close_cutoff(s: Cached, a0: int, a1: int, margin: int = CLOSE_MARGIN) -> int:
    """Frame where the carriage first reaches its final closed width (within
    0.1 mm) plus a 1 s hold margin.  Later frames exist only for the verdict
    machinery (press/release/battery), not for imitation."""
    g = s.state[a0:a1, 6]
    if len(g) < 40:
        return a1
    arrived = np.nonzero(g - float(np.median(g[-20:])) < 1e-4)[0]
    if not len(arrived):
        return a1
    return min(a1, a0 + int(arrived[0]) + margin)


def plane_input(n_cam: Sequence[float], rng: np.random.Generator, train: bool,
                w: float = 0.0, knobs: Knobs = Knobs()) -> np.ndarray:
    """Camera-frame plane unit normal (sign toward the camera), perturbed at
    train time by Gaussian rotation noise (sigma ``plane_sig`` deg) plus a
    ``plane_glitch_p`` chance of a large 10-45 deg error - the measured
    real-world mask+depth plane noise - and deteriorated toward the close by
    the ramp weight ``w`` (see ``Knobs``).  A degenerate normal returns zeros."""
    n = np.asarray(n_cam, np.float64)
    nn = np.linalg.norm(n)
    if nn < 0.5:
        return np.zeros(3, np.float32)   # degenerate frame marker
    n = n / nn
    if train:
        sig = knobs.plane_sig + w * (knobs.plane_sig_end - knobs.plane_sig)
        gp = knobs.plane_glitch_p + w * (knobs.plane_glitch_end - knobs.plane_glitch_p)
        if rng.random() < gp:
            if w > 0 and rng.random() < knobs.plane_rand_frac:
                n = rng.standard_normal(3)          # fully random direction
                n /= max(np.linalg.norm(n), 1e-9)
                if n[2] > 0:
                    n = -n
                return n.astype(np.float32)
            ang = np.radians(rng.uniform(10.0, 45.0))
        else:
            ang = np.radians(abs(rng.normal(0.0, sig)))
        ax = rng.standard_normal(3)
        ax -= (ax @ n) * n
        ax /= max(np.linalg.norm(ax), 1e-9)
        n = n * np.cos(ang) + np.cross(ax, n) * np.sin(ang)
        if n[2] > 0:
            n = -n
    return n.astype(np.float32)


def kept_frames(s: Cached, a0: int, end: int, resample: bool = True) -> np.ndarray:
    """Arc-length resampling of ``[a0, end)``: keep a frame once cumulative
    motion since the last kept frame reaches about one normal step (``JQ`` rad
    on joints / ``GQ`` m on the carriage).  Stalls vanish, slow crawl
    compresses, normal-speed motion keeps every frame."""
    T = end - a0
    if T <= 2 or not resample:
        return np.arange(a0, end)
    d = np.abs(np.diff(s.state[a0:end], axis=0))
    step = np.maximum(d[:, :6].max(1) / JQ, d[:, 6] / GQ)
    kept, acc = [0], 0.0
    for t in range(1, T):
        acc += step[t - 1]
        if acc >= 1.0:
            kept.append(t)
            acc = 0.0
    if kept[-1] != T - 1:
        kept.append(T - 1)
    return a0 + np.asarray(kept, np.int64)


class DS:
    """Map-style dataset over the kept frames of the selected episodes.

    Each item is one policy input (two image streams by ``mode``, the state
    vector by ``state_mode``) with its ``chunk_h`` action chunk, normalized
    when ``norm`` is given.  ``train`` selects the first 95% of the episodes
    of every set (in index order) and enables augmentation; ``subset`` is an
    optional per-set set of episode indices (half-data regime).
    """

    def __init__(self, sets: Sequence[Cached], mode: str, train: bool = True,
                 val_frac: float = VAL_FRAC,
                 keep_eps: set[int] | None = None, trim: str = "close",
                 resample: bool = True, norm: tuple | None = None,
                 state_mode: str = "full",
                 subset: Sequence[set[int] | None] | None = None,
                 knobs: Knobs = Knobs()):
        self.sets, self.mode, self.train = list(sets), mode, train
        self.state_mode = state_mode
        self.norm = norm
        self.k = knobs
        self.index: list[tuple[int, int, int]] = []   # (set, episode, kept position)
        self.kept: dict[tuple[int, int], np.ndarray] = {}
        for si, s in enumerate(self.sets):
            eps = list(range(len(s.episodes)))
            if subset is not None and subset[si] is not None:
                eps = [e for e in eps if e in subset[si]]
            if keep_eps is not None:
                eps = [e for e in eps if e in keep_eps]
            cut = len(eps) - max(1, int(len(eps) * val_frac))
            for e in (eps[:cut] if train else eps[cut:]):
                a0, a1 = s.episodes[e]
                end = close_cutoff(s, a0, a1) if trim == "close" else a1
                arr = kept_frames(s, a0, end, resample and not s.prekept)
                self.kept[(si, e)] = arr
                self.index += [(si, e, p) for p in range(len(arr))]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int) -> dict[str, Any]:
        si, e, p = self.index[k]
        s = self.sets[si]
        arr = self.kept[(si, e)]
        i = int(arr[p])
        H = self.k.chunk_h
        rng = np.random.default_rng()
        aug = self.train and not self.k.no_aug
        img, m3 = self._frame(s, i, rng, aug)
        if self.k.video_t > 0:
            # video mode (fastwam): frame 0 = the augmented policy input,
            # frames 1..T-1 = clean future kept frames video_ratio apart
            # (clamped to the episode end, like the action padding)
            imgs, m3s = [img], [m3]
            for t in range(1, self.k.video_t):
                j = int(arr[min(p + t * self.k.video_ratio, len(arr) - 1)])
                fi, fm = self._frame(s, j, np.random.default_rng(), False)
                imgs.append(fi)
                m3s.append(fm)
            img = np.stack(imgs)   # [T,RES,RES,3]
            m3 = np.stack(m3s)
        seg = arr[p:p + H]
        pad = np.arange(p, p + H) >= len(arr)
        if len(seg) < H:
            seg = np.concatenate([seg, np.full(H - len(seg), arr[-1])])
        act = s.action[seg].astype(np.float32)
        # late-trajectory ramp weight: 0 before drop_start, 1 at the close
        f = p / max(len(arr) - 1, 1)
        drop_w = (max(0.0, (f - self.k.drop_start)
                      / max(1.0 - self.k.drop_start, 1e-6)) if aug else 0.0)
        if self.state_mode == "qpos":
            state = s.state[i].astype(np.float32).copy()
        elif self.state_mode == "qpos_plane":
            state = np.concatenate([
                s.state[i].astype(np.float32)[:7],
                plane_input(s.plane[i], rng, aug, drop_w, self.k)])
        else:
            state = np.concatenate([s.state[i],
                                    s.gpose[i]]).astype(np.float32)
        if self.train and (self.k.drop_mask_p > 0 or self.k.drop_plane_p > 0):
            w = drop_w
            r = rng.random()
            if r < self.k.drop_mask_p * w:          # blind: no mask -> no plane
                img = np.zeros_like(img)
                m3 = np.zeros_like(m3)
                if self.state_mode == "qpos_plane":
                    state[7:10] = 0.0
            elif (r < (self.k.drop_mask_p + self.k.drop_plane_p) * w
                  and self.state_mode == "qpos_plane"):
                state[7:10] = 0.0                  # plane lost, mask kept
        if self.norm is not None:
            sm, ss, am, asd = self.norm
            state = (state - sm) / ss
            act = (act - am) / asd
        perm = (0, 3, 1, 2) if img.ndim == 4 else (2, 0, 1)
        out = {"observation.images.wrist": torch.from_numpy(img).permute(*perm),
               "observation.state": torch.from_numpy(state),
               "action": torch.from_numpy(act),
               "action_is_pad": torch.from_numpy(pad),
               "task": TASK}
        if self.mode != "masked_rgb_nodepth":
            # no-depth ablation = the depth stream removed entirely (single
            # camera; the policies handle a variable camera count natively)
            out["observation.images.mask"] = torch.from_numpy(m3).permute(*perm)
        return out

    def _frame(self, s: Cached, i: int, rng: np.random.Generator,
               aug: bool) -> tuple[np.ndarray, np.ndarray]:
        """One frame's two image streams (float32 HWC in [0,1]): the
        masked/filled RGB and the second stream (mask or depth by mode)."""
        seg_aug = _seg_aug()
        rgb = np.ascontiguousarray(s.rgb[i])
        msk = np.ascontiguousarray(s.msk[i])
        gmk = np.ascontiguousarray(s.gmk[i])
        if self.k.nogrip:  # only the tool is ever visible
            gmk = np.zeros_like(gmk)
        m_in = seg_aug.corrupt_mask(msk, rng) if aug \
            else (msk > 0).astype(np.uint8)
        if self.mode in ("masked_rgb", "masked_rgbd", "masked_rgb_nodepth"):
            rgb = masked_input(rgb, msk, gmk, m_in, rng)
            cam2 = ((m_in > 0) | (gmk > 0)).astype(np.uint8)
            if self.mode == "masked_rgb_nodepth":
                # the second stream is dropped by __getitem__; the plane
                # normal stays in the state. cam2 is unused in this mode.
                cam2 = np.zeros_like(cam2)
        else:
            cam2 = m_in
        if self.mode in ("masked_rgbd", "full_rgbd"):
            dep = np.ascontiguousarray(s.dep[i]).astype(np.float32) / 1000.0
            dep = np.clip(dep / 1.5, 0, 1)  # 0..1.5 m -> 0..1
            if aug:
                dep = seg_aug.augment_depth(dep, rng)
            if self.mode == "masked_rgbd":
                visible = ((m_in > 0) & (msk > 0)) | (gmk > 0)
                dep = dep * visible
                leak = (m_in > 0) & ~(msk > 0) & ~(gmk > 0)
                if self.train and leak.any():
                    dep[leak] = seg_aug.random_depth_fill(*dep.shape, rng)[leak]
            # full_rgbd (no-mask ablation): unmasked RGB + unmasked depth in
            # the same two-stream layout; augment_depth still applies.
            cam2 = None  # depth replaces the mask camera
            if self.k.float_depth:
                d3f = np.repeat(dep.astype(np.float32)[:, :, None], 3, 2)
                m3 = letterbox(d3f)          # ~1 mm granularity end-to-end
            else:
                d3 = np.repeat((dep * 255).astype(np.uint8)[:, :, None], 3, 2)
                m3 = letterbox(d3).astype(np.float32) / 255.0
        img = letterbox(rgb).astype(np.float32) / 255.0
        if cam2 is not None:
            m3 = letterbox(np.repeat((cam2 * 255)[:, :, None], 3, 2)) \
                .astype(np.float32) / 255.0
        return np.ascontiguousarray(img), np.ascontiguousarray(m3)


def stats_from(sets: Sequence[Cached], state_mode: str = "full",
               quantile: bool = False) -> dict[str, dict[str, Any]]:
    """Normalization statistics over the FULL sets (a subset never changes
    them): z-score (mean/std) or the quantile affine form (mean=(q01+q99)/2,
    std=(q99-q01)/2, i.e. q01 -> -1, q99 -> +1)."""
    if state_mode == "qpos":
        st = np.concatenate([s.state for s in sets])
    elif state_mode == "qpos_plane":
        st = np.concatenate(
            [np.concatenate([s.state.astype(np.float32),
                             s.plane.astype(np.float32)], 1)
             for s in sets])
    else:
        st = np.concatenate([np.concatenate([s.state, s.gpose], 1)
                             for s in sets])
    ac = np.concatenate([s.action for s in sets])
    t = lambda x: torch.tensor(x, dtype=torch.float32)  # noqa: E731
    z224 = {"mean": torch.zeros(3, 1, 1), "std": torch.ones(3, 1, 1)}
    if quantile:
        def qaff(x):
            q01, q99 = np.quantile(x, 0.01, axis=0), np.quantile(x, 0.99, axis=0)
            return {"mean": t((q01 + q99) / 2), "std": t((q99 - q01) / 2 + 1e-6)}
        log.info("normalization: quantiles (q01 -> -1, q99 -> +1) for state and action")
        return {"observation.state": qaff(st), "action": qaff(ac),
                "observation.images.wrist": z224, "observation.images.mask": z224}
    return {"observation.state": {"mean": t(st.mean(0)),
                                  "std": t(st.std(0) + 1e-6)},
            "action": {"mean": t(ac.mean(0)), "std": t(ac.std(0) + 1e-6)},
            "observation.images.wrist": z224,
            "observation.images.mask": z224}


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------
def _add_bool(ap: argparse.ArgumentParser, flag: str, dest: str, default: bool,
              legacy: str, help_: str) -> None:
    ap.add_argument(flag, dest=dest, action=argparse.BooleanOptionalAction,
                    default=env_bool(dest.upper(), default, legacy),
                    help=f"{help_} [env {legacy}] (default: {default})")


def build_parser(use_config: bool = True) -> argparse.ArgumentParser:
    """The trainer's argument parser (all former environment knobs are flags
    with environment fallback; defaults are the production values).
    ``use_config=False`` resolves presets from the builtin table only."""
    ap = argparse.ArgumentParser(
        prog="simretract-train",
        description="Train one SimRetract policy row.",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("row selection")
    g.add_argument("--preset", choices=_presets.preset_names(use_config),
                   help="named row from configs/train/presets.yaml; installs "
                        "defaults for --policy/--data/--ablation/--dropout "
                        "(explicit flags still win)")
    g.add_argument("--policy", choices=POLICIES, default=None,
                   help="policy class (default: pi0)")
    g.add_argument("--data", choices=DATA_MODES, default=None,
                   help="full = strict + recovery sets sampled 50/50; "
                        "strict_only = strict set only; half = deterministic "
                        "half selection of both (default: full)")
    g.add_argument("--ablation", choices=ABLATIONS, default=None,
                   help="no_plane -> --state qpos; no_mask -> --input-mode "
                        "full_rgbd; no_depth -> --input-mode masked_rgb_nodepth; "
                        "no_aug -> --no-aug (default: none)")
    g.add_argument("--dropout", action=argparse.BooleanOptionalAction, default=None,
                   help="late-trajectory blind dropout + plane deterioration; "
                        "--no-dropout sets --drop-mask-p 0, --plane-sig-end = "
                        "--plane-sig, --plane-glitch-end = --plane-glitch-p, "
                        "--plane-rand-frac 0 (default: on)")
    g.add_argument("--smoke", action="store_true",
                   help="60 steps, log every 5, no snapshots; --batch 4 "
                        "--accum 1 unless given")

    g = ap.add_argument_group("data")
    g.add_argument("--cache", default=env_value("CACHE"),
                   help="frame cache directory (simretract-build-cache) "
                        "[env CACHE] (default: configs/paths.yaml data/cache)")
    g.add_argument("--sets", default=env_value("SETS"),
                   help="comma list of cache set names [env SETS] (default: "
                        "strict,recovery; strict for --data strict_only)")
    g.add_argument("--set-frac", "--set_frac", dest="set_frac",
                   default=env_value("SET_FRAC"),
                   help="per-set sampling probabilities, comma list or 'none' "
                        "for frame-proportional [env SET_FRAC] (default: equal "
                        "shares, i.e. 0.5,0.5 for two sets)")
    g.add_argument("--subset-json", "--subset_json", dest="subset_json",
                   default=env_value("SUBSET_JSON"),
                   help="{set: [episode indices]} filter applied before the "
                        "train/val split; written automatically by --data half "
                        "[env SUBSET_JSON]")
    g.add_argument("--strict-json", default=env_value("STRICT_JSON"),
                   help="strict selection json (seeds + grasp_z_cm) used by "
                        "--data half (default: <selections>/strict_500.json)")
    g.add_argument("--input-mode", "--input_mode", dest="input_mode",
                   choices=INPUT_MODES, default=None,
                   help="full: raw RGB + corrupted mask; masked_rgb: masked RGB "
                        "+ mask; masked_rgbd: masked RGB + masked depth; "
                        "full_rgbd: unmasked RGB + depth; masked_rgb_nodepth: "
                        "masked RGB only (default: masked_rgbd)")
    g.add_argument("--state", choices=STATE_MODES, default=None,
                   help="qpos (7), qpos_plane (7 + plane normal), full (qpos + "
                        "grasp pose oracle) (default: qpos_plane)")
    g.add_argument("--episodes", default="all", choices=["all", "strict"],
                   help="strict: keep only strict_pass episodes of --verdicts")
    g.add_argument("--verdicts", default=None,
                   help="verdict json for --episodes strict (default: "
                        "<cache>/<first set>_verdicts.json)")
    g.add_argument("--trim", default="close", choices=["close", "none"],
                   help="close: train only up to gripper close (+1 s); none: "
                        "full episodes")
    g.add_argument("--resample", default="arc", choices=["arc", "none"],
                   help="arc: arc-length resample the timeline (no-op on "
                        "pre-kept caches); none: raw timeline")
    g.add_argument("--normalize", default="zscore",
                   choices=["zscore", "quantile", "none"],
                   help="state/action normalization saved to norm.json; "
                        "quantile = q01 -> -1, q99 -> +1 in the same affine form")
    _add_bool(g, "--float-depth", "float_depth", True, "FLOAT_DEPTH",
              "depth stream as float32 end-to-end (~1 mm) instead of uint8")
    _add_bool(g, "--nogrip", "nogrip", True, "NOGRIP",
              "zero the gripper mask (only the tool is ever visible)")
    g.add_argument("--no-aug", dest="no_aug", action="store_true",
                   default=env_bool("NO_AUG", False, "NO_AUG"),
                   help="disable every train-time augmentation and the dropout "
                        "ramp [env NO_AUG]")
    g.add_argument("--chunk-h", "--chunk_h", dest="chunk_h", type=int,
                   default=env_int("CHUNK_H", None),
                   help="action chunk length [env CHUNK_H] (default: 50; 48 for "
                        "fastwam)")
    g.add_argument("--video-t", "--video_t", dest="video_t", type=int,
                   default=env_int("VIDEO_T", None),
                   help="video mode frames (current + T-1 future) [env VIDEO_T] "
                        "(default: 0; 9 for fastwam)")
    g.add_argument("--video-ratio", "--video_ratio", dest="video_ratio", type=int,
                   default=env_int("VIDEO_RATIO", 6),
                   help="kept-frame spacing between video frames [env VIDEO_RATIO]")
    g.add_argument("--num-workers", dest="num_workers", type=int,
                   default=env_int("NUM_WORKERS", 8),
                   help="DataLoader workers [env NUM_WORKERS]")

    g = ap.add_argument_group("dropout / plane deterioration (docs/DROPOUT.md)")
    for key, val in DROPOUT_DEFAULTS.items():
        legacy = key.upper()
        g.add_argument("--" + key.replace("_", "-"), dest=key, type=float,
                       default=env_float(legacy, None, legacy),
                       help=f"[env {legacy}] (default: {val})")

    g = ap.add_argument_group("schedule")
    g.add_argument("--steps", type=int, default=15000,
                   help="optimizer steps (each = --accum micro-batches)")
    g.add_argument("--batch", type=int, default=None,
                   help="physical batch size (default: 16; 4 with --smoke)")
    g.add_argument("--accum", type=int, default=None,
                   help="gradient accumulation (default: 4; 1 with --smoke)")
    g.add_argument("--lr", type=float, default=2.5e-5,
                   help="peak LR of the baseline regime (AdamW betas 0.9/0.95, "
                        "eps 1e-8, wd 1e-10, clip 10)")
    g.add_argument("--warmup", type=int, default=1000,
                   help="linear warm-up steps, then cosine to 0.1x")
    g.add_argument("--save-every", "--save_every", dest="save_every", type=int,
                   default=env_int("SAVE_EVERY", 5000),
                   help="snapshot every N steps to <out>/step<N>/ [env SAVE_EVERY]")
    g.add_argument("--device", default="cuda", help="torch device")

    g = ap.add_argument_group("initialisation / output")
    g.add_argument("--init", default=env_value("INIT"),
                   help="checkpoint dir (or hub id) to initialise from "
                        "(default: <checkpoints>/pi0_base for pi0; the "
                        "base-weight id from configs/hub.yaml otherwise; ACT "
                        "trains from scratch)")
    g.add_argument("--allow-unverified-init", action="store_true",
                   help="accept a hub id / unverified directory as the pi0 --init")
    g.add_argument("--norm-from", "--norm_from", dest="norm_from", default="",
                   help="run dir whose norm.json stats are reused verbatim")
    g.add_argument("--out", default=None,
                   help="run directory (default: <runs>/<preset or policy>)")
    _add_bool(g, "--hf-offline", "hf_offline", False, "HF_HUB_OFFLINE",
              "set HF_HUB_OFFLINE=1 (pre-cached gated tokenizer/weights only)")

    g = ap.add_argument_group("policy-specific")
    _add_bool(g, "--smolvla-full", "smolvla_full", True, "SMOLVLA_FULL",
              "smolvla: unfreeze the VLM + vision tower and use the baseline "
              "optimizer regime")
    g.add_argument("--act-opt", dest="act_opt", choices=["baseline", "official"],
                   default=("baseline" if env_bool("ACT_BASELINE_OPT", True,
                                                   "ACT_BASELINE_OPT")
                            else "official"),
                   help="act: baseline pi0 regime or ACT's official preset "
                        "[env ACT_BASELINE_OPT] (default: baseline)")
    g.add_argument("--pi05-opt", dest="pi05_opt", choices=["baseline", "official"],
                   default=("official" if env_bool("PI05_OFFICIAL_OPT", False,
                                                   "PI05_OFFICIAL_OPT")
                            else "baseline"),
                   help="pi05: baseline pi0 regime or the port's preset "
                        "[env PI05_OFFICIAL_OPT] (default: baseline)")
    _add_bool(g, "--wam-gc", "wam_gc", True, "WAM_GC",
              "fastwam: gradient checkpointing")
    _add_bool(g, "--wam-freeze-video", "wam_freeze_video", False, "WAM_FREEZE_VIDEO",
              "fastwam: freeze the video expert (lambda_video defaults to 0)")
    g.add_argument("--wam-shift", dest="wam_shift", type=float,
                   default=env_float("WAM_SHIFT", 5.0),
                   help="fastwam: action scheduler train/infer shift [env WAM_SHIFT]")
    g.add_argument("--wam-lambda-video", dest="wam_lambda_video", type=float,
                   default=env_float("WAM_LAMBDA_VIDEO", None),
                   help="fastwam: video loss weight [env WAM_LAMBDA_VIDEO] "
                        "(default: 1, or 0 when the video expert is frozen)")
    _add_bool(g, "--wam-fp32-master", "wam_fp32_master", False, "WAM_FP32_MASTER",
              "fastwam: fp32 shadow master weights + fp32 Adam moments")
    g.add_argument("--wam-beta2", dest="wam_beta2", type=float,
                   default=env_float("WAM_BETA2", None),
                   help="fastwam: override Adam betas to (0.9, beta2) [env WAM_BETA2]")
    g.add_argument("--wam-cos-floor", dest="wam_cos_floor", type=float,
                   default=env_float("WAM_COS_FLOOR", 0.1),
                   help="fastwam: cosine floor multiplier [env WAM_COS_FLOOR]")
    return ap


def _split_csv(text: str | None) -> list[str]:
    return [x.strip() for x in (text or "").split(",") if x.strip()]


def finalize_args(a: argparse.Namespace) -> argparse.Namespace:
    """Resolve the high-level switches into concrete flags (in place).

    Ablation and data regime fill the input mode / state mode / sets only when
    those were not given explicitly; a conflicting explicit value is an error
    rather than a silent override.  ``--no-dropout`` resets the dropout knobs
    to the non-dropout recipe.
    """
    a.policy = a.policy or "pi0"
    a.data = a.data or "full"
    a.ablation = a.ablation or "none"
    a.dropout = True if a.dropout is None else bool(a.dropout)

    want_mode = {"no_mask": "full_rgbd", "no_depth": "masked_rgb_nodepth"}.get(
        a.ablation, "masked_rgbd")
    want_state = "qpos" if a.ablation == "no_plane" else "qpos_plane"
    if a.ablation != "none":
        if a.input_mode not in (None, want_mode):
            raise SystemExit(f"--ablation {a.ablation} implies --input-mode "
                             f"{want_mode}, got {a.input_mode}")
        if a.state not in (None, want_state):
            raise SystemExit(f"--ablation {a.ablation} implies --state "
                             f"{want_state}, got {a.state}")
        if a.ablation == "no_aug":
            a.no_aug = True
    a.input_mode = a.input_mode or want_mode
    a.state = a.state or want_state

    if a.sets is None:
        a.sets = DEFAULT_SETS[0] if a.data == "strict_only" else ",".join(DEFAULT_SETS)
    a.set_list = _split_csv(a.sets)
    if not a.set_list:
        raise SystemExit("--sets must name at least one cache set")
    if a.data == "half" and len(a.set_list) != 2:
        raise SystemExit("--data half needs exactly two sets (strict, recovery)")
    if a.set_frac is None:
        a.set_frac = (",".join(f"{1 / len(a.set_list):g}" for _ in a.set_list)
                      if len(a.set_list) > 1 else "none")
    a.set_frac_list = (None if a.set_frac.lower() == "none"
                       else [float(x) for x in _split_csv(a.set_frac)])
    if a.set_frac_list is not None and len(a.set_frac_list) != len(a.set_list):
        raise SystemExit("--set-frac must have one entry per set")

    for key, val in DROPOUT_DEFAULTS.items():
        if getattr(a, key) is None:
            setattr(a, key, val)
    if not a.dropout:
        a.drop_mask_p = 0.0
        a.drop_plane_p = 0.0
        a.plane_sig_end = a.plane_sig
        a.plane_glitch_end = a.plane_glitch_p
        a.plane_rand_frac = 0.0

    if a.chunk_h is None:
        a.chunk_h = 48 if a.policy == "fastwam" else 50
    if a.video_t is None:
        a.video_t = 9 if a.policy == "fastwam" else 0
    if a.batch is None:
        a.batch = 4 if a.smoke else 16
    if a.accum is None:
        a.accum = 1 if a.smoke else 4

    if a.cache is None:
        a.cache = config_path("data_cache", "data/cache")
    if a.out is None:
        a.out = os.path.join(config_path("runs", "runs"), a.preset or a.policy)
    if a.strict_json is None:
        a.strict_json = default_strict_json()
    if a.verdicts is None:
        a.verdicts = os.path.join(a.cache, f"{a.set_list[0]}_verdicts.json")
    if a.init is None and a.policy == "pi0":
        a.init = os.path.join(config_path("checkpoints", "checkpoints"), "pi0_base")
    if a.init is None:
        a.init = ""
    return a


def parse_args(argv: Sequence[str] | None = None,
               use_config: bool = True) -> argparse.Namespace:
    """Parse ``argv`` with preset defaults applied, then finalize.

    ``--smoke`` shrinks the batch to 4 x 1 unless ``--batch`` / ``--accum``
    were given on the command line (a preset's schedule does not count).
    """
    ap = build_parser(use_config)
    pre, _ = ap.parse_known_args(argv)
    if pre.preset:
        ap.set_defaults(**_presets.preset_flags(pre.preset, use_config))
    a = ap.parse_args(argv)
    given = {x.split("=", 1)[0] for x in (argv if argv is not None else sys.argv[1:])}
    if a.smoke:
        if "--batch" not in given:
            a.batch = 4
        if "--accum" not in given:
            a.accum = 1
    return finalize_args(a)


def validate_init(a: argparse.Namespace) -> None:
    """pi0 must start from a local, verified base-weight directory (the
    published pi0 base tensors differ from other hub copies), unless the user
    opts out with ``--allow-unverified-init``."""
    if a.policy != "pi0" or a.allow_unverified_init:
        return
    p = Path(a.init)
    if not (p.is_dir() and (p / "config.json").is_file()
            and (p / "model.safetensors").is_file()):
        raise SystemExit(
            f"pi0 --init must be a local checkpoint directory with config.json "
            f"and model.safetensors (got {a.init!r}). Download the verified base "
            f"weights with `simretract-download base-weights`, or pass "
            f"--allow-unverified-init to use a hub id / other directory.")


def _portable(path: str | None) -> str | None:
    """Local paths are recorded by basename so norm.json carries no machine
    layout; hub ids and ``None`` pass through unchanged."""
    if not path or not os.path.exists(path):
        return path
    return os.path.basename(os.path.normpath(path))


def train_regime(a: argparse.Namespace) -> dict[str, Any]:
    """The regime record written into norm.json."""
    return {"preset": a.preset, "policy": a.policy, "data": a.data,
            "ablation": a.ablation, "dropout": a.dropout,
            "dropout_knobs": {k: getattr(a, k) for k in DROPOUT_DEFAULTS},
            "smolvla_full": a.smolvla_full, "act_opt": a.act_opt,
            "pi05_opt": a.pi05_opt, "sets": a.set_list,
            "set_frac": a.set_frac_list, "input_mode": a.input_mode,
            "state_mode": a.state, "normalize": a.normalize, "trim": a.trim,
            "resample": a.resample, "steps": a.steps, "batch": a.batch,
            "accum": a.accum, "lr": a.lr, "warmup": a.warmup,
            "chunk_h": a.chunk_h, "video_t": a.video_t,
            "video_ratio": a.video_ratio, "init": _portable(a.init),
            "subset_json": _portable(a.subset_json)}


# --------------------------------------------------------------------------
# half-data selection (in-process)
# --------------------------------------------------------------------------
def run_half_selection(a: argparse.Namespace) -> str:
    """Run the deterministic half selection (``simretract.data.select_half``,
    RNG 0) and write ``half_subset.json`` / ``half_selection.json`` beside
    the run; returns the subset json path."""
    from simretract.data import select_half

    subset, selection = select_half.select_half(
        a.cache, a.strict_json, strict_set=a.set_list[0],
        far_set=a.set_list[1], rng_seed=0)
    os.makedirs(a.out, exist_ok=True)
    sub_path, _sel_path = select_half.write_selection(
        os.path.join(a.out, "half"), subset, selection)
    log.info("half selection: %s", {k: len(v) for k, v in subset.items()})
    return str(sub_path)


# --------------------------------------------------------------------------
# policy construction (lerobot imported here only)
# --------------------------------------------------------------------------
class PromptTokenizer:
    """Training-batch replica of the pi0_fast / pi05 processors.

    Neither port has a state-vector input: the normalized state is digitized
    over ``linspace(-1, 1, 257)[:-1]`` (minus one), padded to
    ``max_state_dim`` numbers and embedded in the prompt (pi0_fast:
    ``"Task: ..., State: ...;\\n"``; pi05: ``"...;\\nAction: "``), right-padded
    tokens.  pi0_fast additionally needs FAST action-token targets: the
    normalized chunk is FAST-tokenized per sample, mapped into the tail of the
    vocabulary, wrapped ``[BOS]"Action: "<ids>"|"+EOS`` and padded to
    ``max_action_tokens``.
    """

    def __init__(self, cfg: Any, task: str, suffix: str, with_action_tokens: bool,
                 default_tokenizer: str):
        from transformers import AutoTokenizer

        self.c, self.sfx, self.wat = cfg, suffix, with_action_tokens
        self.tname = getattr(cfg, "text_tokenizer_name", None) or default_tokenizer
        self.ltok = AutoTokenizer.from_pretrained(self.tname, padding_side="right")
        self.bins = np.linspace(-1, 1, 257)[:-1]
        self.clean = task.strip().replace("_", " ").replace("\n", " ")
        if self.wat:
            from transformers import AutoProcessor
            self.fast = AutoProcessor.from_pretrained(
                cfg.action_tokenizer_name, trust_remote_code=True)
            self.atok = AutoTokenizer.from_pretrained(
                self.tname, add_eos_token=True, add_bos_token=False)
            self.prefix = self.atok.encode("Action: ", add_special_tokens=False)
            self.end = self.atok.encode("|")

    def prompts(self, st: np.ndarray) -> list[str]:
        st = np.pad(st, ((0, 0), (0, self.c.max_state_dim - st.shape[1])))
        dig = np.digitize(st, bins=self.bins) - 1
        return [f"Task: {self.clean}, State: {' '.join(map(str, d))}{self.sfx}"
                for d in dig]

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        dv = batch["observation.state"].device
        st = batch["observation.state"].detach().float().cpu().numpy()
        enc = self.ltok(self.prompts(st), padding="max_length", truncation=True,
                        max_length=self.c.tokenizer_max_length,
                        return_tensors="pt")
        batch["observation.language.tokens"] = enc["input_ids"].to(dv)
        batch["observation.language.attention_mask"] = \
            enc["attention_mask"].to(dv, dtype=torch.bool)
        if not self.wat:
            return batch
        act = batch["action"].detach().float().cpu()
        V, S = self.ltok.vocab_size, self.c.fast_skip_tokens
        M = self.c.max_action_tokens
        toks, msks = [], []
        for i in range(act.shape[0]):
            ids = np.asarray(self.fast(act[i:i + 1])).flatten()
            seq = np.concatenate(([self.atok.bos_token_id], self.prefix,
                                  V - 1 - S - ids, self.end))[:M]
            m = np.zeros(M, dtype=bool)
            m[:len(seq)] = True
            toks.append(np.pad(seq, (0, M - len(seq))))
            msks.append(m)
        batch["action.tokens"] = torch.as_tensor(
            np.stack(toks), dtype=torch.long).to(dv)
        batch["action.token_mask"] = torch.as_tensor(np.stack(msks)).to(dv)
        return batch


@dataclass
class PolicyBundle:
    """A constructed policy with the batch adapters its family needs."""
    policy: Any
    cfg: Any
    lang_tokens: Any = None          # fixed task tokens (pi0 / smolvla)
    lang_masks: Any = None
    prompt_adapter: PromptTokenizer | None = None   # pi05 / pi0_fast
    imagenet_norm: bool = False      # act
    outer_autocast: bool = True      # False for policies managing precision


def _count(policy: Any) -> tuple[float, float]:
    n_all = sum(p.numel() for p in policy.parameters()) / 1e6
    n_tr = sum(p.numel() for p in policy.parameters() if p.requires_grad) / 1e6
    return n_all, n_tr


def build_policy(a: argparse.Namespace, stats: dict[str, Any],
                 dev: str) -> PolicyBundle:
    """Instantiate the policy for ``a.policy`` from its base weights with the
    shared input/output features and chunk length; returns the policy on
    ``dev`` in train mode together with its batch adapters."""
    from lerobot.configs.types import FeatureType, PolicyFeature

    H = a.chunk_h
    state_dim = STATE_DIM[a.state]
    in_feats = {
        "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (3, RES, RES)),
        "observation.images.mask": PolicyFeature(FeatureType.VISUAL, (3, RES, RES)),
        "observation.state": PolicyFeature(FeatureType.STATE, (state_dim,))}
    out_feats = {"action": PolicyFeature(FeatureType.ACTION, (7,))}
    if a.input_mode == "masked_rgb_nodepth":
        in_feats.pop("observation.images.mask")   # single camera
        log.info("input features: single camera (no depth stream)")

    bundle_kw: dict[str, Any] = {}
    if a.policy == "fastwam":
        # Wan2.2-based world-action model: the horizon must be a multiple of
        # the video transitions, the video co-training loss needs the
        # dataset's video mode, the language is tokenised inside the policy.
        if not (a.video_t > 1 and (a.video_t - 1) % 4 == 0
                and H % (a.video_t - 1) == 0):
            raise SystemExit("fastwam needs --video-t 4k+1 and --chunk-h a "
                             "multiple of --video-t - 1")
        from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
        from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
        lam_video = a.wam_lambda_video
        if lam_video is None:
            lam_video = 0.0 if a.wam_freeze_video else 1.0
        cfg = FastWAMConfig(
            input_features=in_feats, output_features=out_feats,
            proprio_dim=state_dim, action_dim=7,
            action_horizon=H, n_action_steps=H,
            num_video_frames=1 + (a.video_t - 1) * a.video_ratio,
            action_video_freq_ratio=a.video_ratio,
            use_gradient_checkpointing=a.wam_gc,
            freeze_video_expert=a.wam_freeze_video,
            action_scheduler={"train_shift": a.wam_shift,
                              "infer_shift": a.wam_shift,
                              "num_train_timesteps": 1000},
            loss={"lambda_video": lam_video, "lambda_action": 1.0})
        log.info("fastwam config: horizon %d | video frames %d ratio %d (model "
                 "frames %d) | image_size %s | proprio %d | grad-ckpt %s | "
                 "freeze video %s | shift %s | lambda_video %s | dtype %s",
                 cfg.action_horizon, cfg.num_video_frames,
                 cfg.action_video_freq_ratio, a.video_t, cfg.image_size,
                 cfg.proprio_dim, cfg.use_gradient_checkpointing,
                 cfg.freeze_video_expert, cfg.action_scheduler["train_shift"],
                 cfg.loss["lambda_video"], cfg.torch_dtype)
        init = a.init or hub_id("fastwam")
        policy = FastWAMPolicy.from_pretrained(init, config=cfg)
        n_all, n_tr = _count(policy)
        log.info("initialised from %s | fastwam params %.1fM, trainable %.1fM",
                 init, n_all, n_tr)
        bundle_kw["outer_autocast"] = False
    elif a.policy in ("pi05", "pi0_fast"):
        # Same-family pi0 ports: chunk-50 native, the pi0 batch/optimizer/loop
        # paths apply; the normalized state travels in the prompt.
        if a.policy == "pi05":
            from lerobot.policies.pi05.configuration_pi05 import PI05Config
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy as Cls
            cfg = PI05Config()
            init = a.init or hub_id("pi05")
        else:
            from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig
            from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy as Cls
            cfg = PI0FastConfig()
            init = a.init or hub_id("pi0_fast")
        log.info("%s config: chunk %s | dtype %s | tokenizer_max_length %s",
                 a.policy, cfg.chunk_size, cfg.dtype, cfg.tokenizer_max_length)
        _assign_features(cfg, in_feats, out_feats, H)
        policy = Cls.from_pretrained(init, config=cfg, dataset_stats=stats)
        log.info("initialised from %s | %s params %.1fM", init, a.policy,
                 _count(policy)[0])
        adapter = PromptTokenizer(
            cfg, TASK, ";\n" if a.policy == "pi0_fast" else ";\nAction: ",
            a.policy == "pi0_fast", hub_id("paligemma_tokenizer"))
        log.info("%s prompt adapter: state bins 256 pad %d | example prompt: %r%s",
                 a.policy, cfg.max_state_dim,
                 adapter.prompts(np.zeros((1, state_dim), np.float32))[0],
                 (f" | FAST action tokens max {cfg.max_action_tokens} skip "
                  f"{cfg.fast_skip_tokens}" if adapter.wat else ""))
        bundle_kw["prompt_adapter"] = adapter
    elif a.policy == "act":
        # ACT trained from scratch (ResNet-18 ImageNet backbone, 4-layer
        # encoder / 1-layer decoder, VAE); chunk = n_action_steps so the whole
        # chunk executes; images are ImageNet-normalized in the loop.
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        cfg = ACTConfig()
        log.info("act config: backbone %s (%s) | dim %d enc %d dec %d | vae %s "
                 "kl %s | dropout %s | ensemble %s", cfg.vision_backbone,
                 cfg.pretrained_backbone_weights, cfg.dim_model,
                 cfg.n_encoder_layers, cfg.n_decoder_layers, cfg.use_vae,
                 cfg.kl_weight, cfg.dropout, cfg.temporal_ensemble_coeff)
        _assign_features(cfg, in_feats, out_feats, H)
        if a.init:
            log.warning("act trains from scratch; --init %s ignored", a.init)
        policy = ACTPolicy(cfg)
        log.info("act from scratch | params %.1fM (all trainable)", _count(policy)[0])
        bundle_kw["imagenet_norm"] = True
    elif a.policy == "smolvla":
        # SmolVLA on the same data/inputs/z-scoring/chunking as pi0.
        # load_vlm_weights=False: all parameters come from the checkpoint, so
        # the run has fp32 master weights + bf16 autocast exactly like pi0.
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        cfg = SmolVLAConfig(load_vlm_weights=False, pad_language_to="max_length",
                            **({"train_expert_only": False,
                                "freeze_vision_encoder": False}
                               if a.smolvla_full else {}))
        log.info("smolvla config: full fine-tune %s (train_expert_only=%s, "
                 "freeze_vision_encoder=%s)", a.smolvla_full,
                 cfg.train_expert_only, cfg.freeze_vision_encoder)
        _assign_features(cfg, in_feats, out_feats, H)
        init = a.init or hub_id("smolvla")
        policy = SmolVLAPolicy.from_pretrained(init, config=cfg)
        # from_pretrained is strict=False for this class: verify the weights
        # really covered the model (a silent key mismatch = random init)
        st = policy.state_dict()
        ref = SmolVLAPolicy(cfg).state_dict()
        same = [k for k in st if k in ref and st[k].shape == ref[k].shape
                and st[k].dtype.is_floating_point
                and torch.equal(st[k].detach().float().cpu(),
                                ref[k].detach().float().cpu())]
        n_all, n_tr = _count(policy)
        log.info("initialised from %s | smolvla params %.1fM, trainable %.1fM | "
                 "tensors identical to a fresh init: %d", init, n_all, n_tr,
                 len(same))
        del ref, st
        bundle_kw.update(_fixed_task_tokens(cfg.vlm_model_name, cfg, dev))
    else:
        from lerobot.policies.pi0.configuration_pi0 import PI0Config
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        cfg = PI0Config()
        _assign_features(cfg, in_feats, out_feats, H)
        policy = PI0Policy.from_pretrained(a.init, config=cfg, dataset_stats=stats)
        log.info("initialised from %s", a.init)
        bundle_kw.update(_fixed_task_tokens(hub_id("paligemma_tokenizer"), cfg, dev))
    policy.to(dev).train()
    return PolicyBundle(policy=policy, cfg=cfg, **bundle_kw)


def _assign_features(cfg: Any, in_feats: dict, out_feats: dict, H: int) -> None:
    cfg.input_features = in_feats
    cfg.output_features = out_feats
    cfg.chunk_size = H
    cfg.n_action_steps = H


def _fixed_task_tokens(tok_name: str, cfg: Any, dev: str) -> dict[str, Any]:
    """Tokenize the single task prompt once (pi0 / smolvla)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_name)
    enc = tok(TASK + "\n", padding="max_length", truncation=True,
              max_length=cfg.tokenizer_max_length, return_tensors="pt")
    return {"lang_tokens": enc["input_ids"].to(dev),
            "lang_masks": enc["attention_mask"].to(dev, dtype=torch.bool)}


# --------------------------------------------------------------------------
# optimizer regimes
# --------------------------------------------------------------------------
@dataclass
class OptimBundle:
    opt: Any
    sched: Any
    params: list
    clip: float
    masters: list | None = None   # fastwam fp32 shadow weights


def _warmup_cosine(warmup: int, steps: int, floor: float) -> Callable[[int], float]:
    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        t = (step - warmup) / max(steps - warmup, 1)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t))
    return lr_lambda


def build_optimizer(a: argparse.Namespace, b: PolicyBundle) -> OptimBundle:
    """Per-policy optimizer regime.  The baseline regime (AdamW lr 2.5e-5,
    betas 0.9/0.95, eps 1e-8, wd 1e-10, warm-up + cosine to 0.1x, clip 10)
    is shared by pi0, pi05 (baseline opt), smolvla-full and act (baseline
    opt); the other branches use the ports' own presets."""
    policy, cfg = b.policy, b.cfg
    if a.policy == "fastwam":
        # the port's AdamW preset (lr 1e-4, wd 1e-2) with the baseline's
        # warm-up + cosine schedule on top and grad clip 1.0
        opt_params = list(policy.get_optim_params())
        masters = None
        if a.wam_fp32_master:
            # fp32 shadow of every trainable tensor: step the optimizer on the
            # shadows and copy back to the bf16 params each step
            masters = [p.detach().clone().float() for p in opt_params]
            opt = cfg.get_optimizer_preset().build(masters)
            log.info("fastwam fp32 master weights: %d tensors, %.2fB params",
                     len(masters), sum(m.numel() for m in masters) / 1e9)
        else:
            opt = cfg.get_optimizer_preset().build(opt_params)
        if a.wam_beta2 is not None:
            for grp in opt.param_groups:
                grp["betas"] = (0.9, a.wam_beta2)
        clip = 1.0
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, _warmup_cosine(a.warmup, a.steps, a.wam_cos_floor))
        log.info("fastwam optimizer: preset lr %s wd %s betas %s, warm-up %d + "
                 "cosine to %sx, clip %s, %d param tensors", cfg.optimizer_lr,
                 cfg.optimizer_weight_decay, opt.param_groups[0]["betas"],
                 a.warmup, a.wam_cos_floor, clip, len(opt_params))
        return OptimBundle(opt, sched, opt_params, clip, masters)
    if a.policy == "pi0_fast" or (a.policy == "pi05" and a.pi05_opt == "official"):
        # the port's own optimizer preset and cosine-with-warmup scheduler
        # preset (auto-scaled to --steps)
        opt_params = [p for p in policy.parameters() if p.requires_grad]
        preset = cfg.get_optimizer_preset()
        opt = preset.build(opt_params)
        clip = preset.grad_clip_norm
        sched = cfg.get_scheduler_preset().build(opt, a.steps)
        sc = min(1.0, a.steps / cfg.scheduler_decay_steps)
        log.info("%s official preset: lr %s betas %s wd %s clip %s | scheduler "
                 "warmup %d (preset %d scaled x%.2f) decay over %d -> %s",
                 a.policy, cfg.optimizer_lr, cfg.optimizer_betas,
                 cfg.optimizer_weight_decay, clip,
                 int(cfg.scheduler_warmup_steps * sc), cfg.scheduler_warmup_steps,
                 sc, int(cfg.scheduler_decay_steps * sc), cfg.scheduler_decay_lr)
        return OptimBundle(opt, sched, opt_params, clip)
    if a.policy == "act" and a.act_opt == "official":
        # ACT's preset: AdamW lr 1e-5 (backbone param group), wd 1e-4,
        # constant schedule, the preset's grad clip
        preset = cfg.get_optimizer_preset()
        opt = preset.build(policy.get_optim_params())
        clip = preset.grad_clip_norm
        opt_params = [p for g in opt.param_groups for p in g["params"]]
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
        log.info("act official preset: lr %s backbone lr %s wd %s clip %s | "
                 "constant schedule | %d param tensors", cfg.optimizer_lr,
                 cfg.optimizer_lr_backbone, cfg.optimizer_weight_decay, clip,
                 len(opt_params))
        return OptimBundle(opt, sched, opt_params, clip)
    if a.policy == "smolvla" and not a.smolvla_full:
        # SmolVLA's presets: AdamW lr 1e-4, betas 0.9/0.95, eps 1e-8, wd 1e-10;
        # cosine-with-warmup auto-scaled to the run length
        opt_params = [p for p in policy.parameters() if p.requires_grad]
        clip = 10.0
        opt = cfg.get_optimizer_preset().build(opt_params)
        sched = cfg.get_scheduler_preset().build(opt, a.steps)
        log.info("smolvla optimizer: lr %s warm-up %s->scaled, decay to %s",
                 cfg.optimizer_lr, cfg.scheduler_warmup_steps, cfg.scheduler_decay_lr)
        return OptimBundle(opt, sched, opt_params, clip)
    opt_params = list(policy.parameters())
    clip = 10.0
    opt = torch.optim.AdamW(opt_params, lr=a.lr, betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=1e-10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, _warmup_cosine(a.warmup, a.steps, 0.1))
    log.info("baseline optimizer: AdamW lr %s betas (0.9, 0.95) wd 1e-10, "
             "warm-up %d + cosine to 0.1x, clip %s", a.lr, a.warmup, clip)
    return OptimBundle(opt, sched, opt_params, clip)


# --------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------
def save_checkpoint(policy: Any, ckpt_dir: str, normd: dict | None) -> None:
    """``save_pretrained`` into ``ckpt_dir`` plus its norm.json copy."""
    policy.save_pretrained(ckpt_dir)
    if normd:
        with open(os.path.join(ckpt_dir, "norm.json"), "w") as fh:
            json.dump(normd, fh)


def link_flat(run_dir: str, final_dir: str = "final") -> None:
    """Symlink final/{config.json, model.safetensors[, context.pt]} into the
    run directory so ``--ckpt runs/<name>`` and ``--ckpt runs/<name>/final``
    are interchangeable for serving and evaluation."""
    for name in ("config.json", "model.safetensors", "context.pt"):
        src = os.path.join(run_dir, final_dir, name)
        dst = os.path.join(run_dir, name)
        if not os.path.exists(src):
            continue
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.join(final_dir, name), dst)


def export_fastwam_context(policy: Any, cfg: Any, final_dir: str) -> None:
    """Precompute the UMT5 context of the single task prompt so a server can
    load the checkpoint without the text encoder."""
    try:
        policy.eval()
        with torch.no_grad():
            prompt = cfg.prompt_template.format(task=TASK)
            ctx = policy.model.encode_prompt([prompt])
            if isinstance(ctx, (tuple, list)):
                out = {"context": ctx[0].detach().float().cpu(),
                       "context_mask": ctx[1].detach().cpu(), "prompt": prompt}
            else:
                out = {"context": ctx.detach().float().cpu(), "prompt": prompt}
            torch.save(out, os.path.join(final_dir, "context.pt"))
            log.info("context.pt saved: %s",
                     {k: (tuple(v.shape) if torch.is_tensor(v) else v)
                      for k, v in out.items()})
    except Exception as ex:  # noqa: BLE001 - the checkpoint is still usable
        log.warning("context export skipped: %r", ex)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def _load_subset(a: argparse.Namespace) -> list[set[int] | None] | None:
    if not a.subset_json:
        return None
    with open(a.subset_json) as fh:
        sj = json.load(fh)
    log.info("subset: %s", {n: (len(sj[n]) if n in sj else "all") for n in a.set_list})
    return [set(sj[n]) if n in sj else None for n in a.set_list]


def _make_loader(a: argparse.Namespace, tr: DS) -> Any:
    """DataLoader with the per-set sampling fractions (a weighted sampler with
    replacement over the flat frame index, one nominal epoch per pass) or a
    plain shuffle when no fractions are given."""
    sampler = None
    if a.set_frac_list is not None:
        cnt = np.zeros(len(a.set_list))
        for si, _e, _p in tr.index:
            cnt[si] += 1
        w = np.array([a.set_frac_list[si] / cnt[si] for si, _e, _p in tr.index])
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(w, dtype=torch.double), num_samples=len(tr),
            replacement=True)
        log.info("set fractions %s frames %s", dict(zip(a.set_list, a.set_frac_list)),
                 dict(zip(a.set_list, cnt.astype(int).tolist())))
    if len(tr) < a.batch:
        raise SystemExit(f"only {len(tr)} training frames for --batch {a.batch}; "
                         f"use a larger cache or a smaller batch")
    return torch.utils.data.DataLoader(
        tr, batch_size=a.batch, shuffle=sampler is None, sampler=sampler,
        num_workers=a.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=a.num_workers > 0)


def _log_parts(tag: str, parts: Any) -> None:
    if isinstance(parts, dict) and parts:
        log.info("  %s parts %s", tag, " ".join(
            f"{k} {float(v):.4f}" for k, v in parts.items()
            if torch.is_tensor(v) or isinstance(v, float)))


def train_loop(a: argparse.Namespace, b: PolicyBundle, o: OptimBundle, dl: Any,
               normd: dict | None, dev: str) -> None:
    policy = b.policy
    steps = 60 if a.smoke else a.steps
    every = 5 if a.smoke else 100
    im_mean = im_std = None
    if b.imagenet_norm:
        # ImageNet normalization for the pretrained ResNet backbone (the port
        # applies it in its processor pipeline, which this stack bypasses)
        im_mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        im_std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    autocast_dev = "cuda" if str(dev).startswith("cuda") else "cpu"
    it, step, t0, losses = iter(dl), 0, time.time(), []
    while step < steps:
        o.opt.zero_grad(set_to_none=True)
        for micro in range(a.accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl)
                batch = next(it)
            batch = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            bs = batch["observation.state"].shape[0]
            if b.lang_tokens is not None:
                batch["observation.language.tokens"] = b.lang_tokens.expand(bs, -1)
                batch["observation.language.attention_mask"] = \
                    b.lang_masks.expand(bs, -1)
            if not b.outer_autocast:
                loss, parts = policy.forward(batch)
                if step % every == 0 and micro == 0:
                    _log_parts(a.policy, parts)
            else:
                if b.prompt_adapter is not None:
                    batch = b.prompt_adapter(batch)
                if im_mean is not None:
                    for k in list(batch):
                        if k.startswith("observation.images."):
                            batch[k] = (batch[k] - im_mean) / im_std
                with torch.autocast(autocast_dev, dtype=torch.bfloat16):
                    loss, parts = policy.forward(batch)
                if a.policy == "act" and step % every == 0 and micro == 0:
                    _log_parts("act", parts)
            (loss / a.accum).backward()
            losses.append(float(loss))
        if o.masters is not None:
            # fp32-master step: move the bf16 grads onto the fp32 shadows,
            # clip + step there, write the result back to the model params
            for p_, m_ in zip(o.params, o.masters):
                m_.grad = None if p_.grad is None else p_.grad.detach().float()
                p_.grad = None
            torch.nn.utils.clip_grad_norm_(o.masters, o.clip)
            o.opt.step()
            with torch.no_grad():
                for p_, m_ in zip(o.params, o.masters):
                    p_.data.copy_(m_.data)
        else:
            torch.nn.utils.clip_grad_norm_(o.params, o.clip)
            o.opt.step()
        o.sched.step()
        step += 1
        if step % every == 0:
            log.info("step %d/%d loss %.4f %.2fs/step", step, steps,
                     float(np.mean(losses[-100:])), (time.time() - t0) / step)
        if not a.smoke and a.save_every > 0 and step % a.save_every == 0:
            save_checkpoint(policy, os.path.join(a.out, f"step{step}"), normd)


def main(argv: Sequence[str] | None = None) -> int:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(message)s")
    a = parse_args(argv)
    if torch is None:
        raise SystemExit("torch is not installed in this interpreter; run "
                         "simretract-train from the policy environment")
    if a.hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    validate_init(a)
    os.makedirs(a.out, exist_ok=True)
    log.info("run %s | policy %s data %s ablation %s dropout %s | sets %s frac %s",
             a.out, a.policy, a.data, a.ablation, a.dropout, a.set_list, a.set_frac)
    if a.data == "half" and not a.subset_json:
        a.subset_json = run_half_selection(a)
    dev = a.device
    sets = [Cached(a.cache, n) for n in a.set_list]
    keep = None
    if a.episodes == "strict":
        with open(a.verdicts) as fh:
            keep = {int(k) for k, r in json.load(fh).items() if r["strict_pass"]}
        log.info("strict-only subset: %d episodes", len(keep))
    stats = stats_from(sets, a.state, quantile=(a.normalize == "quantile"))
    if a.norm_from:
        with open(os.path.join(a.norm_from, "norm.json")) as fh:
            nf = json.load(fh)
        t = lambda x: torch.tensor(x, dtype=torch.float32)  # noqa: E731
        stats["observation.state"] = {"mean": t(nf["state_mean"]),
                                      "std": t(nf["state_std"])}
        stats["action"] = {"mean": t(nf["action_mean"]), "std": t(nf["action_std"])}
        log.info("norm stats reused from %s", a.norm_from)
    knobs = knobs_from_args(a)
    norm = None
    normd = None
    if a.normalize in ("zscore", "quantile"):   # both are affine (x-mean)/std
        sm = stats["observation.state"]["mean"].numpy()
        ss = stats["observation.state"]["std"].numpy()
        am = stats["action"]["mean"].numpy()
        asd = stats["action"]["std"].numpy()
        norm = (sm, ss, am, asd)
        normd = {"state_mean": sm.tolist(), "state_std": ss.tolist(),
                 "action_mean": am.tolist(), "action_std": asd.tolist(),
                 "normalize": a.normalize,
                 "drop_aug": {"mask_p": knobs.drop_mask_p,
                              "plane_p": knobs.drop_plane_p,
                              "start": knobs.drop_start,
                              "plane_sig_end": knobs.plane_sig_end,
                              "plane_glitch_end": knobs.plane_glitch_end,
                              "plane_rand_frac": knobs.plane_rand_frac},
                 "float_depth": knobs.float_depth, "nogrip": knobs.nogrip,
                 "input_mode": a.input_mode, "state_mode": a.state,
                 "no_aug": knobs.no_aug, "policy": a.policy,
                 "chunk_h": knobs.chunk_h, "video_t": knobs.video_t,
                 "video_ratio": knobs.video_ratio,
                 "train_regime": train_regime(a)}
        with open(os.path.join(a.out, "norm.json"), "w") as fh:
            json.dump(normd, fh)
    subset = _load_subset(a)
    ds_kw = dict(keep_eps=keep, trim=a.trim, resample=a.resample == "arc",
                 norm=norm, state_mode=a.state, subset=subset, knobs=knobs)
    tr = DS(sets, a.input_mode, True, **ds_kw)
    va = DS(sets, a.input_mode, False, **ds_kw)
    log.info("mode=%s train %d frames, val %d (held out, not evaluated)",
             a.input_mode, len(tr), len(va))
    with open(os.path.join(a.out, "train_args.json"), "w") as fh:
        json.dump({k: v for k, v in vars(a).items()}, fh, indent=1, default=str)

    bundle = build_policy(a, stats, dev)
    optim = build_optimizer(a, bundle)
    dl = _make_loader(a, tr)
    train_loop(a, bundle, optim, dl, normd, dev)

    final = os.path.join(a.out, "final")
    save_checkpoint(bundle.policy, final, normd)
    if a.subset_json:
        for name in ("half_subset.json", "half_selection.json"):
            src = os.path.join(a.out, name)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(final, name))
    if a.policy == "fastwam":
        export_fastwam_context(bundle.policy, bundle.cfg, final)
    link_flat(a.out)
    log.info("TRAIN DONE %s", final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
