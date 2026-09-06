"""Build one SET of the flat frame cache directly from raw episode
directories.

Raw layout consumed (one directory per episode, as written by the strict
recorder ``ep_<seed>`` and the recovery recorder ``ep_<seed>_p<draw>``)::

    <raw>/ep_200005/
        rgb_0000.png ... rgb_NNNN.png      uint8  480x640x3  wrist RGB
        depth_0000.png ...                 uint16 480x640    depth, millimetres, 0 = invalid
        mask_0000.png ...                  uint8  480x640    tool mask (0 / 255)
        gmask_0000.png ...                 uint8  480x640    gripper-finger mask (0 / 255)
        omask_0000.png ...                 uint8  480x640    operator-hand mask (not cached)
        meta.npz
            states     float32 (T, 7)  joint positions 0..5 (rad) + carriage width (m)
            actions    float32 (T, 7)  commanded targets, same layout
            grasp_pose float32 (7,)    tool grasp pose (xyz + wxyz quaternion)
            seed       int             generation seed; the recovery draw index is in the dir name
            verdict    str             json: {"strict_pass": bool | null, ...} (absent/null when the battery was skipped)
            info       str             json: grasp diagnostics; recovery episodes add
                                       "perturb_win": {"start", "end"} (frames of the
                                       perturbation excursion) and "perturb": {"frac", ...}
            (tool_pos_t, tool_quat_t, task, fps, tool, mass_g, lighting, occ_z, tilt_deg
             are present but not needed here)

Cache layout produced in ``<out>`` (several sets share one directory)::

    <name>_rgb.npy    uint8  (N, 480, 640, 3)   memmap
    <name>_dep.npy    uint16 (N, 480, 640)      memmap
    <name>_msk.npy    uint8  (N, 480, 640)      memmap
    <name>_gmk.npy    uint8  (N, 480, 640)      memmap; left sparse (reads as 0) with skip_gmk
    <name>_state.npy  float32 (N, 7)
    <name>_action.npy float32 (N, 7)
    <name>_gpose.npy  float32 (N, 7)
    <name>_verdicts.json   {"<episode idx>": {"strict_pass" (bool; false when skipped), "battery_skipped", "seed"}}
    meta.json[<name>] = {"N", "episodes": [[a0, a1], ...], "prekept", "flags", ...}

Episode selection and trimming rules (all applied in this order):

* episodes without ``meta.npz`` / ``verdict`` or with ``seed < seed_min`` are
  skipped;
* seeds listed in the held-out test pool (``--exclude-seeds``) are skipped,
  so the benchmark seeds can never leak into training;
* ``pretrim``: cache only ``[0, close_cutoff)``, the first frame at which the
  carriage reaches its final width (median of the last 20 frames, within
  0.1 mm) plus a 25-frame margin. The trainer never reads past this, so the
  cut is lossless and ~40% smaller;
* ``perturb_trim`` (recovery sets): start at ``perturb_win.end + 1`` so only
  the recovery segment after the excursion trains and no action chunk
  crosses it; episodes without a usable window (start <= 0 or fewer than 20
  frames left) are skipped;
* an empty tool mask on all three probe frames (1/4, 1/2, 3/4 of the
  segment) marks a broken render and skips the episode;
* ``keep_only``: store only the frames the trainer's arc-length resampling
  would keep (joint quantum 2.5e-3 rad, carriage quantum 2.5e-4 m) and flag
  the set ``prekept`` so the trainer skips its own resampling; the training
  index is identical to caching every frame.

The build ends with a 20-frame bitwise spot check against fresh PNG decodes
(``VERIFY PASS``) and, optionally, an md5 fingerprint comparison of the
state-side arrays against the reference values in ``configs/data.yaml``
(sets without a recorded fingerprint report that and skip the comparison).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

FRAME_H, FRAME_W = 480, 640
JQ, GQ = 2.5e-3, 2.5e-4
"""Joint (rad) and carriage (m) motion quanta of the trainer's arc-length resampling."""
CLOSE_MARGIN = 25
"""Frames kept after the carriage reaches its final width (1 s at 25 fps)."""
N_SPOT_CHECK = 20

_STRICT_FLAGS = dict(pretrim=True, keep_only=True, seed_min=200000, skip_gmk=True, perturb_trim=False)
_RECOVERY_FLAGS = dict(pretrim=True, keep_only=True, seed_min=200000, skip_gmk=True, perturb_trim=True)

SET_PRESETS: dict[str, dict[str, Any]] = {
    "strict": dict(_STRICT_FLAGS),
    "recovery": dict(_RECOVERY_FLAGS),
    "strict_nosmooth": dict(_STRICT_FLAGS),
    "recovery_nosmooth": dict(_RECOVERY_FLAGS),
    "strict_nostandoff": dict(_STRICT_FLAGS),
    "recovery_nostandoff": dict(_RECOVERY_FLAGS),
}
"""Production flag sets per cache set (configs/data.yaml ``sets.<name>.cache_flags``):
every strict set uses the strict flags, every recovery set the recovery flags."""

REFERENCE_FINGERPRINTS: dict[str, dict[str, Any]] = {
    "strict": {"frames": 69402, "md5": {"action": "7d0e2c3b", "plane": "034cb12d", "gpose": "6a415e21"}},
    "recovery": {"frames": 106868, "md5": {"action": "636630ba", "plane": "93dda3b0", "gpose": "7f918202"}},
}
"""Reference fingerprints of the main pair (configs/data.yaml overrides; a set whose
``fingerprint`` is null there has none, and ``--verify`` skips it with a message)."""


class Episode(NamedTuple):
    """One accepted raw episode: directory, first raw frame, segment length, cached raw indices."""

    path: str
    s0: int
    n: int
    idx: np.ndarray


@dataclass
class CacheConfig:
    """All knobs of one cache build (recorded verbatim in meta.json)."""

    raw: Path
    out: Path
    name: str
    pretrim: bool = True
    keep_only: bool = True
    seed_min: int = 200000
    skip_gmk: bool = True
    perturb_trim: bool = False
    strict_only: bool = False
    held_only: bool = False
    exclude_seeds: tuple[Path, ...] = field(default_factory=tuple)
    workers: int = 12

    def flags(self) -> dict[str, Any]:
        d = asdict(self)
        d["raw"] = str(self.raw)
        d["out"] = str(self.out)
        d["exclude_seeds"] = [str(p) for p in self.exclude_seeds]
        return d


def repo_root() -> Path:
    """Repository root (``simretract.config`` when available, else the package parent)."""
    try:
        from simretract import config as _cfg

        return Path(_cfg.repo_root())
    except Exception:
        return Path(__file__).resolve().parents[2]


def default_test_pool() -> Path:
    """The shipped held-out seed pool excluded from every training cache
    (``configs/data.yaml: test.exclude_from_training``)."""
    try:
        from simretract import config as _cfg

        name = (_cfg.data_config().get("test") or {}).get("exclude_from_training", "test_pool.json")
        return _cfg.selection_path(name)
    except Exception:
        return repo_root() / "data" / "selections" / "test_pool.json"


def files(d: str, prefix: str, n: int) -> list[str]:
    """First ``n`` ``<prefix>_*.png`` frames of an episode dir, sorted."""
    return sorted(glob.glob(os.path.join(d, f"{prefix}_*.png")))[:n]


def load_excluded_seeds(paths: tuple[Path, ...] | list[Path]) -> set[int]:
    """Union of the ``seeds`` lists of the given selection json files."""
    frozen: set[int] = set()
    for p in paths:
        j = json.load(open(p))
        seeds = {int(s) for s in j["seeds"]}
        frozen |= seeds
        log.info("[CACHE] held-out set %s: %d seeds EXCLUDED", j.get("name", Path(p).name), len(seeds))
    return frozen


def close_cutoff_frames(states: np.ndarray, margin: int = CLOSE_MARGIN) -> int:
    """First frame the carriage is within 0.1 mm of its final width
    (median of the last 20 frames) plus ``margin``; the full length for
    episodes shorter than 40 frames or without a closing motion."""
    g = np.asarray(states)[:, 6]
    if len(g) < 40:
        return len(g)
    arrived = np.nonzero(g - float(np.median(g[-20:])) < 1e-4)[0]
    if not len(arrived):
        return len(g)
    return min(len(g), int(arrived[0]) + margin)


def kept_idx(states: np.ndarray) -> np.ndarray:
    """Arc-length resampling of one (already trimmed) state sequence:
    keep a frame once the motion since the last kept frame reaches one
    quantum (JQ on any joint or GQ on the carriage); the last frame is
    always kept. Returns indices relative to the sequence start."""
    T = len(states)
    if T <= 2:
        return np.arange(T)
    d = np.abs(np.diff(np.asarray(states, np.float32), axis=0))
    step = np.maximum(d[:, :6].max(1) / JQ, d[:, 6] / GQ)
    kept, acc = [0], 0.0
    for t in range(1, T):
        acc += step[t - 1]
        if acc >= 1.0:
            kept.append(t)
            acc = 0.0
    if kept[-1] != T - 1:
        kept.append(T - 1)
    return np.asarray(kept, np.int64)


def scan_episodes(cfg: CacheConfig, frozen: set[int]) -> list[Episode]:
    """Apply the selection and trimming rules to every ``ep_*`` dir under ``cfg.raw``."""
    eps: list[Episode] = []
    for d in sorted(glob.glob(os.path.join(str(cfg.raw), "ep_*"))):
        base = os.path.basename(d)
        mp = os.path.join(d, "meta.npz")
        if not os.path.exists(mp):
            continue
        m = np.load(mp, allow_pickle=True)
        if "verdict" not in m.files:
            log.info("[CACHE] SKIP %s (no verdict)", base)
            continue
        seed = int(m["seed"])
        if seed < cfg.seed_min:
            log.info("[CACHE] SKIP %s (below seed_min %d)", base, cfg.seed_min)
            continue
        if seed in frozen:
            log.info("[CACHE] SKIP %s (HELD-OUT TEST SEED)", base)
            continue
        verdict = json.loads(str(m["verdict"]))
        if cfg.strict_only and not verdict.get("strict_pass"):
            continue
        if cfg.held_only and verdict.get("strict_pass"):
            continue
        T = len(m["states"])
        if cfg.pretrim:
            T = close_cutoff_frames(m["states"])
        s0 = 0
        if cfg.perturb_trim:
            try:
                pw = json.loads(str(m["info"])).get("perturb_win") or {}
                s0 = int(pw.get("end", -1)) + 1
            except Exception:
                s0 = 0
            if s0 <= 0 or s0 >= T - 20:
                log.info("[CACHE] SKIP %s (no usable perturb window)", base)
                continue
        masks = files(d, "mask", T)
        n = min(T, len(masks)) - s0
        if n < 20:
            continue
        probe = [np.asarray(Image.open(masks[s0 + i])).max() for i in (n // 4, n // 2, 3 * n // 4)]
        if max(probe) == 0:
            log.info("[CACHE] SKIP %s (empty tool mask)", base)
            continue
        if cfg.keep_only:
            idx = s0 + kept_idx(m["states"][s0:s0 + n])
        else:
            idx = np.arange(s0, s0 + n)
        eps.append(Episode(d, s0, n, idx))
    return eps


def _decode_episode(args: tuple[int, Episode, int, bool]) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Pool worker: decode one episode's cached frames (no writes)."""
    ei, ep, b0, skip_gmk = args
    d, s0, n, idx = ep
    fr = files(d, "rgb", s0 + n)
    fd = files(d, "depth", s0 + n)
    fm = files(d, "mask", s0 + n)
    fg = files(d, "gmask", s0 + n)
    R = np.stack([np.asarray(Image.open(fr[i]))[..., :3] for i in idx]).astype(np.uint8)
    D = np.stack([np.asarray(Image.open(fd[i])) for i in idx]).astype(np.uint16)
    M = np.stack([np.asarray(Image.open(fm[i])) for i in idx]).astype(np.uint8)
    G = None if skip_gmk else np.stack([np.asarray(Image.open(fg[i])) for i in idx]).astype(np.uint8)
    return ei, b0, R, D, M, G


def _write_frames(cfg: CacheConfig, eps: list[Episode], ranges: list[list[int]]) -> None:
    """Decode episodes in a worker pool and write them from this process
    only (single writer, at most ``2 * workers`` episodes in flight)."""
    out, name = str(cfg.out), cfg.name
    mm = {nm: np.load(f"{out}/{name}_{nm}.npy", mmap_mode="r+") for nm in ("rgb", "dep", "msk", "gmk")}
    tasks = [(i, eps[i], ranges[i][0], cfg.skip_gmk) for i in range(len(eps))]
    batch = 2 * max(1, cfg.workers)
    with Pool(max(1, cfg.workers)) as pool:
        for k in range(0, len(tasks), batch):
            for ei, b0, R, D, M, G in pool.imap_unordered(_decode_episode, tasks[k:k + batch]):
                mm["rgb"][b0:b0 + len(R)] = R
                mm["dep"][b0:b0 + len(D)] = D
                mm["msk"][b0:b0 + len(M)] = M
                if G is not None:
                    mm["gmk"][b0:b0 + len(G)] = G
                del R, D, M, G
                if ei % 25 == 0:
                    log.info("[CACHE] ep %d written", ei)
    for m in mm.values():
        m.flush()
    del mm


def spot_verify(cfg: CacheConfig, eps: list[Episode], ranges: list[list[int]], n_check: int = N_SPOT_CHECK) -> int:
    """Re-decode ``n_check`` random cached frames and count bitwise mismatches."""
    out, name = str(cfg.out), cfg.name
    rgb = np.load(f"{out}/{name}_rgb.npy", mmap_mode="r")
    dep = np.load(f"{out}/{name}_dep.npy", mmap_mode="r")
    msk = np.load(f"{out}/{name}_msk.npy", mmap_mode="r")
    gmk = np.load(f"{out}/{name}_gmk.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(n_check):
        ei = int(rng.integers(len(eps)))
        d, s0, n, idx = eps[ei]
        j = int(rng.integers(len(idx)))
        fi = int(idx[j])
        b0 = ranges[ei][0]
        ok = (
            np.array_equal(rgb[b0 + j], np.asarray(Image.open(files(d, "rgb", s0 + n)[fi]))[..., :3])
            and np.array_equal(dep[b0 + j], np.asarray(Image.open(files(d, "depth", s0 + n)[fi])))
            and np.array_equal(msk[b0 + j], np.asarray(Image.open(files(d, "mask", s0 + n)[fi])))
            and (cfg.skip_gmk or np.array_equal(gmk[b0 + j], np.asarray(Image.open(files(d, "gmask", s0 + n)[fi]))))
        )
        bad += 0 if ok else 1
    return bad


def build_cache(cfg: CacheConfig) -> dict[str, Any]:
    """Build set ``cfg.name`` from ``cfg.raw`` into ``cfg.out``.

    Returns a summary ``{"N", "episodes", "verify_bad", "n_strict"}``. Raises
    ``RuntimeError`` when no episode survives the selection rules.
    """
    out, name = str(cfg.out), cfg.name
    os.makedirs(out, exist_ok=True)
    frozen = load_excluded_seeds(cfg.exclude_seeds) if cfg.exclude_seeds else set()
    if not cfg.exclude_seeds:
        log.warning("[CACHE] no held-out seed pool given: test seeds are NOT excluded")
    eps = scan_episodes(cfg, frozen)
    if not eps:
        raise RuntimeError(f"no episode under {cfg.raw} passed the selection rules")
    ranges: list[list[int]] = []
    a = 0
    for ep in eps:
        ranges.append([a, a + len(ep.idx)])
        a += len(ep.idx)
    N = a
    log.info("[CACHE] %d episodes, %d frames -> %s", len(eps), N, out)

    for nm, dt, shp in (
        ("rgb", np.uint8, (N, FRAME_H, FRAME_W, 3)),
        ("dep", np.uint16, (N, FRAME_H, FRAME_W)),
        ("msk", np.uint8, (N, FRAME_H, FRAME_W)),
        ("gmk", np.uint8, (N, FRAME_H, FRAME_W)),
    ):
        np.lib.format.open_memmap(f"{out}/{name}_{nm}.npy", mode="w+", dtype=dt, shape=shp)

    _write_frames(cfg, eps, ranges)

    st = np.zeros((N, 7), np.float32)
    ac = np.zeros((N, 7), np.float32)
    gp = np.zeros((N, 7), np.float32)
    verdicts: dict[str, dict[str, Any]] = {}
    for i, (d, s0, n, idx) in enumerate(eps):
        m = np.load(os.path.join(d, "meta.npz"), allow_pickle=True)
        b0, b1 = ranges[i]
        st[b0:b1] = m["states"][idx]
        ac[b0:b1] = m["actions"][idx]
        gp[b0:b1] = np.tile(np.asarray(m["grasp_pose"], np.float32), (len(idx), 1))
        v = json.loads(str(m["verdict"]))
        # absent or null when the acceptance battery was skipped -> false
        verdicts[str(i)] = {"strict_pass": bool(v.get("strict_pass")),
                            "battery_skipped": bool(v.get("battery_skipped", False)),
                            "seed": int(m["seed"])}
    np.save(f"{out}/{name}_state.npy", st)
    np.save(f"{out}/{name}_action.npy", ac)
    np.save(f"{out}/{name}_gpose.npy", gp)

    meta: dict[str, Any] = {}
    meta_path = os.path.join(out, "meta.json")
    if os.path.exists(meta_path):  # several sets share one cache dir
        meta = json.load(open(meta_path))
    meta[name] = {
        "N": N,
        "episodes": ranges,
        "prekept": bool(cfg.keep_only),
        "flags": cfg.flags(),
        "n_excluded_seeds": len(frozen),
        "source_episodes": [os.path.basename(ep.path) for ep in eps],
    }
    json.dump(meta, open(meta_path, "w"))
    json.dump(verdicts, open(f"{out}/{name}_verdicts.json", "w"))
    n_strict = sum(1 for r in verdicts.values() if r["strict_pass"])
    log.info("[CACHE] verdicts: %d/%d strict", n_strict, len(eps))

    bad = spot_verify(cfg, eps, ranges)
    log.info("VERIFY %s", "PASS" if bad == 0 else f"FAIL ({bad}/{N_SPOT_CHECK})")
    log.info("CACHE FROM RAW DONE")
    return {"N": N, "episodes": len(eps), "verify_bad": bad, "n_strict": n_strict}


def _md5_prefix(path: Path, n: int = 8) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def fingerprint(cache: str | Path, name: str) -> dict[str, Any]:
    """Frame count and md5[:8] of the action / plane / gpose arrays of a set."""
    cache = Path(cache)
    fp: dict[str, Any] = {"frames": int(json.load(open(cache / "meta.json"))[name]["N"]), "md5": {}}
    for key in ("action", "plane", "gpose"):
        p = cache / f"{name}_{key}.npy"
        fp["md5"][key] = _md5_prefix(p) if p.exists() else None
    return fp


def reference_fingerprint(set_key: str) -> dict[str, Any] | None:
    """Reference fingerprint of ``set_key`` from ``configs/data.yaml``
    (``sets.<key>.fingerprint.{frames, md5_8}``), else the built-in table."""
    try:
        from simretract import config as _cfg

        sets = _cfg.data_config().get("sets") or {}
        if set_key in sets:
            entry = (sets[set_key] or {}).get("fingerprint")
            if entry and "frames" in entry:
                return {"frames": int(entry["frames"]), "md5": dict(entry.get("md5_8") or entry.get("md5") or {})}
            return None  # fingerprint: null -> no reference build recorded for this set
    except Exception as exc:  # unreadable config: fall back to the built-in table
        log.warning("[VERIFY] configs/data.yaml unavailable (%s); using built-in fingerprints", exc)
    return REFERENCE_FINGERPRINTS.get(set_key)


def verify_fingerprint(cache: str | Path, name: str, expected: dict[str, Any]) -> bool:
    """Compare a built set with a reference fingerprint; logs every field."""
    got = fingerprint(cache, name)
    ok = got["frames"] == int(expected["frames"])
    log.info("[VERIFY] frames %d (expected %d) %s", got["frames"], int(expected["frames"]), "OK" if ok else "MISMATCH")
    for key, want in (expected.get("md5") or {}).items():
        have = got["md5"].get(key)
        good = have is not None and want is not None and str(have) == str(want)
        ok &= good
        log.info("[VERIFY] md5 %s %s (expected %s) %s", key, have, want, "OK" if good else "MISMATCH")
    log.info("VERIFY FINGERPRINT %s", "PASS" if ok else "FAIL")
    return ok


def _seed_min_env() -> int | None:
    for name in ("SIMRETRACT_SEED_MIN", "ERA_MIN"):
        if name in os.environ:
            return int(os.environ[name])
    return None


def _flag(p: argparse.ArgumentParser, name: str, env: str, help_: str) -> None:
    """``--name/--no-name`` pair; unset (None) means "use the --set preset", env overrides."""
    default = None
    if env in os.environ:
        default = os.environ[env] == "1"
    dest = name.replace("-", "_")
    p.add_argument(f"--{name}", dest=dest, action="store_true", default=default, help=help_ + f" (env {env}=1)")
    p.add_argument(f"--no-{name}", dest=dest, action="store_false", help=f"disable --{name}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build one frame-cache set from raw episode directories. Flags default to the "
                    "production preset of --set (strict sets: pretrim, keep-only, seed-min 200000, "
                    "skip-gmk, no perturb-trim; recovery sets: the same plus perturb-trim); explicit "
                    "flags and the legacy environment variables override it. Sets: strict, recovery, "
                    "strict_nosmooth, recovery_nosmooth, strict_nostandoff, recovery_nostandoff.",
    )
    p.add_argument("--set", choices=sorted(SET_PRESETS), default="strict",
                   help="cache set: selects the flag preset and the default --name (default strict)")
    p.add_argument("--raw", default=os.environ.get("RAW_ROOT"), required="RAW_ROOT" not in os.environ,
                   help="directory of ep_* raw episode dirs (env RAW_ROOT)")
    p.add_argument("--out", default=os.environ.get("OUT_DIR"), required="OUT_DIR" not in os.environ,
                   help="cache directory; several sets share it (env OUT_DIR)")
    p.add_argument("--name", default=os.environ.get("NAME"), help="set name / file prefix (env NAME; default = --set)")
    p.add_argument("--workers", type=int, default=int(os.environ.get("NW", "12")),
                   help="PNG decode processes (env NW; default 12)")
    p.add_argument("--seed-min", type=int, default=_seed_min_env(),
                   help="skip episodes whose seed is below this (env SIMRETRACT_SEED_MIN or legacy "
                        "ERA_MIN; preset 200000)")
    _flag(p, "pretrim", "PRETRIM", "cache only [0, close_cutoff) frames (preset on)")
    _flag(p, "keep-only", "KEEP_ONLY", "store only arc-length-kept frames and flag the set prekept (preset on)")
    _flag(p, "skip-gmk", "SKIP_GMK", "leave the gripper-mask memmap unwritten, reads as zeros (preset on)")
    _flag(p, "perturb-trim", "PERTURB_TRIM", "start each episode after info.perturb_win.end (preset: recovery only)")
    p.add_argument("--strict-only", action="store_true", default=os.environ.get("STRICT_ONLY") == "1",
                   help="keep only verdict.strict_pass episodes (env STRICT_ONLY; default off)")
    p.add_argument("--held-only", action="store_true", default=os.environ.get("HELD_ONLY") == "1",
                   help="keep only non-strict episodes (env HELD_ONLY; default off)")
    p.add_argument("--exclude-seeds", action="append", default=None, metavar="JSON",
                   help="selection json(s) whose 'seeds' are excluded (repeatable; default the held-out pool "
                        "data/selections/test_pool.json from configs/data.yaml)")
    p.add_argument("--no-exclude-seeds", action="store_true", help="exclude no seeds (never for a training cache)")
    p.add_argument("--plane", dest="plane", action="store_true", default=True,
                   help="compute the plane-normal sidecar after the build (default on)")
    p.add_argument("--no-plane", dest="plane", action="store_false", help="skip the plane sidecar")
    p.add_argument("--verify", action="store_true",
                   help="compare the frame count and md5[:8] fingerprints with the reference build of --set")
    return p


def config_from_args(a: argparse.Namespace) -> CacheConfig:
    """Resolve preset + explicit flags + env fallbacks into a CacheConfig."""
    preset = SET_PRESETS[a.set]
    if a.no_exclude_seeds:
        exclude: tuple[Path, ...] = ()
    elif a.exclude_seeds:
        exclude = tuple(Path(x) for x in a.exclude_seeds)
    else:
        pool = default_test_pool()
        if not pool.exists():
            raise FileNotFoundError(
                f"held-out seed pool {pool} not found; pass --exclude-seeds JSON or --no-exclude-seeds"
            )
        exclude = (pool,)
    pick = lambda v, k: preset[k] if v is None else v  # noqa: E731
    return CacheConfig(
        raw=Path(a.raw), out=Path(a.out), name=a.name or a.set,
        pretrim=pick(a.pretrim, "pretrim"), keep_only=pick(a.keep_only, "keep_only"),
        seed_min=pick(a.seed_min, "seed_min"), skip_gmk=pick(a.skip_gmk, "skip_gmk"),
        perturb_trim=pick(a.perturb_trim, "perturb_trim"),
        strict_only=a.strict_only, held_only=a.held_only, exclude_seeds=exclude, workers=a.workers,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    a = build_parser().parse_args(argv)
    cfg = config_from_args(a)
    log.info("[CACHE] flags: %s", json.dumps(cfg.flags()))
    summary = build_cache(cfg)
    rc = 0 if summary["verify_bad"] == 0 else 1
    if a.plane:
        from simretract.data.plane_sidecar import precompute_plane

        precompute_plane(cfg.out, cfg.name)
    if a.verify:
        ref = reference_fingerprint(a.set)
        if ref is None:
            log.warning("[VERIFY] no reference fingerprint recorded for set %s (configs/data.yaml "
                        "fingerprint: null); the fingerprint comparison is skipped", a.set)
        elif not verify_fingerprint(cfg.out, cfg.name, ref):
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
