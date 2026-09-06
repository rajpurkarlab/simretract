"""Assemble training selections from raw episode directories (no simulator).

Strict set: every recorded strict episode is placed in a grasp-height bin
(the stored world-frame grasp point expressed along the tool axis through
the seed's presented pose), bins tile ``[z_lo, z_hi]`` and each bin keeps
the ``per_bin`` best episodes ranked by keep quality
(``pose_t_mm/10 + pose_r_deg/10``, lower is better). Output: a selection
JSON plus a directory of symlinks to the chosen episodes.

Recovery set: ``exact-N`` keeps the N lowest draw indices per strict seed
(surplus keeps from top-up rounds stay on disk but are not trained), and
``far-N`` takes the N training episodes with the smallest perturbation
fraction (the excursion started farthest from the grasp), ties broken by
episode name.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from simretract.sim.sampling import presented_pose_for_seed

log = logging.getLogger(__name__)

EP_RE = re.compile(r"^ep_(\d+)$")
REC_RE = re.compile(r"^ep_(\d+)_p(\d+)$")


def read_meta(meta_path: str | os.PathLike) -> dict[str, Any]:
    """Load the fields the assembly needs from an episode ``meta.npz``."""
    m = np.load(meta_path, allow_pickle=True)
    info = m["info"]
    info = info.item() if getattr(info, "shape", None) == () else info
    if isinstance(info, (bytes, np.bytes_)):
        info = info.decode()
    if isinstance(info, str):
        info = json.loads(info)
    out = {"seed": int(m["seed"]), "info": dict(info)}
    out["frames"] = int(m["states"].shape[0]) if "states" in m else 0
    return out


def grasp_z_m(seed: int, grasp_w) -> float:
    """Grasp point along the tool axis (m) for ``seed``: the stored world
    grasp point mapped back through the seed's presented pose."""
    pos, rot = presented_pose_for_seed(seed)
    return float(rot.inv().apply(np.asarray(grasp_w, dtype=float) - pos)[2])


def recipe_summary(kind: str = "strict") -> dict[str, Any]:
    """Scalar recorder knobs of ``configs/generation/<kind>.yaml`` (the
    settings the set was recorded under), stored in the selection JSON as ``recipe``.
    Output-only keys are left out; ``{}`` when the file is unavailable."""
    from simretract import config

    try:
        raw = config.generation_recipe(kind)
    except config.ConfigError:
        return {}
    return {k: v for k, v in raw.items()
            if isinstance(v, (bool, int, float, str)) and k not in ("sample_every",)}


def bin_edges(z_lo: float, z_hi: float, bins: int) -> list[float]:
    return [z_lo + (z_hi - z_lo) * i / bins for i in range(bins + 1)]


def bin_names(z_lo: float, z_hi: float, bins: int) -> list[str]:
    e = bin_edges(z_lo, z_hi, bins)

    def cm(v):
        s = f"{v * 100:+.1f}".rstrip("0").rstrip(".")
        return s.replace("+0", "0") if s in ("+0", "-0") else s
    return [f"{cm(e[i])}..{cm(e[i + 1])}" for i in range(bins)]


def bin_index(z: float, z_lo: float, z_hi: float, bins: int) -> int | None:
    """Bin of ``z`` or None when outside ``[z_lo, z_hi]``."""
    if not (z_lo <= z <= z_hi):
        return None
    return min(bins - 1, int((z - z_lo) / ((z_hi - z_lo) / bins)))


def scan_strict_raw(raw_dir: str | os.PathLike) -> list[dict[str, Any]]:
    """Rows for every ``ep_<seed>/meta.npz`` under ``raw_dir``."""
    raw_dir = Path(raw_dir)
    rows = []
    for name in sorted(os.listdir(raw_dir)):
        m = EP_RE.match(name)
        mp = raw_dir / name / "meta.npz"
        if not m or not mp.is_file():
            continue
        meta = read_meta(mp)
        info = meta["info"]
        seed = meta["seed"]
        z = grasp_z_m(seed, info["grasp_w"])
        q = float(info.get("pose_t_mm", 0)) / 10.0 + float(info.get("pose_r_deg", 0)) / 10.0
        rows.append({"seed": seed, "name": name, "z_m": z, "quality": q,
                     "frames": meta["frames"]})
    return rows


def select_strict(rows: list[dict[str, Any]], per_bin: int, bins: int,
                  z_lo: float, z_hi: float) -> tuple[list[dict], list[dict], dict[str, int]]:
    """Quota selection: (chosen, trimmed, short_bins)."""
    names = bin_names(z_lo, z_hi, bins)
    in_range = []
    for r in rows:
        b = bin_index(r["z_m"], z_lo, z_hi, bins)
        if b is None:
            continue
        in_range.append(dict(r, bin=b))
    chosen, trimmed, short = [], [], {}
    for b in range(bins):
        # stable sort: ties keep directory (name) order
        cand = sorted([r for r in in_range if r["bin"] == b], key=lambda r: r["quality"])
        take = cand[:per_bin]
        chosen += take
        trimmed += cand[per_bin:]
        if len(take) < per_bin:
            short[names[b]] = per_bin - len(take)
        log.info("bin %9s: available %3d -> taking %3d", names[b], len(cand), len(take))
    return chosen, trimmed, short


def link_episodes(raw_dir: str | os.PathLike, names: list[str], out_dir: str | os.PathLike) -> int:
    """Symlink ``raw_dir/<name>`` into ``out_dir`` for every name; returns
    the number of links created (existing entries are left alone)."""
    raw_dir, out_dir = Path(raw_dir).resolve(), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in names:
        dst = out_dir / name
        if dst.exists() or dst.is_symlink():
            continue
        os.symlink(raw_dir / name, dst)
        n += 1
    return n


def assemble_strict(raw_dir: str | os.PathLike, out_json: str | os.PathLike, *,
                    per_bin: int = 125, bins: int = 4, z_lo: float = -0.05, z_hi: float = 0.05,
                    train_dir: str | os.PathLike | None = None,
                    allow_short: bool = False,
                    recipe: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the strict assembly and write the selection JSON (+ symlink dir).
    ``recipe`` defaults to :func:`recipe_summary` of the strict recipe."""
    rows = scan_strict_raw(raw_dir)
    chosen, trimmed, short = select_strict(rows, per_bin, bins, z_lo, z_hi)
    if short and not allow_short:
        raise SystemExit(f"short bins {short}: not assembling (pass --allow-short to force)")
    out = {
        "name": Path(out_json).stem,
        "seeds": sorted(r["seed"] for r in chosen),
        "per_bin": per_bin,
        "bins_cm": bin_names(z_lo, z_hi, bins),
        "rule": (f"{per_bin} episodes per grasp-height bin over [{z_lo * 100:g}, {z_hi * 100:g}] cm "
                 "along the tool axis, ranked by pose_t_mm/10 + pose_r_deg/10 (lower is better)"),
        "recipe": recipe_summary("strict") if recipe is None else dict(recipe),
        "grasp_z_cm": {str(r["seed"]): round(r["z_m"] * 100, 2) for r in chosen},
        "trimmed_out": sorted(r["seed"] for r in trimmed),
        "short_bins": short,
        "n_recorded_total": len(rows),
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=1)
    if train_dir is not None:
        n = link_episodes(raw_dir, [r["name"] for r in chosen], train_dir)
        log.info("train dir %s: %d new links", train_dir, n)
    if chosen:
        log.info("wrote %s: %d seeds | trimmed %d | mean frames/episode %d", out_json,
                 len(out["seeds"]), len(trimmed), int(np.mean([r["frames"] for r in chosen])))
    return out


def scan_recovery_raw(raw_dir: str | os.PathLike) -> dict[int, list[dict[str, Any]]]:
    """Kept recovery episodes per strict seed: seed -> [{draw, name, frac}]."""
    raw_dir = Path(raw_dir)
    per: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for name in sorted(os.listdir(raw_dir)):
        m = REC_RE.match(name)
        mp = raw_dir / name / "meta.npz"
        if not m or not mp.is_file():
            continue
        info = read_meta(mp)["info"]
        frac = (info.get("perturb") or {}).get("frac")
        per[int(m.group(1))].append({"draw": int(m.group(2)), "name": name,
                                     "frac": None if frac is None else float(frac)})
    return dict(per)


def select_recovery_exact(per_seed: dict[int, list[dict[str, Any]]], seeds: list[int],
                          exact: int) -> tuple[list[dict[str, Any]], list[str]]:
    """Lowest ``exact`` draw indices per seed -> (train rows, surplus names)."""
    train, surplus = [], []
    for s in sorted(seeds):
        ds = sorted(per_seed.get(s, []), key=lambda d: d["draw"])
        train += ds[:exact]
        surplus += [d["name"] for d in ds[exact:]]
    return train, sorted(surplus)


def select_far(rows: list[dict[str, Any]], n: int) -> tuple[list[str], float | None]:
    """``n`` episodes with the smallest perturbation fraction (ties by name).
    Returns (sorted names, largest selected fraction)."""
    with_frac = [r for r in rows if r.get("frac") is not None]
    ranked = sorted(with_frac, key=lambda r: (r["frac"], r["name"]))[:n]
    if not ranked:
        return [], None
    return sorted(r["name"] for r in ranked), max(r["frac"] for r in ranked)


def keep_histogram(per_seed: dict[int, list], seeds: list[int], top: int = 6) -> list[int]:
    """Number of seeds with 0, 1, ..., top-1 keeps."""
    return [sum(1 for s in seeds if len(per_seed.get(s, [])) == k) for k in range(top)]


def assemble_recovery(raw_dir: str | os.PathLike, strict_json: str | os.PathLike, *,
                      exact: int = 4, far: int = 1000,
                      out_train: str | os.PathLike | None = None,
                      out_far: str | os.PathLike | None = None,
                      train_dir: str | os.PathLike | None = None,
                      far_dir: str | os.PathLike | None = None) -> dict[str, Any]:
    """Exact-N training selection and far-N subset; writes the JSONs and
    optional symlink dirs."""
    with open(strict_json) as f:
        seeds = sorted(int(s) for s in json.load(f)["seeds"])
    per = scan_recovery_raw(raw_dir)
    hist = keep_histogram(per, seeds)
    log.info("per-seed keeps 0..5: %s | seeds with >=%d: %d/%d | total keeps %d", hist, exact,
             sum(1 for s in seeds if len(per.get(s, [])) >= exact), len(seeds),
             sum(len(v) for v in per.values()))
    train, surplus = select_recovery_exact(per, seeds, exact)
    train_names = sorted(r["name"] for r in train)
    result: dict[str, Any] = {"train_episodes": train_names, "surplus": surplus}
    if out_train is not None:
        Path(out_train).parent.mkdir(parents=True, exist_ok=True)
        with open(out_train, "w") as f:
            json.dump({"name": Path(out_train).stem,
                       "rule": (f"per strict seed, the lowest {exact} draw indices with a kept "
                                "recovery episode (ep_<seed>_p<draw>); surplus = kept draws beyond "
                                f"{exact}"),
                       "draws_per_seed": exact, "train_episodes": train_names, "surplus": surplus,
                       "perturb_frac": {r["name"]: r["frac"] for r in train if r["frac"] is not None}},
                      f, indent=1)
    if train_dir is not None:
        link_episodes(raw_dir, train_names, train_dir)
    if far > 0:
        far_names, frac_max = select_far(train, far)
        covered = {n.split("_")[1] for n in far_names}
        result.update({"far_episodes": far_names, "frac_max": frac_max})
        log.info("far-%d: %d episodes, frac_max %s, seeds covered %d/%d", far, len(far_names),
                 None if frac_max is None else round(frac_max, 4), len(covered), len(seeds))
        if out_far is not None:
            Path(out_far).parent.mkdir(parents=True, exist_ok=True)
            with open(out_far, "w") as f:
                json.dump({"name": Path(out_far).stem, "episodes": far_names,
                           "rule": (f"{far} recovery training episodes with the smallest "
                                    "perturbation fraction (info.perturb.frac)"),
                           "frac_max": frac_max, "n": len(far_names),
                           "seeds_covered": len(covered)}, f, indent=1)
        if far_dir is not None:
            link_episodes(raw_dir, far_names, far_dir)
    return result


def add_strict_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--raw", required=True, help="raw strict episode root (ep_<seed>/)")
    ap.add_argument("--out", required=True, help="selection JSON to write")
    ap.add_argument("--per-bin", type=int, default=125, help="episodes kept per bin")
    ap.add_argument("--bins", type=int, default=4, help="number of grasp-height bins")
    ap.add_argument("--z-range", type=float, nargs=2, default=(-0.05, 0.05),
                    metavar=("Z_LO", "Z_HI"), help="grasp height range along the tool axis (m)")
    ap.add_argument("--train-dir", default=None,
                    help="directory of symlinks to the chosen episodes")
    ap.add_argument("--allow-short", action="store_true",
                    help="write the selection even when a bin is short of --per-bin")


def add_recovery_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--raw", required=True, help="raw recovery episode root (ep_<seed>_p<draw>/)")
    ap.add_argument("--strict-json", required=True, help="strict selection defining the seeds")
    ap.add_argument("--exact", type=int, default=4, help="kept draws per seed (lowest indices)")
    ap.add_argument("--far", type=int, default=1000,
                    help="size of the far subset (smallest perturbation fraction); 0 = skip")
    ap.add_argument("--out-train", default=None, help="exact-N selection JSON")
    ap.add_argument("--out-far", default=None, help="far-N selection JSON")
    ap.add_argument("--train-dir", default=None, help="symlink dir for the exact-N set")
    ap.add_argument("--far-dir", default=None, help="symlink dir for the far-N set")


def run_strict(args: argparse.Namespace) -> int:
    assemble_strict(args.raw, args.out, per_bin=args.per_bin, bins=args.bins,
                    z_lo=args.z_range[0], z_hi=args.z_range[1], train_dir=args.train_dir,
                    allow_short=args.allow_short)
    return 0


def run_recovery(args: argparse.Namespace) -> int:
    assemble_recovery(args.raw, args.strict_json, exact=args.exact, far=args.far,
                      out_train=args.out_train, out_far=args.out_far,
                      train_dir=args.train_dir, far_dir=args.far_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    add_strict_args(sub.add_parser("strict", help="quota-balanced strict selection"))
    add_recovery_args(sub.add_parser("recovery", help="exact-N and far-N recovery selections"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return run_strict(args) if args.mode == "strict" else run_recovery(args)


if __name__ == "__main__":
    raise SystemExit(main())
