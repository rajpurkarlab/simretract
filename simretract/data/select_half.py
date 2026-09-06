"""Deterministic half-size dataset selection (the ``half_data`` ablation).

The full training data is the strict-500 set plus its far-1000 recovery
draws. The half-data row keeps the same balance at half the size:

strict
    250 of the 500 seeds, stratified by the 2.5 cm grasp-height bin of the
    generation recipe (bin = ``min(3, max(0, int((z_cm + 5) / 2.5)))``),
    drawn without replacement inside each bin with ``numpy`` generator seed
    0. With four bins of 125 the quota is 63/62/63/62.

recovery
    the far-set episodes whose seed is among the selected strict seeds
    (the far set consists of recovery draws OF the strict seeds), then
    trimmed or topped up to exactly ``n_far``: trimming drops the
    highest-index draw of the seeds with the most draws, round-robin;
    topping up adds far episodes of unselected seeds at random.

The result is a subset mapping ``{set_name: [episode index, ...]}`` in
cache episode order, which the trainer applies before its train/val split
(normalisation statistics still come from the full sets), plus a seed-level
selection record. Episode indices are resolved through the per-set
``<name>_verdicts.json`` written by the cache builder, so the selection is
tied to a specific cache build order.
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

N_BINS = 4
BIN_ORDER_FOR_EXTRAS = (0, 2, 1, 3)
"""Bins receiving the remainder when the strict quota does not divide by 4."""


def grasp_bin(z_cm: float) -> int:
    """2.5 cm grasp-height bin over [-5, +5) cm, clamped to 0..3."""
    return min(N_BINS - 1, max(0, int((z_cm + 5) / 2.5)))


def strict_quota(n_strict: int) -> dict[int, int]:
    """Per-bin quota: an even split, extras to bins 0, 2, 1, 3 in that order
    (250 -> {0: 63, 1: 62, 2: 63, 3: 62})."""
    base, extra = divmod(n_strict, N_BINS)
    want = {b: base for b in range(N_BINS)}
    for b in BIN_ORDER_FOR_EXTRAS[:extra]:
        want[b] += 1
    return want


def select_half(
    cache: str | Path,
    strict_json: str | Path,
    *,
    strict_set: str = "strict",
    far_set: str = "recovery",
    n_strict: int = 250,
    n_far: int = 500,
    rng_seed: int = 0,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """Compute the half-data subset for a cache.

    Args:
        cache: cache directory holding ``<strict_set>_verdicts.json`` and
            ``<far_set>_verdicts.json`` (episode index -> seed).
        strict_json: the strict selection file with ``seeds`` and
            ``grasp_z_cm`` (seed -> grasp height in cm).
        strict_set, far_set: cache set names.
        n_strict, n_far: target episode counts.
        rng_seed: generator seed (0 reproduces the shipped row).

    Returns:
        ``(subset, selection)`` where ``subset`` is
        ``{strict_set: [idx...], far_set: [idx...]}`` (sorted indices) and
        ``selection`` records the rule, selected seeds and bin histograms.
    """
    cache = Path(cache)
    spec = json.load(open(strict_json))
    z = {int(k): float(v) for k, v in spec["grasp_z_cm"].items()}
    rng = np.random.default_rng(rng_seed)

    by_bin: dict[int, list[int]] = collections.defaultdict(list)
    for s in spec["seeds"]:
        by_bin[grasp_bin(z[int(s)])].append(int(s))
    want = strict_quota(n_strict)
    sel_seeds: set[int] = set()
    for b, seeds in sorted(by_bin.items()):
        pick = rng.choice(sorted(seeds), size=want[b], replace=False)
        sel_seeds |= {int(x) for x in pick}
    if len(sel_seeds) != n_strict:
        raise RuntimeError(f"selected {len(sel_seeds)} strict seeds, expected {n_strict}")

    vs = json.load(open(cache / f"{strict_set}_verdicts.json"))
    vf = json.load(open(cache / f"{far_set}_verdicts.json"))
    strict_idx = sorted(int(i) for i, r in vs.items() if int(r["seed"]) in sel_seeds)
    if len(strict_idx) != n_strict:
        raise RuntimeError(
            f"{strict_set}_verdicts.json holds {len(strict_idx)} of the {n_strict} selected seeds"
        )

    far_by_seed: dict[int, list[int]] = collections.defaultdict(list)
    for i, r in vf.items():
        far_by_seed[int(r["seed"])].append(int(i))
    far_sel: list[int] = []
    for s in sorted(sel_seeds):
        far_sel += sorted(far_by_seed.get(s, []))
    log.info("far episodes of the %d selected seeds: %d", n_strict, len(far_sel))
    if len(far_sel) > n_far:
        # drop extras round-robin from the seeds with the most draws
        per = collections.Counter(int(vf[str(i)]["seed"]) for i in far_sel)
        drop = len(far_sel) - n_far
        order = sorted(per, key=lambda s: -per[s])
        dropped = []
        while drop > 0:
            for s in order:
                if drop == 0:
                    break
                if per[s] > 1:
                    cand = max(i for i in far_sel if int(vf[str(i)]["seed"]) == s)
                    far_sel.remove(cand)
                    dropped.append(cand)
                    per[s] -= 1
                    drop -= 1
        log.info("trimmed %d extra draws", len(dropped))
    elif len(far_sel) < n_far:
        need = n_far - len(far_sel)
        chosen = set(far_sel)
        pool = [int(i) for i in vf if int(i) not in chosen and int(vf[i]["seed"]) not in sel_seeds]
        add = rng.choice(sorted(pool), size=need, replace=False)
        far_sel += [int(x) for x in add]
        log.info("topped up %d from unselected seeds", need)
    far_sel = sorted(far_sel)
    if len(far_sel) != n_far:
        raise RuntimeError(f"far selection holds {len(far_sel)} episodes, expected {n_far}")

    bins_s = collections.Counter(grasp_bin(z[int(vs[str(i)]["seed"])]) for i in strict_idx)
    bins_f = collections.Counter(
        grasp_bin(z[int(vf[str(i)]["seed"])]) for i in far_sel if int(vf[str(i)]["seed"]) in z
    )
    seeds_f = len({int(vf[str(i)]["seed"]) for i in far_sel})
    subset = {strict_set: strict_idx, far_set: far_sel}
    selection = {
        "rule": (
            f"half-size balanced: {n_strict}/{len(spec['seeds'])} strict seeds stratified by "
            f"2.5 cm grasp-z bin (RNG {rng_seed}), far-{n_far} = the far draws of those seeds "
            f"trimmed/topped to {n_far}"
        ),
        "strict_seeds": sorted(sel_seeds),
        "strict_bins": {int(k): int(v) for k, v in sorted(bins_s.items())},
        "far_bins": {int(k): int(v) for k, v in sorted(bins_f.items())},
        "far_seed_coverage": seeds_f,
    }
    log.info(
        "HALF SELECTION DONE: strict %d (bins %s) | far %d over %d seeds (bins %s)",
        len(strict_idx), selection["strict_bins"], len(far_sel), seeds_f, selection["far_bins"],
    )
    return subset, selection


def write_selection(
    out_prefix: str | Path, subset: dict[str, list[int]], selection: dict[str, Any]
) -> tuple[Path, Path]:
    """Write ``<prefix>_subset.json`` (trainer input) and ``<prefix>_selection.json``."""
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    subset_path = out_prefix.with_name(out_prefix.name + "_subset.json")
    sel_path = out_prefix.with_name(out_prefix.name + "_selection.json")
    json.dump(subset, open(subset_path, "w"))
    json.dump(selection, open(sel_path, "w"), indent=1)
    return subset_path, sel_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Deterministic half-size (strict 250 + far 500) subset selection.")
    p.add_argument("--cache", required=True, help="cache directory with the per-set verdict jsons")
    p.add_argument("--strict-json", required=True, help="strict selection json (seeds + grasp_z_cm)")
    p.add_argument("--out", required=True, help="output prefix: writes <out>_subset.json and <out>_selection.json")
    p.add_argument("--strict-set", default="strict", help="strict cache set name (default strict)")
    p.add_argument("--far-set", default="recovery", help="recovery cache set name (default recovery)")
    p.add_argument("--n-strict", type=int, default=250, help="strict episodes to keep (default 250)")
    p.add_argument("--n-far", type=int, default=500, help="recovery episodes to keep (default 500)")
    p.add_argument("--rng-seed", type=int, default=0, help="generator seed (default 0 = the shipped row)")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    a = build_parser().parse_args(argv)
    subset, selection = select_half(
        a.cache, a.strict_json, strict_set=a.strict_set, far_set=a.far_set,
        n_strict=a.n_strict, n_far=a.n_far, rng_seed=a.rng_seed,
    )
    sp, lp = write_selection(a.out, subset, selection)
    log.info("wrote %s and %s", sp, lp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
