"""The half-data selection is deterministic and balanced.

On a shipped strict-500 selection with a synthetic cache order the strict
bins must be 63/62/63/62, and with a synthetic far set the recovery half
must contain exactly 500 draws over the selected seeds.  The selection is
parametrised by the cache set names, so it runs unchanged on the
no-smoothing and no-standoff pairs.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import pytest

from simretract.data.select_half import grasp_bin, select_half, strict_quota, write_selection

ROOT = Path(__file__).resolve().parents[2]


def strict_json() -> Path:
    p = ROOT / "data" / "selections" / "strict_500.json"
    if p.exists():
        return p
    pytest.skip("strict-500 selection json not shipped")


def make_cache(tmp_path: Path, seeds: list[int], draws_per_seed: dict[int, int], strict_set="strict", far_set="recovery"):
    """Synthetic verdict jsons: strict in a shuffled cache order, far draws grouped by seed."""
    order = list(seeds)
    np.random.default_rng(42).shuffle(order)
    vs = {str(i): {"strict_pass": False, "seed": int(s)} for i, s in enumerate(order)}
    vf = {}
    k = 0
    for s in sorted(seeds):
        for _ in range(draws_per_seed.get(s, 0)):
            vf[str(k)] = {"strict_pass": False, "seed": int(s)}
            k += 1
    json.dump(vs, open(tmp_path / f"{strict_set}_verdicts.json", "w"))
    json.dump(vf, open(tmp_path / f"{far_set}_verdicts.json", "w"))
    return vs, vf


def test_quota_pattern():
    assert strict_quota(250) == {0: 63, 1: 62, 2: 63, 3: 62}
    assert strict_quota(248) == {0: 62, 1: 62, 2: 62, 3: 62}
    assert grasp_bin(-4.9) == 0 and grasp_bin(-2.4) == 1 and grasp_bin(0.1) == 2 and grasp_bin(4.9) == 3
    assert grasp_bin(-9) == 0 and grasp_bin(9) == 3


def test_strict_bins_and_far_500(tmp_path):
    sj = strict_json()
    spec = json.load(open(sj))
    seeds = [int(s) for s in spec["seeds"]]
    assert len(seeds) == 500
    # two recovery draws per seed -> the far-1000 analogue; no trim / top-up needed
    vs, vf = make_cache(tmp_path, seeds, {s: 2 for s in seeds})
    subset, sel = select_half(tmp_path, sj)
    assert len(subset["strict"]) == 250 and len(subset["recovery"]) == 500
    assert sel["strict_bins"] == {0: 63, 1: 62, 2: 63, 3: 62}
    z = {int(k): v for k, v in spec["grasp_z_cm"].items()}
    bins = collections.Counter(grasp_bin(z[vs[str(i)]["seed"]]) for i in subset["strict"])
    assert dict(bins) == {0: 63, 1: 62, 2: 63, 3: 62}
    chosen = set(sel["strict_seeds"])
    assert len(chosen) == 250
    assert all(vs[str(i)]["seed"] in chosen for i in subset["strict"])
    far_seeds = {vf[str(i)]["seed"] for i in subset["recovery"]}
    assert far_seeds <= chosen and sel["far_seed_coverage"] == 250
    assert subset["strict"] == sorted(subset["strict"]) and subset["recovery"] == sorted(subset["recovery"])
    # deterministic across runs
    subset2, sel2 = select_half(tmp_path, sj)
    assert subset2 == subset and sel2["strict_seeds"] == sel["strict_seeds"]


def test_far_trim_and_topup(tmp_path):
    sj = strict_json()
    seeds = [int(s) for s in json.load(open(sj))["seeds"]]
    # uneven draw counts (1..4 per seed, 1250 far episodes): trimming path
    draws = {s: 1 + (i % 4) for i, s in enumerate(sorted(seeds))}
    vs, vf = make_cache(tmp_path, seeds, draws)
    subset, sel = select_half(tmp_path, sj)
    chosen = set(sel["strict_seeds"])
    assert len(subset["recovery"]) == 500
    assert {vf[str(i)]["seed"] for i in subset["recovery"]} <= chosen
    # every selected seed with a draw keeps at least one (round-robin trim)
    per = collections.Counter(vf[str(i)]["seed"] for i in subset["recovery"])
    assert all(per[s] >= 1 for s in chosen if draws[s] > 0)
    # one draw per seed (500 far): the 250 selected give 250 -> top-up from unselected seeds
    vs, vf = make_cache(tmp_path, seeds, {s: 1 for s in seeds})
    subset, sel = select_half(tmp_path, sj)
    assert len(subset["recovery"]) == 500
    far_seeds = {vf[str(i)]["seed"] for i in subset["recovery"]}
    assert len(far_seeds & chosen) == 250 and len(far_seeds - chosen) == 250


@pytest.mark.parametrize("selection, strict_set, far_set", [
    ("strict_500_no_smoothing.json", "strict_nosmooth", "recovery_nosmooth"),
    ("strict_500_no_standoff.json", "strict_nostandoff", "recovery_nostandoff"),
])
def test_half_selection_on_the_variant_pairs(tmp_path, selection, strict_set, far_set):
    sj = ROOT / "data" / "selections" / selection
    spec = json.load(open(sj))
    seeds = [int(s) for s in spec["seeds"]]
    assert len(seeds) == 500 and set(spec["grasp_z_cm"]) == {str(s) for s in seeds}
    vs, vf = make_cache(tmp_path, seeds, {s: 2 for s in seeds}, strict_set=strict_set, far_set=far_set)
    subset, sel = select_half(tmp_path, sj, strict_set=strict_set, far_set=far_set)
    assert set(subset) == {strict_set, far_set}
    assert len(subset[strict_set]) == 250 and len(subset[far_set]) == 500
    assert sel["strict_bins"] == {0: 63, 1: 62, 2: 63, 3: 62}
    chosen = set(sel["strict_seeds"])
    assert {vf[str(i)]["seed"] for i in subset[far_set]} <= chosen and sel["far_seed_coverage"] == 250
    subset2, _ = select_half(tmp_path, sj, strict_set=strict_set, far_set=far_set)
    assert subset2 == subset


def test_write_selection(tmp_path):
    sj = strict_json()
    seeds = [int(s) for s in json.load(open(sj))["seeds"]]
    make_cache(tmp_path, seeds, {s: 2 for s in seeds})
    subset, sel = select_half(tmp_path, sj)
    sp, lp = write_selection(tmp_path / "half", subset, sel)
    assert json.load(open(sp)) == subset
    assert json.load(open(lp))["strict_bins"] == {"0": 63, "1": 62, "2": 63, "3": 62}
