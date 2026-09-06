"""Selection rules of simretract.sim.assemble on synthetic episodes."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from simretract.sim import assemble
from simretract.sim.sampling import presented_pose_for_seed

SELECTIONS = Path(__file__).resolve().parents[2] / "data" / "selections"


def write_meta(ep_dir: Path, seed: int, info: dict, frames: int = 300) -> None:
    ep_dir.mkdir(parents=True, exist_ok=True)
    np.savez(ep_dir / "meta.npz", states=np.zeros((frames, 7), np.float32),
             actions=np.zeros((frames, 7), np.float32), seed=seed,
             info=json.dumps(info), verdict=json.dumps({"battery_skipped": True}))


def grasp_w_at(seed: int, z_tool: float) -> list[float]:
    """World grasp point that sits at ``z_tool`` along the presented tool axis."""
    pos, rot = presented_pose_for_seed(seed)
    return (pos + rot.apply([0.004, -0.002, z_tool])).tolist()


def make_strict_raw(root: Path, spec: list[tuple[int, float, float, float]]) -> None:
    """spec rows: (seed, z_tool_m, pose_t_mm, pose_r_deg)."""
    for seed, z, pt, pr in spec:
        write_meta(root / f"ep_{seed}", seed,
                   {"grasp_w": grasp_w_at(seed, z), "R_grasp": np.eye(3).tolist(),
                    "gap_mm": 5.0, "drop_mm": 0.2, "pose_t_mm": pt, "pose_r_deg": pr})


def test_grasp_z_roundtrip():
    for seed, z in ((200005, -0.031), (400851, 0.049), (601001, 0.0)):
        assert assemble.grasp_z_m(seed, grasp_w_at(seed, z)) == pytest.approx(z, abs=1e-9)


def test_bins_names_and_index():
    assert assemble.bin_names(-0.05, 0.05, 4) == ["-5..-2.5", "-2.5..0", "0..+2.5", "+2.5..+5"]
    assert assemble.bin_index(-0.05, -0.05, 0.05, 4) == 0
    assert assemble.bin_index(-0.0251, -0.05, 0.05, 4) == 0
    assert assemble.bin_index(-0.024, -0.05, 0.05, 4) == 1
    assert assemble.bin_index(0.05, -0.05, 0.05, 4) == 3
    assert assemble.bin_index(0.0501, -0.05, 0.05, 4) is None
    assert assemble.bin_index(-0.06, -0.05, 0.05, 4) is None


def test_strict_quota_selection(tmp_path):
    raw = tmp_path / "raw"
    # bin 0: 3 episodes (quota 2 -> worst quality trimmed); bin 1: 2; bin 2: 2;
    # bin 3: 1 (short); one episode outside the range is ignored
    rows = [
        (100, -0.045, 1.0, 1.0), (101, -0.030, 9.0, 9.0), (102, -0.040, 2.0, 2.0),
        (200, -0.020, 3.0, 0.0), (201, -0.001, 0.5, 0.5),
        (300, 0.010, 4.0, 1.0), (301, 0.024, 0.0, 0.0),
        (400, 0.040, 1.0, 1.0),
        (500, 0.080, 0.0, 0.0),
    ]
    make_strict_raw(raw, rows)
    scanned = assemble.scan_strict_raw(raw)
    assert [r["seed"] for r in scanned] == sorted(r[0] for r in rows)
    chosen, trimmed, short = assemble.select_strict(scanned, per_bin=2, bins=4, z_lo=-0.05, z_hi=0.05)
    assert sorted(r["seed"] for r in chosen) == [100, 102, 200, 201, 300, 301, 400]
    assert [r["seed"] for r in trimmed] == [101]
    assert short == {"+2.5..+5": 1}
    # quality ranks within a bin: lower pose error first
    b0 = [r["seed"] for r in chosen if r["bin"] == 0]
    assert b0 == [100, 102]

    out_json = tmp_path / "sel.json"
    with pytest.raises(SystemExit):
        assemble.assemble_strict(raw, out_json, per_bin=2, bins=4)
    sel = assemble.assemble_strict(raw, out_json, per_bin=2, bins=4, train_dir=tmp_path / "train",
                                   allow_short=True)
    saved = json.loads(out_json.read_text())
    assert saved["seeds"] == sel["seeds"] == [100, 102, 200, 201, 300, 301, 400]
    assert saved["per_bin"] == 2 and saved["trimmed_out"] == [101]
    assert saved["n_recorded_total"] == 9 and saved["short_bins"] == {"+2.5..+5": 1}
    assert saved["grasp_z_cm"]["100"] == pytest.approx(-4.5, abs=0.01)
    links = sorted(os.listdir(tmp_path / "train"))
    assert links == [f"ep_{s}" for s in saved["seeds"]]
    assert all(os.path.islink(tmp_path / "train" / n) for n in links)
    assert (tmp_path / "train" / "ep_100" / "meta.npz").is_file()


def test_strict_quality_ties_keep_name_order(tmp_path):
    raw = tmp_path / "raw"
    make_strict_raw(raw, [(300, -0.04, 1.0, 1.0), (105, -0.045, 1.0, 1.0), (207, -0.03, 1.0, 1.0)])
    chosen, trimmed, _ = assemble.select_strict(assemble.scan_strict_raw(raw), 2, 4, -0.05, 0.05)
    assert [r["seed"] for r in chosen] == [105, 207] and [r["seed"] for r in trimmed] == [300]


def make_recovery_raw(root: Path, keeps: dict[int, dict[int, float]]) -> None:
    """keeps: seed -> {draw: frac}; a negative frac writes a .filtered marker."""
    root.mkdir(parents=True, exist_ok=True)
    for seed, draws in keeps.items():
        for draw, frac in draws.items():
            name = f"ep_{seed}_p{draw}"
            if frac < 0:
                (root / f"{name}.filtered").touch()
                continue
            write_meta(root / name, seed,
                       {"grasp_w": [0.4, 0.0, 0.2], "R_grasp": np.eye(3).tolist(),
                        "perturb": {"frac": frac, "dpos": [0, 0, 0], "drot": [0, 0, 0], "grip": 0.0},
                        "perturb_win": {"start": 0, "end": 30}, "pose_t_mm": 1.0, "pose_r_deg": 1.0})


def test_recovery_exact_and_far(tmp_path):
    raw = tmp_path / "rec"
    strict_json = tmp_path / "strict.json"
    seeds = [10, 20, 30, 40]
    strict_json.write_text(json.dumps({"seeds": seeds}))
    make_recovery_raw(raw, {
        10: {0: 0.30, 1: 0.70, 2: 0.45, 3: 0.60, 5: 0.26},   # 5 keeps -> p5 surplus
        20: {0: 0.50, 1: -1, 2: 0.31, 3: 0.72},              # p1 filtered -> 3 keeps
        30: {1: 0.28, 4: 0.90, 6: 0.33, 7: 0.29},           # 4 keeps, gap draws
        # seed 40: nothing recorded
    })
    per = assemble.scan_recovery_raw(raw)
    assert sorted(per) == [10, 20, 30]
    assert assemble.keep_histogram(per, seeds) == [1, 0, 0, 1, 1, 1]
    train, surplus = assemble.select_recovery_exact(per, seeds, 4)
    assert sorted(r["name"] for r in train) == [
        "ep_10_p0", "ep_10_p1", "ep_10_p2", "ep_10_p3",
        "ep_20_p0", "ep_20_p2", "ep_20_p3",
        "ep_30_p1", "ep_30_p4", "ep_30_p6", "ep_30_p7"]
    assert surplus == ["ep_10_p5"]

    far, frac_max = assemble.select_far(train, 5)
    assert far == ["ep_10_p0", "ep_20_p2", "ep_30_p1", "ep_30_p6", "ep_30_p7"]
    assert frac_max == pytest.approx(0.33)
    # ties broken by name
    tie_rows = [{"name": "ep_2_p0", "frac": 0.3}, {"name": "ep_1_p0", "frac": 0.3},
                {"name": "ep_3_p0", "frac": 0.1}]
    assert assemble.select_far(tie_rows, 2)[0] == ["ep_1_p0", "ep_3_p0"]
    assert assemble.select_far([], 3) == ([], None)

    res = assemble.assemble_recovery(
        raw, strict_json, exact=4, far=5, out_train=tmp_path / "train.json",
        out_far=tmp_path / "far.json", train_dir=tmp_path / "train_dir", far_dir=tmp_path / "far_dir")
    tj = json.loads((tmp_path / "train.json").read_text())
    fj = json.loads((tmp_path / "far.json").read_text())
    assert tj["train_episodes"] == res["train_episodes"] and tj["surplus"] == ["ep_10_p5"]
    assert tj["draws_per_seed"] == 4 and set(tj["perturb_frac"]) <= set(tj["train_episodes"])
    shipped = SELECTIONS / "recovery_train_2000.json"
    if shipped.is_file():
        assert set(json.loads(shipped.read_text())) == set(tj)
    assert fj["episodes"] == far and fj["n"] == 5 and fj["seeds_covered"] == 3
    assert fj["frac_max"] == pytest.approx(0.33)
    assert sorted(os.listdir(tmp_path / "train_dir")) == tj["train_episodes"]
    assert sorted(os.listdir(tmp_path / "far_dir")) == far


def test_far_rule_against_shipped_selection():
    """The far-N rule reproduces the shipped far-1000 file whenever the
    shipped train selection carries per-episode fractions; otherwise only
    the membership relation between the two files is checked."""
    train_json = SELECTIONS / "recovery_train_2000.json"
    far_json = SELECTIONS / "recovery_far1000.json"
    if not (train_json.is_file() and far_json.is_file()):
        pytest.skip("shipped recovery selections not present")
    train = json.loads(train_json.read_text())
    far = json.loads(far_json.read_text())
    train_names = set(train["train_episodes"])
    assert set(far["episodes"]) <= train_names
    assert far["n"] == len(far["episodes"]) == len(set(far["episodes"]))
    assert far["episodes"] == sorted(far["episodes"])
    fracs = train.get("frac") or train.get("perturb_frac")
    if not fracs:
        pytest.skip("shipped train selection carries no per-episode fractions; "
                    "the far-N rule is covered on synthetic data")
    rows = [{"name": n, "frac": float(fracs[n])} for n in train_names if n in fracs]
    names, frac_max = assemble.select_far(rows, far["n"])
    assert names == far["episodes"]
    assert frac_max == pytest.approx(far["frac_max"], abs=1e-6)  # stored fractions are rounded


def test_shipped_strict_selection_matches_the_assembler_schema():
    """The shipped strict-500 file carries exactly the keys ``assemble_strict``
    writes, and its ``recipe`` is the strict generation recipe."""
    path = SELECTIONS / "strict_500.json"
    if not path.is_file():
        pytest.skip("shipped strict selection not present")
    shipped = json.loads(path.read_text())
    assert set(shipped) == {"name", "seeds", "per_bin", "bins_cm", "rule", "recipe", "grasp_z_cm",
                            "trimmed_out", "short_bins", "n_recorded_total"}
    recipe = assemble.recipe_summary("strict")
    assert recipe and shipped["recipe"] == recipe
    assert (recipe["tries"], recipe["standoff"], recipe["keep_mode"]) == (20, 0.03, "pose")
    assert "sample_every" not in recipe
    assert len(shipped["seeds"]) == 500 and shipped["per_bin"] == 125
    assert set(shipped["grasp_z_cm"]) == {str(s) for s in shipped["seeds"]}
