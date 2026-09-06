"""Funnel scorer on fixture result files, plus the driver's pure pass
selection / outcome parsing that feed it."""

import json
import math

import pytest

from simretract.eval import driver, score

SEEDS = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]


def _row(seed, outcome, *, drop=None, held=None, t=None, r=None, tw=None, note=None):
    row = {"seed": seed, "ckpt": "fixture", "tilt_deg": 10, "mass_g": 100.0, "occ_z": -0.08,
           "grasp_pose": [0.0] * 7, "queries": 3, "ticks": 140, "outcome": outcome}
    if outcome == "completed":
        verdict = {"strict_pass": False, "left_gripper": not held}
        if note:
            verdict["note"] = note
        if t is not None:
            verdict.update({"grasp_t_mm": t, "grasp_r_deg": r, "grasp_twist_deg": tw,
                            "grasp_swing_deg": 1.0})
        row.update({"closed_at": 139, "drop_mm": drop, "held": held, "strict": False,
                    "verdict": verdict})
    else:
        row.update({"held": False, "strict": False})
    return row


@pytest.fixture
def eval_dir(tmp_path):
    rows = [
        _row(1001, "completed", drop=0.0, held=True, t=4.7, r=2.9, tw=2.2),      # kept
        _row(1002, "completed", drop=1.2, held=True, t=12.0, r=3.0, tw=1.0),     # held, not kept (t)
        _row(1003, "completed", drop=0.5, held=True, t=6.0, r=14.0, tw=13.5),    # held, not kept (r)
        _row(1004, "completed", drop=3.0, held=False, t=8.0, r=4.0, tw=3.0),     # secured, battery pulled it
        _row(1005, "completed", drop=235.6, held=False, note="dropped_on_release"),  # grabbed only
        _row(1006, "timeout"),
        # 1007: no result at all; 1008 completed but NOT in the score set below
        _row(1008, "completed", drop=0.0, held=True, t=1.0, r=1.0, tw=0.5),
        _row(9999, "completed", drop=0.0, held=True, t=1.0, r=1.0, tw=0.5),      # not in the set
    ]
    for row in rows:
        with open(tmp_path / f"result_{row['seed']}.json", "w") as f:
            json.dump(row, f)
    # earlier attempts must be ignored
    with open(tmp_path / "result_1001.try1.json", "w") as f:
        json.dump(_row(1001, "timeout"), f)
    with open(tmp_path / "result_1006.try2.json", "w") as f:
        json.dump(_row(1006, "completed", drop=0.0, held=True, t=1.0, r=1.0, tw=1.0), f)
    sel = {"name": "new_xz_fixture8", "seeds": SEEDS}
    with open(tmp_path / "seeds.json", "w") as f:
        json.dump(sel, f)
    return tmp_path


def test_funnel_counts_and_stats(eval_dir):
    seeds, label = score.load_seeds(eval_dir / "seeds.json")
    assert label == "fixture8"
    f = score.funnel(score.load_results(eval_dir), seeds)
    assert f.n_seeds == 8 and f.n_results == 7
    assert f.missing == [1007]
    assert (f.grabbed, f.secured, f.held, f.kept, f.timeouts) == (6, 5, 4, 2, 1)
    assert f.outcomes == {"completed": 6, "timeout": 1}
    # offset over the five completed rows carrying grasp_t_mm: 4.7 12 6 8 1
    assert f.offset_mm.n == 5
    assert f.offset_mm.median == 6.0
    assert math.isclose(f.offset_mm.mean, (4.7 + 12 + 6 + 8 + 1) / 5)
    assert f.twist_deg.n == 5 and f.twist_deg.median == 2.2


def test_row_line_and_table(eval_dir):
    f, label = score.score_dir(eval_dir, eval_dir / "seeds.json")
    line = score.row_line("fixture_run", label, f, exec_h=50, passes=3)
    assert line.startswith("ROW fixture_run fixture8 3-pass EXEC_H=50: grabbed 6/8 | "
                           "secured 5 | held 4 | kept 2 | med offset 6.0 mm | med twist 2.2 deg")
    md = score.table_md("fixture_run", label, f, exec_h=50)
    assert "| 6/8 | 5 | 4 | 2 | 1 |" in md
    assert "missing [1007]" in md


def test_cli_writes_json_and_md(eval_dir, capsys):
    out_json = eval_dir / "funnel.json"
    out_md = eval_dir / "table.md"
    rc = score.main([str(eval_dir), "--seeds", str(eval_dir / "seeds.json"),
                     "--name", "fx", "--json", str(out_json), "--md", str(out_md)])
    assert rc == 0
    assert capsys.readouterr().out.startswith("ROW fx fixture8: grabbed 6/8")
    with open(out_json) as fh:
        d = json.load(fh)
    assert d["funnel"]["kept"] == 2 and d["funnel"]["secured"] == 5
    assert out_md.read_text().startswith("# fx")


def test_empty_dir_gives_nan_stats(tmp_path):
    f = score.funnel({}, [1, 2])
    assert f.grabbed == 0 and f.missing == [1, 2]
    assert math.isnan(f.offset_mm.median)
    assert "nan" in score.row_line("x", "y", f)
    assert score.Stats.of([3.0]).sd != score.Stats.of([3.0]).sd  # NaN for a single value


def test_pass_selection_renames_non_completed_only(eval_dir):
    run = [1001, 1005, 1006, 1007]
    # pass 1: only the seed without a result
    assert driver.select_seeds(eval_dir, run, 1) == [1007]
    # pass 2: non-completed (timeout) + missing; completed rows untouched
    assert driver.select_seeds(eval_dir, run, 2) == [1006, 1007]
    assert (eval_dir / "result_1006.try1.json").is_file()
    assert not (eval_dir / "result_1006.json").is_file()
    assert (eval_dir / "result_1001.json").is_file()
    assert (eval_dir / "result_1005.json").is_file()   # dropped_on_release is completed
    # pass 3 after a still-missing pass 2: nothing to rename, missing seeds re-selected
    assert driver.select_seeds(eval_dir, run, 3) == [1006, 1007]


def test_outcome_line_parsing(tmp_path):
    log = tmp_path / "worker.log"
    log.write_text("noise\n[M5] server ok: {'ok': True}\n"
                   '[M5] {"seed": 7, "outcome": "timeout", "held": false}\n'
                   "[M5] DONE\n")
    rows = driver.parse_outcomes(log)
    assert rows == [{"seed": 7, "outcome": "timeout", "held": False}]


def test_terminal_outcomes_stop_retries():
    attempts = {"5": [{"pass": 1, "outcome": "timeout"}, {"pass": 2, "outcome": None}],
                "6": [{"pass": 1, "outcome": "scene_reject"}]}
    assert driver.terminal_outcome(attempts, 5) is None
    assert driver.terminal_outcome(attempts, 6) == "scene_reject"
    assert driver.terminal_outcome(attempts, 7) is None


def test_regime_check_refuses_mismatch(tmp_path):
    ck = tmp_path / "ck"
    ck.mkdir()
    cfg = driver.DriverConfig(ckpt=ck, out=tmp_path, seeds_json=tmp_path, score_json=tmp_path,
                              name="x", policy_type="pi0")
    good = {"ok": True, "regime": {"ckpt": str(ck), "type": "pi0", "half": True,
                                   "plane_filter": True, "drop_px": 300, "plane_top_cut": 0.0,
                                   "eval_aug": False, "chunk": 50, "horizon": 50}}
    assert driver.check_regime(good, cfg) == []
    bad = json.loads(json.dumps(good))
    bad["regime"].update({"plane_top_cut": 0.1, "half": False})
    problems = driver.check_regime(bad, cfg)
    assert any("plane_top_cut" in p for p in problems) and any("half" in p for p in problems)
    assert driver.check_regime({"ok": True}, cfg)
    act = driver.DriverConfig(ckpt=ck, out=tmp_path, seeds_json=tmp_path, score_json=tmp_path,
                              name="x", policy_type="act")
    act_pong = json.loads(json.dumps(good))
    act_pong["regime"].update({"type": "act", "half": False})
    assert driver.check_regime(act_pong, act) == []
