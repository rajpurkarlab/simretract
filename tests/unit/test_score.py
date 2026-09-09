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
    # earlier attempts differ from the current results
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
    f = score.funnel(score.load_results(eval_dir, attempt="latest"), seeds)
    assert f.n_seeds == 8 and f.n_results == 7
    assert f.missing == [1007]
    assert (f.grabbed, f.secured, f.held, f.kept, f.timeouts) == (6, 5, 4, 2, 1)
    assert f.outcomes == {"completed": 6, "timeout": 1}
    # offset over the five secured rows carrying grasp_t_mm: 4.7 12 6 8 1
    assert f.offset_mm.n == 5
    assert f.offset_mm.median == 6.0
    assert math.isclose(f.offset_mm.mean, (4.7 + 12 + 6 + 8 + 1) / 5)
    assert f.twist_deg.n == 5 and f.twist_deg.median == 2.2


def test_row_line_and_table(eval_dir):
    f, label = score.score_dir(eval_dir, eval_dir / "seeds.json", attempt="latest")
    line = score.row_line("fixture_run", label, f, exec_h=50, passes=3, attempt="latest")
    assert line.startswith("ROW fixture_run fixture8 3-pass EXEC_H=50: grabbed 6/8 | "
                           "secured 5 | held 4 | kept 2 | med offset 6.0 mm | med twist 2.2 deg")
    assert "Grasped 62.5% | attempt=latest" in line
    md = score.table_md("fixture_run", label, f, exec_h=50, attempt="latest")
    assert "| Grasped (%) |" in md and "| 62.5 |" in md
    assert "grabbed 6/8; secured 5; held 4; kept 2; timeouts 1" in md
    assert "Attempt: latest; SD: population" in md
    assert "missing [1007]" in md


def test_cli_writes_json_and_md(eval_dir, capsys):
    out_json = eval_dir / "funnel.json"
    out_md = eval_dir / "table.md"
    rc = score.main([str(eval_dir), "--seeds", str(eval_dir / "seeds.json"),
                     "--name", "fx", "--attempt", "latest",
                     "--json", str(out_json), "--md", str(out_md)])
    assert rc == 0
    assert capsys.readouterr().out.startswith("ROW fx fixture8: grabbed 6/8")
    with open(out_json) as fh:
        d = json.load(fh)
    assert d["funnel"]["kept"] == 2 and d["funnel"]["secured"] == 5
    assert d["attempt"] == "latest" and d["sd"] == "population"
    assert d["grasped_pct"] == 62.5
    assert out_md.read_text().startswith("# fx")


def test_empty_dir_gives_nan_stats(tmp_path):
    f = score.funnel({}, [1, 2])
    assert f.grabbed == 0 and f.missing == [1, 2]
    assert math.isnan(f.offset_mm.median)
    assert "nan" in score.row_line("x", "y", f)
    assert score.Stats.of([3.0]).sd == 0.0
    assert math.isnan(score.Stats.of([3.0], sample_sd=True).sd)


def test_first_attempt_selects_earliest_archive_or_current(eval_dir):
    first = score.load_results(eval_dir)
    latest = score.load_results(eval_dir, attempt="latest")
    assert first[1001]["outcome"] == "timeout"
    assert latest[1001]["outcome"] == "completed"
    # Missing try1: the earliest available numbered archive still wins.
    assert first[1006]["outcome"] == "completed"
    assert latest[1006]["outcome"] == "timeout"
    assert first[1002] == latest[1002]
    # Select numerically, not lexically (try2 precedes try10).
    (eval_dir / "result_1006.try10.json").write_text(json.dumps(_row(1006, "timeout")))
    assert score.load_results(eval_dir)[1006]["outcome"] == "completed"


@pytest.mark.parametrize("bad", ["{broken", "null", '{"seed": 9999, "outcome": "completed"}'])
def test_corrupt_first_attempt_never_uses_later_success(tmp_path, bad):
    (tmp_path / "result_7.try1.json").write_text(bad)
    success = _row(7, "completed", drop=0., held=True, t=1., r=1., tw=1.)
    (tmp_path / "result_7.try2.json").write_text(json.dumps(success))
    (tmp_path / "result_7.json").write_text(json.dumps(success))
    f = score.funnel(score.load_results(tmp_path), [7])
    assert f.missing == [7] and f.secured == 0
    assert score.load_results(tmp_path, attempt="latest")[7] == success


def test_archived_result_without_current_is_scored_first(tmp_path):
    (tmp_path / "result_7.try2.json").write_text(json.dumps(_row(7, "timeout")))
    assert score.load_results(tmp_path)[7]["outcome"] == "timeout"
    assert score.load_results(tmp_path, attempt="latest") == {}
    with pytest.raises(ValueError, match="attempt"):
        score.load_results(tmp_path, attempt="best")


def test_paper_metrics_use_secured_subset_and_population_sd():
    rows = [
        _row(1, "completed", drop=0., held=True, t=2., r=4., tw=6.),
        _row(2, "completed", drop=19.9, held=False, t=6., r=8., tw=10.),
        # Having verdict metrics must not include a released tool in the stats.
        _row(3, "completed", drop=20., held=False, t=100., r=100., tw=100.),
        _row(4, "completed", drop=250., held=False, t=200., r=200., tw=200.),
        _row(5, "timeout"),
    ]
    results = {r["seed"]: r for r in rows}
    f = score.funnel(results, range(1, 7))
    assert (f.grabbed, f.secured, f.held, f.kept) == (4, 2, 1, 1)
    assert f.missing == [6] and math.isclose(f.grasped_pct, 100. / 3.)
    for metric, mean in ((f.offset_mm, 4.), (f.rot_deg, 6.), (f.twist_deg, 8.)):
        assert (metric.n, metric.mean, metric.median, metric.sd) == (2, mean, mean, 2.)
    sample = score.funnel(results, range(1, 7), sample_sd=True)
    assert math.isclose(sample.offset_mm.sd, math.sqrt(8.))
    assert sample.secured == f.secured


def test_score_cli_defaults_to_first_and_records_sample_option(eval_dir, capsys):
    output = eval_dir / "first.json"
    md = eval_dir / "first.md"
    score.main([str(eval_dir), "--seeds", str(eval_dir / "seeds.json"),
                "--json", str(output), "--md", str(md), "--sample-sd"])
    data = json.loads(output.read_text())
    first, _ = score.score_dir(eval_dir, eval_dir / "seeds.json", sample_sd=True)
    assert data["attempt"] == "first" and data["sd"] == "sample"
    assert data["funnel"] == first.to_dict()
    assert "attempt=first" in capsys.readouterr().out
    assert "Attempt: first; SD: sample" in md.read_text()


def test_pass_selection_renames_non_completed_only(eval_dir):
    run = [1001, 1005, 1006, 1007]
    # pass 1: only the seed without a result
    assert driver.select_seeds(eval_dir, run, 1) == [1007]
    # pass 2: non-completed (timeout) + missing; completed rows untouched
    assert driver.select_seeds(eval_dir, run, 2) == [1006, 1007]
    assert (eval_dir / "result_1006.try3.json").is_file()
    assert json.loads((eval_dir / "result_1006.try2.json").read_text())["outcome"] == "completed"
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
