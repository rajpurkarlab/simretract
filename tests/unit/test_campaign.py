"""Campaign driver mechanics without the simulator: chunking, outcome
parsing, ledger persistence, top-up planning, seed pre-screen and the
worker pool (fake workers run in the current interpreter)."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap

import numpy as np

from simretract.sim import campaign
from simretract.sim.recorder import DONE_LINE, OUTCOME_TAG, RecorderConfig, episode_done
from simretract.sim.recovery_recorder import RECOVERY_DEFAULTS, virtual_to_episode

TIP = [-0.00677, 0.0012, 0.12624]


def test_chunking_and_outcome_parsing():
    assert campaign.chunked([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert campaign.chunked([], 3) == []
    text = "\n".join([
        "boot noise", f"{OUTCOME_TAG} stored winner matched (d=1.2mm)",
        f'{OUTCOME_TAG} {json.dumps({"seed": 7, "outcome": "recorded", "frames": 300})}',
        f"{OUTCOME_TAG} not json {{",
        f'  {OUTCOME_TAG} {json.dumps({"seed": 8, "draw": 1, "virt": 33, "outcome": "no_grasp"})}  ',
        DONE_LINE])
    rows = campaign.parse_outcomes(text)
    assert [r["seed"] for r in rows] == [7, 8]
    assert rows[1]["virt"] == 33


def test_ledger_roundtrip(tmp_path):
    led = campaign.Ledger(tmp_path / "ledger.json")
    led.launched(["1", "2"])
    led.record("1", "recorded")
    led.save()
    led2 = campaign.Ledger(tmp_path / "ledger.json")
    assert led2.attempts("1") == 1 and led2.attempts("2") == 1 and led2.attempts("3") == 0
    assert led2.item("1")["outcome"] == "recorded" and led2.data["chunks"] == 1


def test_virtual_index_mapping_and_keys(tmp_path):
    seeds = [300, 100, 200]
    assert virtual_to_episode(0, sorted(seeds), 4) == (100, 0)
    assert virtual_to_episode(7, sorted(seeds), 4) == (200, 3)
    job = campaign.recovery_job(str(tmp_path), [5, 6], ["--blend"], "s.json", "w", 4,
                                sorted(seeds), "t")
    assert job.key_of_row({"virt": 6}) == "6"
    assert job.key_of_row({"seed": 200, "draw": 3}) == "7"
    assert job.key_of_row({"seed": 999, "draw": 0}) is None
    assert campaign.resolve_cmd(job)[-2:][0] == "--virt" or "5,6" in campaign.resolve_cmd(job)
    (tmp_path / "ep_200_p3.filtered").touch()
    assert job.is_done("7") and not job.is_done("6")


def test_plan_topup(tmp_path):
    out = tmp_path / "rec"
    out.mkdir()
    seeds = [10, 20, 30]
    # seed 10: 4 keeps (complete); seed 20: 1 keep + 2 tried; seed 30: nothing
    for d in range(4):
        (out / f"ep_10_p{d}").mkdir()
        (out / f"ep_10_p{d}" / "meta.npz").write_bytes(b"x")
    (out / "ep_20_p0").mkdir()
    (out / "ep_20_p0" / "meta.npz").write_bytes(b"x")
    (out / "ep_20_p1.filtered").touch()
    (out / "ep_20_p2.wedged").touch()
    plan, stats = campaign.plan_topup(str(out), seeds, draws=8, target_keeps=4)
    # seed 20 needs 3 -> round(3/0.3)=10 capped by the 5 untried draws p3..p7
    # seed 30 needs 4 -> 13 capped by 8 draws
    assert plan == [1 * 8 + p for p in range(3, 8)] + [2 * 8 + p for p in range(8)]
    assert stats == {"deficit_seeds": 2, "missing_keeps": 7, "draws_planned": 13,
                     "surplus_seeds": 0}
    plan2, _ = campaign.plan_topup(str(out), [10], draws=8, target_keeps=4)
    assert plan2 == []


def test_prescreen_seed():
    tilt = [0, 0, 0, 0]
    ok, why = campaign.prescreen_seed(200005, TIP, "train", -0.05, -0.025, tilt, 500)
    assert (ok, why) == (True, "ok")
    # a bin far below the hand can never be blocked; an absurd bin inside the
    # hand is
    assert campaign.prescreen_seed(200005, TIP, "train", -0.09, -0.06, tilt, 500)[1] == "hand_blocks_bin"
    # tilt cap: seed 200005 tilts 67 deg -> bin 3; fill that bin past the cap
    full = [0, 0, 0, 200]
    assert campaign.prescreen_seed(200005, TIP, "train", -0.05, -0.025, full, 500)[1] == "tilt_cap"
    assert campaign.tilt_bin(0.0) == 0 and campaign.tilt_bin(59.9) == 2 and campaign.tilt_bin(95) == 3


def test_recorder_config_cli_roundtrip(monkeypatch):
    monkeypatch.setenv("SIMRETRACT_STANDOFF", "0.02")
    monkeypatch.setenv("ERA4_HAND", "0")
    monkeypatch.setenv("E4_PICK", "0")
    monkeypatch.setenv("SKIP_BATTERY", "0")
    ap = argparse.ArgumentParser()
    RecorderConfig.add_args(ap, defaults={"tries": 10})
    cfg = RecorderConfig.from_args(ap.parse_args(["--speed-scale", "1.0", "--no-blend"]))
    assert cfg.standoff == 0.02 and cfg.hand_style == "test" and cfg.picker == "clearance"
    assert cfg.skip_battery is False and cfg.tries == 10
    assert cfg.speed_scale == 1.0 and cfg.blend is False
    monkeypatch.setenv("TRIES", "7")
    ap2 = argparse.ArgumentParser()
    RecorderConfig.add_args(ap2, defaults={"tries": 10})
    assert RecorderConfig.from_args(ap2.parse_args([])).tries == 7
    args = cfg.to_cli_args()
    ap3 = argparse.ArgumentParser()
    monkeypatch.delenv("SIMRETRACT_STANDOFF")
    RecorderConfig.add_args(ap3)
    assert RecorderConfig.from_args(ap3.parse_args(args)) == cfg
    prod = RecorderConfig()
    assert (prod.blend, prod.keep_mode, prod.picker, prod.hand_style) == (True, "pose", "height", "train")
    assert (prod.standoff, prod.ctrl_back, prod.decel_arc, prod.entry_speed) == (0.03, 0.06, 0.0, 1.0)
    assert (prod.joint_rate_deg, prod.speed_scale, prod.tries, prod.mu) == (2.5, 0.9, 20, 0.4)
    assert (prod.skip_battery, prod.interfere_gate, prod.z_lo, prod.z_hi) == (True, True, -0.05, 0.05)


def test_recipe_search_tries(monkeypatch):
    """The strict search tries 20 candidates; the recovery fallback search
    keeps the recorder default of 10 (the recipe file must not raise it)."""
    for name in ("TRIES", "SIMRETRACT_TRIES"):
        monkeypatch.delenv(name, raising=False)
    assert RecorderConfig.recipe_defaults("strict")["tries"] == 20
    assert RecorderConfig.recipe_defaults("recovery")["tries"] == 10
    assert RECOVERY_DEFAULTS["tries"] == 10


def test_generate_recovery_dry_run_uses_recipe_defaults(tmp_path, monkeypatch, capsys):
    """`simretract-generate recovery --dry-run` plans worker commands with
    --search-tries 10 and chunks of the recipe's campaign block."""
    from simretract.cli import generate

    for name in ("TRIES", "SIMRETRACT_TRIES"):
        monkeypatch.delenv(name, raising=False)
    sel = tmp_path / "strict.json"
    sel.write_text(json.dumps({"name": "s", "seeds": [400851, 200350]}))
    rc = generate.main(["recovery", "--out", str(tmp_path / "rec"), "--strict-json", str(sel),
                        "--winners", str(tmp_path), "--draws", "4", "--range", "0:8", "--dry-run"])
    assert rc == 0
    cmds = [line for line in capsys.readouterr().out.splitlines() if "simretract.sim.recovery_recorder" in line]
    assert len(cmds) == 1, cmds                       # recovery.yaml campaign: chunk 8
    argv = cmds[0].split()
    assert argv[argv.index("--search-tries") + 1] == "10"
    assert argv[argv.index("--virt") + 1] == ",".join(str(v) for v in range(8))
    ap = argparse.ArgumentParser()
    campaign.CampaignConfig.add_args(ap, defaults={"chunk": 8, "timeout_s": 2400})
    a = ap.parse_args([])
    assert (a.chunk, a.timeout, a.parallel, a.tries) == (8, 2400.0, 3, 2)


FAKE_WORKER = textwrap.dedent(f"""
    import json, os, sys, time
    items = [int(s) for s in sys.argv[1].split(",") if s]
    out = sys.argv[2]
    mode = sys.argv[3]
    for k in items:
        ep = os.path.join(out, f"ep_{{k}}")
        if mode == "hang":
            print("{OUTCOME_TAG} " + json.dumps({{"seed": k, "outcome": "started"}}), flush=True)
            time.sleep(60)
        if k % 3 == 0:            # deterministic reject -> .filtered marker
            open(ep + ".filtered", "w").close()
            print("{OUTCOME_TAG} " + json.dumps({{"seed": k, "outcome": "record_divergence:pose_delta"}}), flush=True)
        elif k % 3 == 1 or mode == "flaky_pass":
            os.makedirs(ep, exist_ok=True)
            open(os.path.join(ep, "meta.npz"), "wb").close()
            print("{OUTCOME_TAG} " + json.dumps({{"seed": k, "outcome": "recorded"}}), flush=True)
        else:                     # no marker: the driver must retry, then give up
            print("{OUTCOME_TAG} " + json.dumps({{"seed": k, "outcome": "no_grasp"}}), flush=True)
    print("{DONE_LINE}", flush=True)
    if mode == "linger":
        time.sleep(60)
""")


def fake_job(out: str, keys: list[int], mode: str, tag: str) -> campaign.Job:
    job = campaign.strict_job(out, keys, [], None, tag)
    job.cmd = ["-c", FAKE_WORKER, "{items}", out, mode]
    return job


def _cfg(tmp_path, **kw) -> campaign.CampaignConfig:
    base = dict(out=str(tmp_path), parallel=2, stagger_s=0.0, chunk=2, timeout_s=30.0, tries=2,
                poll_s=0.1, done_grace_s=0.3, python=sys.executable)
    base.update(kw)
    return campaign.CampaignConfig(**base)


def test_run_jobs_retries_until_marker_or_tries(tmp_path):
    cfg = _cfg(tmp_path)
    ledger = campaign.Ledger(tmp_path / "ledger.json")
    keys = [1, 2, 3, 4, 5, 6]
    jobs = [fake_job(str(tmp_path), ch, "plain", f"j{ch[0]}") for ch in campaign.chunked(keys, 2)]
    summary = campaign.run_jobs(cfg, ledger, jobs)
    assert episode_done(str(tmp_path / "ep_1")) and episode_done(str(tmp_path / "ep_3"))
    assert not episode_done(str(tmp_path / "ep_2")) and not episode_done(str(tmp_path / "ep_5"))
    led = campaign.Ledger(tmp_path / "ledger.json")
    assert led.attempts("1") == 1 and led.attempts("3") == 1
    assert led.attempts("2") == 2 and led.attempts("5") == 2       # retried once, then given up
    assert led.item("2")["outcome"] == "no_grasp" and led.item("3")["outcome"].startswith("record_divergence")
    assert summary["chunks_launched"] == 3 + 2                     # 3 chunks + 2 retry chunks
    logs = sorted(p.name for p in (tmp_path / ".logs").iterdir())
    assert "j1.log" in logs and any(n.startswith("j1_r") for n in logs)


def test_run_jobs_reaps_lingering_and_times_out(tmp_path):
    cfg = _cfg(tmp_path, timeout_s=1.5, tries=1, parallel=2)
    ledger = campaign.Ledger(tmp_path / "ledger.json")
    reasons = []
    jobs = [fake_job(str(tmp_path), [1], "linger", "linger"),
            fake_job(str(tmp_path), [7], "hang", "hang")]
    campaign.run_jobs(cfg, ledger, jobs, on_finished=lambda w, r: reasons.append((w.job.tag, r)))
    assert dict(reasons) == {"linger": "reaped", "hang": "timeout"}
    assert episode_done(str(tmp_path / "ep_1"))
    assert not episode_done(str(tmp_path / "ep_7")) and ledger.attempts("7") == 1
    assert ledger.item("7")["outcome"] == "started"


def test_run_strict_list_uses_done_markers(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, tries=1)
    (tmp_path / "ep_4").mkdir()
    (tmp_path / "ep_4" / "meta.npz").write_bytes(b"")
    calls = []
    real_strict_job = campaign.strict_job

    def patched(out, keys, recipe_args, stored_winners, tag):
        calls.append(list(keys))
        job = real_strict_job(out, keys, recipe_args, stored_winners, tag)
        job.cmd = ["-c", FAKE_WORKER, "{items}", out, "plain"]
        return job
    monkeypatch.setattr(campaign, "strict_job", patched)
    summary = campaign.run_strict_list(cfg, RecorderConfig(), [1, 4, 7], None)
    assert calls == [[1, 7]]                # seed 4 already done -> not launched
    assert summary["recorded"] == 3 and summary["chunks_launched"] == 1


def test_bin_counts_from_synthetic_raw(tmp_path):
    from simretract.sim.sampling import presented_pose_for_seed

    for seed, z, tilt in ((11, -0.04, 10.0), (12, 0.01, 50.0), (13, 0.2, 70.0)):
        pos, rot = presented_pose_for_seed(seed)
        d = tmp_path / f"ep_{seed}"
        d.mkdir()
        np.savez(d / "meta.npz", states=np.zeros((5, 7)), seed=seed, tilt_deg=tilt,
                 info=json.dumps({"grasp_w": (pos + rot.apply([0, 0, z])).tolist(),
                                  "R_grasp": np.eye(3).tolist(), "pose_t_mm": 1, "pose_r_deg": 1}))
    per_bin, tilt = campaign.bin_counts(str(tmp_path), -0.05, 0.05, 4)
    assert per_bin == [1, 0, 1, 0] and tilt == [1, 0, 1, 0]
