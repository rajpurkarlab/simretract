"""Real-deployment CLI contracts: client defaults, presets, no robot address defaults,
launcher refusing the CHANGE_ME placeholder, and the pure control-loop helpers."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from simretract import real as real_pkg
from simretract.real import (FOLLOWER_IP_PLACEHOLDER, follower_ip_is_set,
                             load_real_config, preset_client_args, resolve_preset)
from simretract.real import client as rc
from simretract.real import home, live_view, mask_server, tally
from simretract.cli import run_real

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "real" / "default.yaml"


# ------------------------------------------------------------------ client flags
def test_client_defaults_are_the_deployment_regime():
    d = rc.build_parser().parse_args([])
    assert d.plane_top_cut == pytest.approx(0.10)
    assert d.follower_ip is None
    assert (d.exec_h, d.exec_min) == (50, 50)
    assert d.speed == pytest.approx(0.8)
    assert d.max_step == pytest.approx(0.06)
    assert d.smooth == pytest.approx(0.6)
    assert d.settle == 0.0
    assert d.view is False
    assert (d.port, d.mask_port) == (8765, 8766)
    assert d.min_tool_px == 300 and d.hold_max == 2.0 and d.close_cmd == pytest.approx(0.020)
    assert d.max_ticks == 451 and d.home_speed == pytest.approx(0.6)
    assert d.go is False


def test_client_view_toggle():
    p = rc.build_parser()
    assert p.parse_args(["--view"]).view is True
    assert p.parse_args(["--view", "--no-view"]).view is False


def test_client_env_fallback(monkeypatch):
    monkeypatch.setenv("SIMRETRACT_PLANE_TOP_CUT", "0.15")
    assert rc.env_default(float, 0.10, "SIMRETRACT_PLANE_TOP_CUT", "PLANE_TOP_CUT") == pytest.approx(0.15)
    monkeypatch.delenv("SIMRETRACT_PLANE_TOP_CUT")
    assert rc.env_default(float, 0.10, "SIMRETRACT_PLANE_TOP_CUT") == pytest.approx(0.10)


def test_client_refuses_placeholder_ip(tmp_path):
    cfg = tmp_path / "real.yaml"
    cfg.write_text("robot:\n  follower_ip: CHANGE_ME\n")
    a = rc.build_parser().parse_args(["--config", str(cfg)])
    with pytest.raises(SystemExit, match=FOLLOWER_IP_PLACEHOLDER):
        rc.resolve_args(a)


# ------------------------------------------------------------------ no IP defaults anywhere
def test_no_robot_address_defaults():
    assert rc.build_parser().get_default("follower_ip") is None
    assert home.build_parser().get_default("follower_ip") is None
    assert live_view.build_parser().get_default("follower_ip") is None
    assert run_real.build_parser().get_default("follower_ip") is None
    ipv4 = re.compile(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b")
    files = list((REPO / "simretract" / "real").glob("*.py")) + [REPO / "simretract" / "cli" / "run_real.py",
                                                                 CONFIG]
    for f in files:
        for m in ipv4.finditer(f.read_text()):
            assert m.group(0) == "127.0.0.1", f"robot address literal {m.group(0)} in {f}"


def test_follower_ip_placeholder_detection():
    assert not follower_ip_is_set(None)
    assert not follower_ip_is_set("")
    assert not follower_ip_is_set("CHANGE_ME")
    assert not follower_ip_is_set(" change_me ")
    assert follower_ip_is_set("10.0.0.7")


# ------------------------------------------------------------------ presets
def test_shipped_config_has_placeholder_ip_and_ports():
    cfg = load_real_config(CONFIG)
    assert cfg["robot"]["follower_ip"] == FOLLOWER_IP_PLACEHOLDER
    assert cfg["services"]["policy_port"] == 8765
    assert cfg["services"]["mask_port"] == 8766
    assert cfg["mask"]["prompt"] == "metal rake"
    assert cfg["mask"]["conf"] == pytest.approx(0.2)
    assert Path(cfg["mask"]["sam3_ckpt"]).is_absolute()


@pytest.mark.parametrize("name,expect", [
    ("default", dict(exec_h=50, exec_min=50, settle=0.0, speed=0.8, max_step=0.06, smooth=0.6,
                     view=True, plane_top_cut=0.10)),
    ("parity", dict(exec_h=20, exec_min=20, settle=0.0, speed=1.0, max_step=0.06, smooth=0.45,
                    view=False, plane_top_cut=0.0)),
    ("conservative", dict(exec_h=10, exec_min=5, settle=0.25, speed=0.5, max_step=0.03, smooth=0.45,
                          view=False, plane_top_cut=0.10)),
])
def test_preset_resolution(name, expect):
    cfg = load_real_config(CONFIG)
    p = resolve_preset(cfg, name)
    for k, v in expect.items():
        assert p[k] == pytest.approx(v) if isinstance(v, float) else p[k] == v, (name, k)
    # the preset round-trips through the client parser
    ns = rc.build_parser().parse_args(preset_client_args(p))
    for k in real_pkg.PRESET_KEYS:
        assert getattr(ns, k) == pytest.approx(p[k]) if isinstance(p[k], float) else getattr(ns, k) == p[k]


def test_unknown_preset():
    with pytest.raises(KeyError, match="unknown real-run preset"):
        resolve_preset(load_real_config(CONFIG), "turbo")


# ------------------------------------------------------------------ launcher
def test_run_real_refuses_change_me(tmp_path):
    with pytest.raises(SystemExit, match=FOLLOWER_IP_PLACEHOLDER):
        run_real.resolve_follower_ip(None, {"robot": {"follower_ip": "CHANGE_ME"}})
    assert run_real.resolve_follower_ip("10.1.2.3", {"robot": {"follower_ip": "CHANGE_ME"}}) == "10.1.2.3"
    # end to end on the shipped config: refuses before touching checkpoints or servers
    ck = tmp_path / "ckpt"
    ck.mkdir()
    with pytest.raises(SystemExit, match=FOLLOWER_IP_PLACEHOLDER):
        run_real.main(["--ckpt", str(ck), "servers-only", "--config", str(CONFIG)])


def test_run_real_modes_and_defaults():
    p = run_real.build_parser()
    a = p.parse_args(["--ckpt", "x"])
    assert a.mode == "servers-only" and a.preset == "default" and a.keep_server is True
    assert a.plane_top_cut is None  # preset value (0.10 for default) unless overridden
    a = p.parse_args(["--ckpt", "x", "go", "--", "--yes", "--no-view"])
    assert a.mode == "go" and "--yes" in a.client_args and "--no-view" in a.client_args
    with pytest.raises(SystemExit):
        p.parse_args(["--ckpt", "x", "fly"])


def test_exec_h_per_backbone():
    assert run_real.exec_h_for({"type": "pi0", "chunk_h": 50}) is None
    assert run_real.exec_h_for({"type": "smolvla", "chunk_h": 50}) is None
    assert run_real.exec_h_for({"type": "act", "chunk_h": 50}) is None
    assert run_real.exec_h_for({"type": "fastwam", "chunk_h": 48}) == 48
    assert run_real.exec_h_for({"type": "fastwam", "chunk_h": 32}) == 10
    assert run_real.exec_h_for({"type": "fastwam", "chunk_h": 32}, fastwam_exec_h=16) == 16


def test_policy_regime_check(tmp_path):
    ck = str(tmp_path / "ckpt")
    assert run_real.policy_regime_ok(None, ck) == (False, "no server")
    assert run_real.policy_regime_ok({"ok": True, "ckpt": ck, "eval_aug": "0"}, ck)[0]
    assert not run_real.policy_regime_ok({"ok": True, "ckpt": ck, "eval_aug": "1"}, ck)[0]
    assert not run_real.policy_regime_ok({"ok": True, "ckpt": ck + "_other"}, ck)[0]
    assert not run_real.policy_regime_ok({"ok": True, "ckpt": ck, "plane_top_cut": 0.15}, ck)[0]
    assert not run_real.policy_regime_ok({"ok": True, "ckpt": ck, "plane_filter": 0}, ck)[0]
    assert run_real.policy_regime_ok({"ok": True, "ckpt": ck, "plane_filter": 1, "plane_top_cut": 0.0}, ck)[0]
    # the shipped server nests its regime under "regime"
    nested = {"ok": True, "ckpt": ck, "eval_aug": "0", "regime": {"plane_filter": True, "plane_top_cut": 0.15}}
    assert not run_real.policy_regime_ok(nested, ck)[0]
    nested["regime"]["plane_top_cut"] = 0.0
    assert run_real.policy_regime_ok(nested, ck)[0]


def test_read_ckpt_meta(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"type": "fastwam"}))
    (tmp_path / "norm.json").write_text(json.dumps({"input_mode": "masked_rgbd", "state_mode": "qpos_plane",
                                                    "chunk_h": 48}))
    m = run_real.read_ckpt_meta(str(tmp_path))
    assert m["type"] == "fastwam" and m["chunk_h"] == 48 and m["state_mode"] == "qpos_plane"
    assert "fastwam" in run_real.describe_ckpt(m)
    with pytest.raises(SystemExit):
        run_real.read_ckpt_meta(str(tmp_path / "missing"))
    # the trainer's <run>/final layout resolves to the inner directory
    run = tmp_path / "run"
    (run / "final").mkdir(parents=True)
    for f in ("config.json", "norm.json"):
        (run / "final" / f).write_text((tmp_path / f).read_text())
    assert run_real.resolve_ckpt_dir(str(run)) == str((run / "final").resolve())
    assert run_real.resolve_ckpt_dir(str(tmp_path)) == str(tmp_path.resolve())
    with pytest.raises(SystemExit):
        run_real.resolve_ckpt_dir(str(tmp_path / "nope"))


# ------------------------------------------------------------------ mask server flags
def test_mask_server_flags(monkeypatch):
    monkeypatch.delenv("SAM3_CKPT", raising=False)
    monkeypatch.delenv("SIMRETRACT_SAM3_CKPT", raising=False)
    p = mask_server.build_parser()
    with pytest.raises(SystemExit):  # --ckpt is required, there is no default path
        p.parse_args([])
    a = p.parse_args(["--ckpt", "sam3.pt"])
    assert a.prompt == "metal rake" and a.port == 8766 and a.conf == pytest.approx(0.2)
    assert a.host == "127.0.0.1"


# ------------------------------------------------------------------ pure loop helpers
def test_d405_crop_math():
    y0, x0, ch, cw = rc.compute_crop(390.86, 319.27, 233.78)
    assert (y0, x0, ch, cw) == (17, 31, 433, 577)
    assert abs(390.86 * 480 / ch - 433.0) < 1.0


def test_adaptive_horizon():
    assert rc.exec_horizon(20, 5, 0.22, 0.10, 0.22) == 20
    assert rc.exec_horizon(20, 5, 0.10, 0.10, 0.22) == 5
    assert rc.exec_horizon(20, 5, 0.16, 0.10, 0.22) == 12  # 12.5 rounds half to even, as in the reference control loop
    assert rc.exec_horizon(20, 5, 0.18, 0.10, 0.22) == 15
    assert rc.exec_horizon(20, 5, None, 0.10, 0.22) == 20
    assert rc.exec_horizon(50, 50, 0.05, 0.10, 0.22) == 50


def test_safety_clamp():
    rng = np.random.default_rng(0)
    cur = np.zeros(7)
    cur[6] = 0.02
    for _ in range(200):
        act = rng.uniform(-2, 2, 7).astype(np.float32)
        _, sent = rc.clamp_target(act, cur, 0.06)
        assert np.all(np.abs(sent[:6] - cur[:6]) <= 0.06 + 1e-9)
        assert abs(sent[6] - cur[6]) <= 0.005 + 1e-9
        assert 0.0 <= sent[6] <= 0.044


def test_close_detector():
    d = rc.CloseDetector(0.020)
    fired = [d.update(0.01, 0.030) for _ in range(13)]
    assert fired.index(True) == 12  # first tick with prev_open set is tick 1 -> 12 stalled ticks
    d = rc.CloseDetector(0.020)
    for _ in range(11):
        assert not d.update(0.01, 0.030)
    assert not d.update(0.01, 0.032)  # jitter above STALL_EPS resets
    d = rc.CloseDetector(0.020)
    assert not any(d.update(0.01, 0.040) for _ in range(30))  # open range never closes
    d = rc.CloseDetector(0.020)
    assert not any(d.update(0.030, 0.030) for _ in range(30))  # not commanded to close


def test_plane_top_cut_on_a_bar():
    mask = np.zeros((480, 640), np.uint8)
    mask[100:400, 300:340] = 255  # vertical bar, 300 px long
    kept, rem = rc.plane_top_cut(mask, 0.15)
    assert rem is not None
    rows = np.nonzero(rem.any(1))[0]
    assert rows.min() == 100 and 35 <= len(rows) <= 55  # about 15% of the length, at the top
    assert (kept > 0).sum() + (rem > 0).sum() == (mask > 0).sum()
    small = np.zeros((480, 640), np.uint8)
    small[10:20, 10:20] = 255
    assert rc.plane_top_cut(small, 0.15)[1] is None
    assert rc.plane_top_cut(mask, 0.0)[1] is None


def test_fit_plane_local_recovers_a_flat_surface():
    fx, cx, cy = rc.PLANE_K[0, 0], rc.PLANE_K[0, 2], rc.PLANE_K[1, 2]
    v, u = np.mgrid[0:480, 0:640]
    n_true = np.array([0.2, -0.1, -1.0])
    n_true /= np.linalg.norm(n_true)
    c0 = np.array([0.0, 0.0, 0.25])
    # z such that n . (P - c0) = 0 with P = ((u-cx)/fx z, (v-cy)/fx z, z)
    denom = n_true[0] * (u - cx) / fx + n_true[1] * (v - cy) / fx + n_true[2]
    z = (n_true @ c0) / denom
    dep_mm = (z * 1000).astype(np.uint16)
    mask = np.zeros((480, 640), np.uint8)
    mask[150:330, 220:420] = 255
    pl = rc.fit_plane_local(dep_mm, mask)
    assert pl is not None and pl["n"][2] < 0
    assert abs(float(pl["n"] @ n_true)) > 0.999


def test_tally_fixture(tmp_path):
    runs = tmp_path / "modelA"
    for i, (closed, tick) in enumerate([(True, 120), (False, None), (True, 200)]):
        d = runs / f"run_{i:04d}"
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({"closed": closed, "closed_tick": tick, "aborted": False}))
    rows = tally.tally(str(tmp_path))
    assert rows == [{"model": "modelA", "runs": 3, "closed": 2, "aborted": 0, "median_close_tick": 200}]
    assert "modelA" in tally.format_rows(rows)
