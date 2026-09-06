"""Trainer argument resolution: presets, ablation mapping, dropout switch,
environment fallback and the pi0 init guard.  No torch / lerobot needed."""
from __future__ import annotations

import os

import pytest

from simretract.train import presets
from simretract.train import train

pytestmark = pytest.mark.unit

#: Legacy environment knobs the trainer honours; cleared for every test so
#: the developer's shell cannot leak into the assertions.
LEGACY_ENV = ("CACHE", "SETS", "SET_FRAC", "FLOAT_DEPTH", "NOGRIP", "SAVE_EVERY",
              "SUBSET_JSON", "NO_AUG", "SMOLVLA_FULL", "ACT_BASELINE_OPT",
              "PI05_OFFICIAL_OPT", "CHUNK_H", "VIDEO_T", "VIDEO_RATIO",
              "DROP_MASK_P", "DROP_PLANE_P", "DROP_START", "PLANE_SIG",
              "PLANE_SIG_END", "PLANE_GLITCH_P", "PLANE_GLITCH_END",
              "PLANE_RAND_FRAC", "HF_HUB_OFFLINE", "INIT", "NUM_WORKERS",
              "STRICT_JSON", "WAM_GC", "WAM_FREEZE_VIDEO", "WAM_SHIFT",
              "WAM_LAMBDA_VIDEO", "WAM_FP32_MASTER", "WAM_BETA2", "WAM_COS_FLOOR")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in LEGACY_ENV:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv("SIMRETRACT_" + name, raising=False)


#: The launch configuration of the nine presets, translated to trainer
#: flags (cache set names strict/recovery replace the legacy set prefixes).
#: Every row: FLOAT_DEPTH=1 NOGRIP=1 SAVE_EVERY=5000, --input_mode masked_rgbd
#: --state qpos_plane --episodes all --steps 15000 --batch 16 --accum 4,
#: dropout env DROP_MASK_P=0.25 DROP_PLANE_P=0 DROP_START=0.5 PLANE_SIG_END=8
#: PLANE_GLITCH_END=0.25 PLANE_RAND_FRAC=0.5 (no_dropout: the knobs unset).
COMMON = {
    "float_depth": True, "nogrip": True, "save_every": 5000, "no_aug": False,
    "input_mode": "masked_rgbd", "state": "qpos_plane", "episodes": "all",
    "steps": 15000, "batch": 16, "accum": 4, "lr": 2.5e-5, "warmup": 1000,
    "normalize": "zscore", "trim": "close", "resample": "arc", "chunk_h": 50,
    "video_t": 0, "video_ratio": 6,
    "drop_mask_p": 0.25, "drop_plane_p": 0.0, "drop_start": 0.5,
    "plane_sig": 3.0, "plane_sig_end": 8.0, "plane_glitch_p": 0.03,
    "plane_glitch_end": 0.25, "plane_rand_frac": 0.5,
}
FULL = {"policy": "pi0", "data": "full", "ablation": "none", "dropout": True,
        "set_list": ["strict", "recovery"], "set_frac_list": [0.5, 0.5]}
PRESET_ROWS = {
    # both sets, SET_FRAC=0.5,0.5
    "baseline": {**COMMON, **FULL},
    # the dropout knobs unset: blind drop 0, constant plane noise
    "no_dropout": {**COMMON, **FULL, "dropout": False, "drop_mask_p": 0.0, "drop_plane_p": 0.0,
                   "plane_sig_end": 3.0, "plane_glitch_end": 0.03, "plane_rand_frac": 0.0},
    # strict set only, no SET_FRAC
    "no_recovery": {**COMMON, **FULL, "data": "strict_only", "set_list": ["strict"],
                    "set_frac_list": None},
    # SUBSET_JSON from the half selection
    "half_data": {**COMMON, **FULL, "data": "half"},
    # --state qpos
    "no_plane": {**COMMON, **FULL, "ablation": "no_plane", "state": "qpos"},
    # --input_mode full_rgbd
    "no_mask": {**COMMON, **FULL, "ablation": "no_mask", "input_mode": "full_rgbd"},
    # baseline flags on the no-smoothing pair
    "no_smooth": {**COMMON, **FULL, "set_list": ["strict_nosmooth", "recovery_nosmooth"]},
    # baseline flags on the no-standoff pair
    "no_standoff": {**COMMON, **FULL, "set_list": ["strict_nostandoff", "recovery_nostandoff"]},
    # the SmolVLA backbone (full unfreeze) on the baseline flags and data
    "smolvla": {**COMMON, **FULL, "policy": "smolvla", "smolvla_full": True},
}


def _yaml_available() -> bool:
    return presets._yaml_presets() is not None


@pytest.mark.parametrize("name", list(PRESET_ROWS))
@pytest.mark.parametrize("use_config", [False, True])
def test_presets_match_their_launch_rows(name, use_config):
    if use_config and not _yaml_available():
        pytest.skip("configs/train/presets.yaml not loadable in this checkout")
    got = presets.resolve(name, use_config=use_config)
    for key, val in PRESET_ROWS[name].items():
        assert got[key] == val, f"{name}: {key} = {got[key]!r}, expected {val!r}"
    assert got["preset"] == name


def test_preset_table_is_exactly_the_nine_rows():
    assert presets.preset_names(use_config=False) == list(PRESET_ROWS)
    if _yaml_available():
        assert presets.preset_names(use_config=True) == list(PRESET_ROWS)


@pytest.mark.parametrize("name, selection", [
    ("baseline", "strict_500.json"),
    ("half_data", "strict_500.json"),
    ("no_smooth", "strict_500_no_smoothing.json"),
    ("no_standoff", "strict_500_no_standoff.json"),
])
def test_strict_json_follows_the_strict_set(name, selection):
    """``--data half`` stratifies on the strict selection of the pair in use."""
    if not _yaml_available():
        pytest.skip("configs not loadable")
    a = train.parse_args(["--preset", name, "--data", "half"], use_config=True)
    assert os.path.basename(a.strict_json) == selection
    assert a.set_list[0].startswith("strict") and a.set_list[1].startswith("recovery")
    b = train.parse_args(["--preset", name, "--data", "half", "--strict-json", "x.json"], use_config=True)
    assert b.strict_json == "x.json"


@pytest.mark.parametrize("name", presets.preset_names(use_config=False))
def test_every_builtin_preset_resolves(name):
    got = presets.resolve(name, use_config=False)
    assert got["policy"] in train.POLICIES
    assert got["input_mode"] in train.INPUT_MODES
    assert got["state"] in train.STATE_MODES
    if got["policy"] == "fastwam":
        assert (got["chunk_h"], got["video_t"]) == (48, 9)


@pytest.mark.skipif(not _yaml_available(), reason="presets.yaml not loadable")
def test_yaml_and_builtin_tables_agree():
    yaml_names = presets.preset_names(use_config=True)
    assert set(yaml_names) == set(presets.BUILTIN_PRESETS)
    for name in yaml_names:
        y = presets.resolve(name, use_config=True)
        b = presets.resolve(name, use_config=False)
        for key in ("policy", "data", "ablation", "dropout", "input_mode", "state",
                    "set_list", "set_frac_list", "no_aug", "chunk_h", "video_t",
                    "smolvla_full", *train.DROPOUT_DEFAULTS):
            assert y[key] == b[key], f"{name}: {key} yaml={y[key]!r} builtin={b[key]!r}"


def test_no_dropout_resets_the_plane_knobs():
    a = train.parse_args(["--policy", "act", "--no-dropout"], use_config=False)
    assert a.dropout is False
    assert a.drop_mask_p == 0.0 and a.drop_plane_p == 0.0
    assert a.plane_sig_end == a.plane_sig == 3.0
    assert a.plane_glitch_end == a.plane_glitch_p == 0.03
    assert a.plane_rand_frac == 0.0
    on = train.parse_args(["--policy", "act"], use_config=False)
    assert (on.plane_sig_end, on.plane_glitch_end, on.plane_rand_frac) == (8.0, 0.25, 0.5)
    assert on.drop_mask_p == 0.25


def test_no_dropout_overrides_a_dropout_preset():
    a = train.parse_args(["--preset", "baseline", "--no-dropout"], use_config=False)
    assert a.dropout is False and a.drop_mask_p == 0.0 and a.plane_sig_end == 3.0


@pytest.mark.parametrize("ablation, input_mode, state, no_aug", [
    ("none", "masked_rgbd", "qpos_plane", False),
    ("no_plane", "masked_rgbd", "qpos", False),
    ("no_mask", "full_rgbd", "qpos_plane", False),
    ("no_depth", "masked_rgb_nodepth", "qpos_plane", False),
    ("no_aug", "masked_rgbd", "qpos_plane", True),
])
def test_ablation_mapping(ablation, input_mode, state, no_aug):
    a = train.parse_args(["--ablation", ablation], use_config=False)
    assert (a.input_mode, a.state, a.no_aug) == (input_mode, state, no_aug)


def test_conflicting_explicit_flag_is_an_error():
    with pytest.raises(SystemExit):
        train.parse_args(["--ablation", "no_depth", "--input-mode", "masked_rgbd"],
                         use_config=False)


def test_data_modes():
    full = train.parse_args(["--data", "full"], use_config=False)
    assert full.set_list == ["strict", "recovery"] and full.set_frac_list == [0.5, 0.5]
    so = train.parse_args(["--data", "strict_only"], use_config=False)
    assert so.set_list == ["strict"] and so.set_frac_list is None
    half = train.parse_args(["--data", "half"], use_config=False)
    assert half.data == "half" and half.set_list == ["strict", "recovery"]
    assert half.strict_json.endswith(".json")
    with pytest.raises(SystemExit):
        train.parse_args(["--data", "half", "--sets", "strict"], use_config=False)
    three = train.parse_args(["--sets", "a,b,c", "--set-frac", "none"], use_config=False)
    assert three.set_frac_list is None
    with pytest.raises(SystemExit):
        train.parse_args(["--sets", "a,b", "--set-frac", "1"], use_config=False)


def test_explicit_flags_win_over_preset():
    a = train.parse_args(["--preset", "baseline", "--steps", "10", "--batch", "8",
                          "--accum", "8", "--sets", "strict", "--set-frac", "none"],
                         use_config=False)
    assert (a.steps, a.batch, a.accum) == (10, 8, 8)
    assert a.set_list == ["strict"] and a.set_frac_list is None


def test_smoke_defaults():
    a = train.parse_args(["--policy", "act", "--smoke"], use_config=False)
    assert (a.batch, a.accum) == (4, 1) and a.preset is None
    b = train.parse_args(["--policy", "act", "--smoke", "--batch", "2"], use_config=False)
    assert (b.batch, b.accum) == (2, 1)
    c = train.parse_args(["--preset", "baseline", "--smoke"], use_config=False)
    assert (c.batch, c.accum) == (4, 1)


def test_fastwam_defaults_and_policy_switch():
    a = train.parse_args(["--policy", "fastwam"], use_config=False)
    assert (a.chunk_h, a.video_t, a.video_ratio) == (48, 9, 6)
    assert a.wam_gc is True and a.wam_shift == 5.0 and a.wam_fp32_master is False
    b = train.parse_args(["--policy", "pi05"], use_config=False)
    assert b.chunk_h == 50 and b.pi05_opt == "baseline"


def test_other_backbones_are_flags_not_presets():
    """The backbone switch and its knobs stay on the trainer; only smolvla has a preset."""
    assert set(train.POLICIES) == {"pi0", "smolvla", "act", "fastwam", "pi05", "pi0_fast"}
    for policy in train.POLICIES:
        assert (policy in presets.preset_names(use_config=False)) == (policy == "smolvla")
    a = train.parse_args(["--preset", "baseline", "--policy", "act", "--act-opt", "official", "--init", ""],
                         use_config=False)
    assert a.policy == "act" and a.act_opt == "official" and a.set_list == ["strict", "recovery"]
    b = train.parse_args(["--policy", "smolvla", "--no-smolvla-full"], use_config=False)
    assert b.smolvla_full is False and b.dropout is True


def test_env_fallback(monkeypatch):
    monkeypatch.setenv("CACHE", "/tmp/cache-legacy")
    monkeypatch.setenv("SIMRETRACT_SETS", "x,y")
    monkeypatch.setenv("DROP_MASK_P", "0.1")
    monkeypatch.setenv("NOGRIP", "0")
    monkeypatch.setenv("ACT_BASELINE_OPT", "0")
    monkeypatch.setenv("SMOLVLA_FULL", "0")
    monkeypatch.setenv("SAVE_EVERY", "1000")
    a = train.parse_args(["--policy", "act"], use_config=False)
    assert a.cache == "/tmp/cache-legacy"
    assert a.set_list == ["x", "y"] and a.set_frac_list == [0.5, 0.5]
    assert a.drop_mask_p == 0.1 and a.nogrip is False
    assert a.act_opt == "official" and a.smolvla_full is False
    assert a.save_every == 1000
    monkeypatch.setenv("SIMRETRACT_CACHE", "/tmp/cache-new")
    assert train.parse_args([], use_config=False).cache == "/tmp/cache-new"
    b = train.parse_args(["--cache", "/tmp/cli"], use_config=False)
    assert b.cache == "/tmp/cli"


def test_pi0_init_guard(tmp_path):
    a = train.parse_args(["--policy", "pi0", "--init", "some-org/pi0_base"],
                         use_config=False)
    with pytest.raises(SystemExit):
        train.validate_init(a)
    a.allow_unverified_init = True
    train.validate_init(a)
    ckpt = tmp_path / "pi0_base"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    (ckpt / "model.safetensors").write_bytes(b"")
    b = train.parse_args(["--policy", "pi0", "--init", str(ckpt)], use_config=False)
    train.validate_init(b)
    c = train.parse_args(["--policy", "pi0", "--init", str(tmp_path / "missing")],
                         use_config=False)
    with pytest.raises(SystemExit):
        train.validate_init(c)
    d = train.parse_args(["--policy", "act", "--init", "whatever"], use_config=False)
    train.validate_init(d)  # only pi0 is guarded


def test_default_pi0_init_is_a_local_checkpoints_dir():
    a = train.parse_args([], use_config=False)
    assert a.policy == "pi0"
    assert os.path.basename(a.init) == "pi0_base"
    assert "://" not in a.init and not a.init.count("/") == 1


def test_train_regime_record_and_knobs():
    a = train.parse_args(["--policy", "smolvla"], use_config=False)
    reg = train.train_regime(a)
    assert reg["policy"] == "smolvla" and reg["smolvla_full"] is True and reg["preset"] is None
    assert reg["dropout_knobs"]["plane_sig_end"] == 8.0
    k = train.knobs_from_args(a)
    assert (k.drop_mask_p, k.plane_glitch_end, k.chunk_h) == (0.25, 0.25, 50)


def test_help_renders_hardware_notes(capsys):
    with pytest.raises(SystemExit):
        train.build_parser(use_config=False).parse_args(["--help"])
    out = capsys.readouterr().out
    assert "H200" in out and "--batch 8 --accum 8" in out
    for flag in ("--drop-mask-p", "--plane-rand-frac", "--smolvla-full", "--act-opt",
                 "--hf-offline", "--allow-unverified-init", "--data", "--ablation"):
        assert flag in out
