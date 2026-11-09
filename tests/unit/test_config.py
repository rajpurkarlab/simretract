"""Configuration layer: roots, paths, hub ids, interpreters, data fingerprints and selections."""

from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

import pytest

from simretract import config
from simretract.config import ConfigError

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- roots and paths


def test_repo_root_is_the_checkout(repo_root: Path) -> None:
    assert config.repo_root() == repo_root
    assert (config.repo_root() / "configs" / "paths.yaml").is_file()


def test_paths_keys_are_absolute(clean_env: pytest.MonkeyPatch) -> None:
    p = config.paths()
    expected = {"data_raw", "data_cache", "checkpoints", "runs", "eval", "assets", "selections", "third_party", "configs"}
    assert set(p) == expected
    for key, value in p.items():
        assert isinstance(value, Path), key
        assert value.is_absolute(), key
    root = config.repo_root()
    assert p["assets"] == root / "assets"
    assert p["selections"] == root / "data" / "selections"
    assert p["data_raw"] == root / "data" / "raw"
    assert p["checkpoints"] == root / "checkpoints"


def test_storage_root_relocates_only_storage(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clean_env.setenv("SIMRETRACT_ROOT", str(tmp_path))
    p = config.paths()
    for key in config.STORAGE_KEYS:
        assert p[key].is_relative_to(tmp_path), key
    for key in config.REPO_KEYS:
        assert p[key].is_relative_to(config.repo_root()), key


def test_single_storage_override(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clean_env.setenv("SIMRETRACT_DATA_CACHE", str(tmp_path / "cache"))
    p = config.paths()
    assert p["data_cache"] == (tmp_path / "cache").resolve()
    assert p["data_raw"] == config.repo_root() / "data" / "raw"


def test_selection_path(clean_env: pytest.MonkeyPatch) -> None:
    assert config.selection_path("test_core50") == config.repo_root() / "data" / "selections" / "test_core50.json"
    assert config.selection_path("test_core50.json").is_file()


# --------------------------------------------------------------------------- yaml helpers


def test_load_yaml_missing_file_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        config.load_yaml(tmp_path / "nope.yaml")


def test_load_yaml_empty_and_non_mapping(tmp_path: Path) -> None:
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert config.load_yaml(empty) == {}
    seq = tmp_path / "seq.yaml"
    seq.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        config.load_yaml(seq)


def test_config_path_adds_suffix() -> None:
    assert config.config_path("eval/sim").name == "sim.yaml"
    assert config.config_path("eval/sim.yaml") == config.config_path("eval/sim")


# --------------------------------------------------------------------------- env_or


def test_env_or_default_and_cast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIMRETRACT_T_A", raising=False)
    assert config.env_or("SIMRETRACT_T_A", 0.03) == 0.03
    monkeypatch.setenv("SIMRETRACT_T_A", "0.5")
    assert config.env_or("SIMRETRACT_T_A", 0.03) == 0.5
    monkeypatch.setenv("SIMRETRACT_T_A", "7")
    assert config.env_or("SIMRETRACT_T_A", 3) == 7
    assert config.env_or("SIMRETRACT_T_A", "x") == "7"
    assert config.env_or("SIMRETRACT_T_A", None) == "7"
    assert config.env_or("SIMRETRACT_T_A", 0, cast=lambda s: int(s) * 2) == 14


def test_env_or_bool_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    for raw, want in [("1", True), ("0", False), ("true", True), ("no", False), ("YES", True), ("off", False)]:
        monkeypatch.setenv("SIMRETRACT_T_B", raw)
        assert config.env_or("SIMRETRACT_T_B", False) is want, raw
    monkeypatch.setenv("SIMRETRACT_T_B", "maybe")
    with pytest.raises(ConfigError, match="SIMRETRACT_T_B"):
        config.env_or("SIMRETRACT_T_B", False)


def test_env_or_legacy_fallback_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIMRETRACT_STANDOFF", raising=False)
    monkeypatch.setenv("STANDOFF", "0.02")
    assert config.env_or(("SIMRETRACT_STANDOFF", "STANDOFF"), 0.03) == 0.02
    monkeypatch.setenv("SIMRETRACT_STANDOFF", "0.01")
    assert config.env_or(("SIMRETRACT_STANDOFF", "STANDOFF"), 0.03) == 0.01


def test_env_or_bad_cast_names_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIMRETRACT_T_C", "abc")
    with pytest.raises(ConfigError, match="SIMRETRACT_T_C"):
        config.env_or("SIMRETRACT_T_C", 1)


def test_require_kit_eula(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)
    with pytest.raises(ConfigError, match="OMNI_KIT_ACCEPT_EULA"):
        config.require_kit_eula()
    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "YES")
    config.require_kit_eula()


# --------------------------------------------------------------------------- hub


def test_hub_models_configured_and_dataset_placeholders() -> None:
    from simretract.data.hub import _require_repo

    h = config.hub()
    assert config.hub_is_configured()
    assert _require_repo(h, "checkpoints") == "dddw123/omni-retract-checkpoints-public"
    assert "${" not in json.dumps(h)
    assert h["checkpoints"]["pi0_base"]["path"] == "pi0_base_usmd1"
    assert h["checkpoints"]["baseline"]["path"] == "pi0_e5_step1far1000_plane_ng_bal_drop"
    for entry in h["checkpoints"].values():
        for filename in entry["files"]:
            assert entry["sizes"][filename] > 0
            assert len(entry["sha256"][filename]) == 64
    with pytest.raises(config.ConfigError):
        _require_repo(h, "strict_raw")
    for key in ("strict_raw", "recovery_raw", "strict_nosmooth_raw", "recovery_nosmooth_raw",
                "strict_nostandoff_raw", "recovery_nostandoff_raw"):
        assert h["repos"][key].startswith("YOUR_HF_ORG/simretract-"), key
    bw = h["base_weights"]
    assert bw["sam3"] == "facebook/sam3"
    assert bw["paligemma_tokenizer"] == "google/paligemma-3b-pt-224"
    assert bw["smolvla"] == "lerobot/smolvla_base"
    assert bw["smolvlm_tokenizer"] == "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


# --------------------------------------------------------------------------- interpreters


def test_interpreter_unknown_role() -> None:
    with pytest.raises(ConfigError, match="unknown interpreter role"):
        config.interpreter("gpu")


def test_interpreter_template_placeholder_is_actionable(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Point the lookup at a template-only configuration (no envs.local.yaml).
    cfg = tmp_path / "configs"
    cfg.mkdir()
    for name in ("paths.yaml", "envs.yaml"):
        cfg.joinpath(name).write_text(config.config_path(name).read_text(encoding="utf-8"), encoding="utf-8")
    clean_env.setenv("SIMRETRACT_REPO", str(tmp_path))
    with pytest.raises(ConfigError, match="envs/setup.sh"):
        config.interpreter("policy", validate=False)


def test_interpreter_env_override_and_validation(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("SIMRETRACT_POLICY_PYTHON", sys.executable)
    assert config.interpreter("policy", validate=False) == Path(sys.executable)
    # The running interpreter certainly imports `yaml`; validate against it instead of lerobot.
    clean_env.setitem(config.ROLES, "policy", "yaml")
    assert config.interpreter("policy") == Path(sys.executable)
    clean_env.setitem(config.ROLES, "policy", "definitely_not_a_module_xyz")
    config._validate_interpreter.cache_clear()
    with pytest.raises(ConfigError, match="cannot import"):
        config.interpreter("policy")
    config._validate_interpreter.cache_clear()


def test_interpreter_missing_file(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clean_env.setenv("SIMRETRACT_SIM_PYTHON", str(tmp_path / "python"))
    with pytest.raises(ConfigError, match="does not exist"):
        config.interpreter("sim", validate=False)


def test_config_path_appends_suffix_to_dotted_names() -> None:
    assert config.config_path("envs.local").name == "envs.local.yaml"
    assert config.config_path("envs.local.yaml").name == "envs.local.yaml"
    assert config.config_path("train/presets") == config.config_dir() / "train" / "presets.yaml"


def test_interpreter_prefers_local_file_over_template(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # envs/setup.sh writes configs/envs.local.yaml; it must win over the tracked template.
    cfg = tmp_path / "configs"
    cfg.mkdir()
    for name in ("paths.yaml", "envs.yaml"):
        cfg.joinpath(name).write_text(config.config_path(name).read_text(encoding="utf-8"), encoding="utf-8")
    cfg.joinpath("envs.local.yaml").write_text(f"policy_python: {sys.executable}\n", encoding="utf-8")
    clean_env.setenv("SIMRETRACT_REPO", str(tmp_path))
    assert config.interpreter("policy", validate=False) == Path(sys.executable)
    with pytest.raises(ConfigError, match="envs/setup.sh"):
        config.interpreter("mask", validate=False)  # not in the local file, template placeholder


# --------------------------------------------------------------------------- vendor robot


def test_trossen_paths(clean_env: pytest.MonkeyPatch) -> None:
    scripts = config.paths()["third_party"] / "trossen_ai_isaac" / "scripts"
    if not (scripts / "controller.py").is_file():
        pytest.skip("submodule not checked out")
    assert config.trossen_scripts_path() == scripts
    assert config.trossen_robot_usd().name == "wxai_follower.usd"
    added = config.ensure_trossen_on_path()
    assert str(added) in sys.path


# --------------------------------------------------------------------------- data config and selections


def _sel(name: str) -> dict:
    return json.loads(config.selection_path(name).read_text(encoding="utf-8"))


def test_data_fingerprints_match_the_reference_builds() -> None:
    d = config.data_config()
    strict, rec = d["sets"]["strict"], d["sets"]["recovery"]
    assert strict["fingerprint"] == {"frames": 69402, "md5_8": {"action": "7d0e2c3b", "plane": "034cb12d", "gpose": "6a415e21"}}
    assert rec["fingerprint"] == {"frames": 106868, "md5_8": {"action": "636630ba", "plane": "93dda3b0", "gpose": "7f918202"}}
    base = {"pretrim": True, "keep_only": True, "seed_min": 200000, "skip_gmk": True}
    assert strict["cache_flags"] == {**base, "perturb_trim": False}
    assert rec["cache_flags"] == {**base, "perturb_trim": True}
    assert d["test"]["exclude_from_training"] == "test_pool.json"
    for key in ("strict", "recovery"):
        assert config.selection_path(d["sets"][key]["selection"]).is_file()
    for key in ("pool", "core50"):
        assert config.selection_path(d["test"][key]).is_file()
    # the no-smoothing / no-standoff pairs: same cache flags, no recorded fingerprint (verify is skipped)
    repos = config.hub()["repos"]
    for key, kind in (("strict_nosmooth", "strict"), ("recovery_nosmooth", "recovery"),
                      ("strict_nostandoff", "strict"), ("recovery_nostandoff", "recovery")):
        entry = d["sets"][key]
        assert entry["cache_flags"] == d["sets"][kind]["cache_flags"], key
        assert entry["fingerprint"] is None, key
        assert entry["episodes"] == d["sets"][kind]["episodes"], key
        assert entry["hub_repo"] in repos, key
        assert config.selection_path(entry["selection"]).is_file(), key
    assert d["sets"]["strict_nosmooth"]["selection"] == "strict_500_no_smoothing.json"
    assert d["sets"]["recovery_nosmooth"]["selection"] == "recovery_far1000_no_smoothing.json"
    assert d["sets"]["strict_nostandoff"]["selection"] == "strict_500_no_standoff.json"
    assert d["sets"]["recovery_nostandoff"]["selection"] == "recovery_far1000_no_standoff.json"


def test_selection_counts_and_disjointness() -> None:
    strict = _sel("strict_500")
    far = _sel("recovery_far1000")
    pool_json = _sel("recovery_train_2000")
    test_pool, core50 = _sel("test_pool"), _sel("test_core50")

    assert len(strict["seeds"]) == 500 == len(set(strict["seeds"]))
    assert strict["per_bin"] == 125 and len(strict["bins_cm"]) == 4
    assert set(strict["grasp_z_cm"]) == {str(s) for s in strict["seeds"]}
    assert len(far["episodes"]) == far["n"] == 1000
    assert len(pool_json["train_episodes"]) == 1954
    assert set(far["episodes"]) <= set(pool_json["train_episodes"])
    assert len(test_pool["seeds"]) == test_pool["count"] == 115 == len(test_pool["episodes"])
    assert len(core50["seeds"]) == len(set(core50["seeds"])) == 50
    assert set(core50["seeds"]) <= set(test_pool["seeds"])
    # frozen test seeds never appear in a training selection
    assert not set(test_pool["seeds"]) & set(strict["seeds"])
    far_seeds = {int(e.split("_")[1]) for e in far["episodes"]}
    assert not set(test_pool["seeds"]) & far_seeds
    assert far_seeds <= set(strict["seeds"])


@pytest.mark.parametrize("variant", ["no_smoothing", "no_standoff"])
def test_variant_selection_counts_and_disjointness(variant: str) -> None:
    strict = _sel(f"strict_500_{variant}")
    far = _sel(f"recovery_far1000_{variant}")
    test_pool = _sel("test_pool")
    assert strict["name"] == f"strict_500_{variant}" and far["name"] == f"recovery_far1000_{variant}"
    assert len(strict["seeds"]) == 500 == len(set(strict["seeds"]))
    assert set(strict["grasp_z_cm"]) == {str(s) for s in strict["seeds"]}
    assert len(strict["bins_cm"]) == 4 and strict["per_bin"] == 125 and strict["rule"]
    assert len(far["episodes"]) == far["n"] == 1000 == len(set(far["episodes"]))
    assert far["episodes"] == sorted(far["episodes"]) and far["rule"]
    far_seeds = {int(e.split("_")[1]) for e in far["episodes"]}
    assert far_seeds <= set(strict["seeds"]) and len(far_seeds) == far["seeds_covered"]
    assert not set(test_pool["seeds"]) & set(strict["seeds"])
    assert not set(test_pool["seeds"]) & far_seeds
    assert min(strict["seeds"]) >= 200000
    if variant == "no_smoothing":
        # bin-balanced over the same 2.5 cm bins as the main strict set
        bins = collections.Counter(min(3, max(0, int((z + 5) / 2.5))) for z in strict["grasp_z_cm"].values())
        assert dict(bins) == {0: 125, 1: 125, 2: 125, 3: 125}
        assert far["frac_max"] == 0.5
    else:
        assert far["frac_max"] < 0.5


def test_far1000_is_the_smallest_perturbation_fraction() -> None:
    """The far-N rule (N smallest info.perturb.frac, ties by name) reproduces the shipped selection."""
    pool = _sel("recovery_train_2000")
    far = _sel("recovery_far1000")
    frac = pool["perturb_frac"]
    ranked = sorted(pool["train_episodes"], key=lambda e: (frac[e], e))[:1000]
    assert sorted(ranked) == sorted(far["episodes"])
    assert max(frac[e] for e in ranked) == pytest.approx(far["frac_max"], abs=1e-5)


def test_selection_files_carry_no_dates() -> None:
    for p in config.paths()["selections"].glob("*.json"):
        text = p.read_text(encoding="utf-8")
        assert "created" not in json.loads(text)
        assert "20" + "26-" not in text, p.name


# --------------------------------------------------------------------------- protocol configs


def test_eval_protocol_values() -> None:
    e = config.eval_protocol()
    assert e["exec_h"]["pi0"] == 50 and e["exec_h"]["fastwam"] == 48
    assert e["smooth"] == 0.45 and e["skip_search"] is True and e["max_steps"] == 450
    assert e["passes"] == 1 and e["score_attempt"] == "first"
    assert e["plane_top_cut"] == 0.0 and e["hand_style"] == "test"
    assert e["server"]["port"] == 8767 and e["server"]["half"] is True
    assert e["server"]["plane_filter"] is True and e["server"]["drop_px"] == 300
    assert e["seeds"] == "test_core50.json"


def test_generation_recipes() -> None:
    s = config.generation_recipe("strict")
    assert s["tool"] == "retractor_rake" and s["blend"] is True and s["keep_mode"] == "pose"
    assert s["standoff"] == 0.03 and s["fixed_standoff"] == 0.06 and s["ctrl_back"] == 0.06 and s["decel_arc"] == 0.0
    assert s["joint_rate_deg"] == 2.5 and s["speed_scale"] == 0.9 and s["entry_speed"] == 1.0
    assert s["skip_battery"] is True and s["interfere_gate"] is True and s["tries"] == 20
    assert s["z_lo"] == -0.05 and s["z_hi"] == 0.05 and s["mu"] == 0.4 and s["e4_pick"] is True
    assert s["hand_style"] == "train" and s["quota"]["per_bin"] == 125
    r = config.generation_recipe("recovery")
    assert r["draws"] == 4 and r["range"] == [0, 2000] and r["pert_frac_lo"] == 0.25 and r["pert_frac_hi"] == 0.75
    assert r["rng_xor"] == 0x50455254 and r["assemble"]["far_n"] == 1000 and r["assemble"]["exact_per_seed"] == 4
    for key in ("blend", "standoff", "fixed_standoff", "ctrl_back", "decel_arc", "entry_speed", "joint_rate_deg", "speed_scale",
                "keep_mode", "skip_battery", "interfere_gate", "hand_style", "mu"):
        assert r[key] == s[key], key
    with pytest.raises(ConfigError):
        config.generation_recipe("bogus")


def test_asset_meta_has_no_source_path(clean_env: pytest.MonkeyPatch) -> None:
    meta = json.loads((config.paths()["assets"] / "tools" / "retractor_rake" / "meta.json").read_text(encoding="utf-8"))
    assert "source" not in meta
    assert meta["name"] == "retractor_rake"
    tools = sorted(p.name for p in (config.paths()["assets"] / "tools").iterdir() if p.is_dir())
    assert tools == ["retractor_rake"]


# --------------------------------------------------------------------------- tool source


def test_tool_source_unset_is_none(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The tracked template says CHANGE_ME; with no local file that means 'not configured'."""
    clean_env.setattr(config, "_envs_file", lambda: config.config_path("envs"))
    assert config.tool_source() is None


def test_tool_source_from_env_and_local_file(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mesh = tmp_path / "pack.fbx"
    mesh.write_bytes(b"not really an fbx")
    local = tmp_path / "envs.local.yaml"
    local.write_text(f"sim_python: CHANGE_ME\ntool_source: {mesh}\n", encoding="utf-8")
    clean_env.setattr(config, "_envs_file", lambda: local)
    assert config.tool_source() == mesh
    other = tmp_path / "other.obj"
    other.write_bytes(b"v 0 0 0\n")
    clean_env.setenv("SIMRETRACT_TOOL_SOURCE", str(other))
    assert config.tool_source() == other


def test_tool_source_missing_file_is_actionable(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clean_env.setenv("SIMRETRACT_TOOL_SOURCE", str(tmp_path / "nope.fbx"))
    with pytest.raises(ConfigError, match="tool_source"):
        config.tool_source()
