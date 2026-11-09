"""Training presets resolve to exactly the launch configuration of their rows.

Each of the nine presets is translated back into the environment block and
argument list of its launch configuration (the legacy environment-variable
form of the trainer knobs) and compared against those literal sets.  Number
formatting goes through ``%g`` on both sides so ``0.25``, ``8`` and
``2.5e-05`` compare as strings the way a shell would pass them.

Presets carry flags and a description only: no preset records a result.
"""

from __future__ import annotations

from typing import Any

import pytest

from simretract import config
from simretract.train import presets as _presets

pytestmark = pytest.mark.unit

PRESETS = ("baseline", "no_dropout", "no_recovery", "half_data", "no_plane", "no_mask",
           "no_smooth", "no_standoff", "smolvla")

#: Cache set names of the main pair as recorded in norm.json of checkpoints trained
#: before the rename (configs/data.yaml legacy_name); the other sets keep their names.
LEGACY_SET = {"strict": "rake_e5", "recovery": "step1e5far1000"}


def _g(v: Any) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return "%g" % v
    return str(v)


def to_legacy(flags: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    """Map a resolved preset to (environment block, argparse flags) in the legacy launch form."""
    env: dict[str, str] = {
        "SETS": ",".join(LEGACY_SET.get(s, s) for s in flags["sets"]),
        "FLOAT_DEPTH": _g(flags["float_depth"]),
        "NOGRIP": _g(flags["nogrip"]),
        "SAVE_EVERY": _g(flags["save_every"]),
        "CHUNK_H": _g(flags["chunk_h"]),
        "DROP_MASK_P": _g(flags["drop_mask_p"]),
        "DROP_PLANE_P": _g(flags["drop_plane_p"]),
        "DROP_START": _g(flags["drop_start"]),
        "PLANE_SIG": _g(flags["plane_sig"]),
        "PLANE_SIG_END": _g(flags["plane_sig_end"]),
        "PLANE_GLITCH_P": _g(flags["plane_glitch_p"]),
        "PLANE_GLITCH_END": _g(flags["plane_glitch_end"]),
        "PLANE_RAND_FRAC": _g(flags["plane_rand_frac"]),
    }
    if flags["set_frac"] is not None:
        env["SET_FRAC"] = ",".join(_g(x) for x in flags["set_frac"])
    if flags["no_aug"]:
        env["NO_AUG"] = "1"
    if flags["data"] == "half":
        env["SUBSET_JSON"] = "<half_subset.json>"
    if flags["policy"] == "smolvla":
        env["SMOLVLA_FULL"] = _g(flags["smolvla_full"])
    if flags["policy"] == "act":
        env["ACT_BASELINE_OPT"] = "1" if flags["act_opt"] == "baseline" else "0"
    if flags["video_t"]:
        env["VIDEO_T"] = _g(flags["video_t"])
        env["VIDEO_RATIO"] = _g(flags["video_ratio"])

    args: dict[str, str] = {
        "--policy": flags["policy"],
        "--input_mode": flags["input_mode"],
        "--state": flags["state"],
        "--episodes": flags["episodes"],
        "--trim": flags["trim"],
        "--resample": flags["resample"],
        "--normalize": flags["normalize"],
        "--steps": _g(flags["steps"]),
        "--batch": _g(flags["batch"]),
        "--accum": _g(flags["accum"]),
        "--lr": _g(flags["lr"]),
        "--warmup": _g(flags["warmup"]),
    }
    init = flags["init"]
    if init:
        args["--init"] = "<pi0_base>" if str(init).endswith("/pi0_base") else str(init)
    return env, args


# The launch configuration of the dropout baseline (environment block + trainer defaults).
COMMON_ENV = {
    "SETS": "rake_e5,step1e5far1000",
    "SET_FRAC": "0.5,0.5",
    "FLOAT_DEPTH": "1",
    "NOGRIP": "1",
    "SAVE_EVERY": "5000",
    "CHUNK_H": "50",
    "DROP_MASK_P": "0.25",
    "DROP_PLANE_P": "0",
    "DROP_START": "0.5",
    "PLANE_SIG": "3",
    "PLANE_SIG_END": "8",
    "PLANE_GLITCH_P": "0.03",
    "PLANE_GLITCH_END": "0.25",
    "PLANE_RAND_FRAC": "0.5",
}
COMMON_ARGS = {
    "--policy": "pi0",
    "--input_mode": "masked_rgbd",
    "--state": "qpos_plane",
    "--episodes": "all",
    "--trim": "close",
    "--resample": "arc",
    "--normalize": "zscore",
    "--steps": "15000",
    "--batch": "16",
    "--accum": "4",
    "--lr": "2.5e-05",
    "--warmup": "1000",
    "--init": "<pi0_base>",
}
# The non-dropout launch: no DROP_* / PLANE_*_END variables set, i.e. the trainer's
# pre-dropout defaults (blind drop 0, plane noise constant at sigma 3 deg / 3% glitches).
NO_DROPOUT_ENV = {
    **COMMON_ENV,
    "DROP_MASK_P": "0",
    "DROP_PLANE_P": "0",
    "PLANE_SIG_END": "3",
    "PLANE_GLITCH_END": "0.03",
    "PLANE_RAND_FRAC": "0",
}


def _without(d: dict[str, str], *keys: str) -> dict[str, str]:
    return {k: v for k, v in d.items() if k not in keys}


EXPECTED: dict[str, tuple[dict[str, str], dict[str, str]]] = {
    # pi0 + dropout baseline
    "baseline": (dict(COMMON_ENV), dict(COMMON_ARGS)),
    # the same launch with the dropout rule off
    "no_dropout": (dict(NO_DROPOUT_ENV), dict(COMMON_ARGS)),
    # strict set only, no SET_FRAC (single set)
    "no_recovery": ({**_without(COMMON_ENV, "SET_FRAC"), "SETS": "rake_e5"}, dict(COMMON_ARGS)),
    # half selection passed through SUBSET_JSON
    "half_data": ({**COMMON_ENV, "SUBSET_JSON": "<half_subset.json>"}, dict(COMMON_ARGS)),
    # 7-D state
    "no_plane": (dict(COMMON_ENV), {**COMMON_ARGS, "--state": "qpos"}),
    # unmasked image streams
    "no_mask": (dict(COMMON_ENV), {**COMMON_ARGS, "--input_mode": "full_rgbd"}),
    # baseline flags on the no-smoothing cache pair
    "no_smooth": ({**COMMON_ENV, "SETS": "strict_nosmooth,recovery_nosmooth"}, dict(COMMON_ARGS)),
    # baseline flags on the no-standoff cache pair
    "no_standoff": ({**COMMON_ENV, "SETS": "strict_nostandoff,recovery_nostandoff"}, dict(COMMON_ARGS)),
    # the SmolVLA backbone (full unfreeze) on the baseline flags and data
    "smolvla": ({**COMMON_ENV, "SMOLVLA_FULL": "1"},
                {**COMMON_ARGS, "--policy": "smolvla", "--init": "lerobot/smolvla_base"}),
}


@pytest.mark.parametrize("name", PRESETS)
def test_preset_matches_its_launch(name: str) -> None:
    env, args = to_legacy(config.preset_flags(name))
    want_env, want_args = EXPECTED[name]
    assert env == want_env, name
    assert args == want_args, name


def test_all_presets_exist_and_are_complete() -> None:
    names = config.preset_names()
    assert names == list(PRESETS)
    assert tuple(_presets.PRESET_ORDER) == PRESETS
    defaults = set(config.load_config("train/presets")["defaults"])
    for name in names:
        p = config.preset(name)
        assert p["name"] == name and p["description"]
        assert set(config.preset_flags(name)) == defaults, name
        assert "${" not in str(p), name


def test_no_preset_carries_results() -> None:
    raw = config.load_config("train/presets")
    for name, row in raw["presets"].items():
        assert "reference" not in (row or {}), name
        assert set(row or {}) <= set(raw["defaults"]) | {"description"}, name
    assert "reference" not in config.PRESET_META_KEYS
    text = config.config_path("train/presets").read_text(encoding="utf-8")
    for word in ("reference:", "grabbed", "secured", "timeouts"):
        assert word not in text, word


def test_switch_semantics_agree_with_concrete_flags() -> None:
    """The switches and the flags the trainer actually reads never disagree."""
    for name in config.preset_names():
        f = config.preset_flags(name)
        assert f["policy"] == ("smolvla" if name == "smolvla" else "pi0")
        assert f["smolvla_full"] is True
        if f["data"] == "strict_only":
            assert f["sets"] == ["strict"] and f["set_frac"] is None
        else:
            assert len(f["sets"]) == 2 and f["set_frac"] == [0.5, 0.5]
            assert f["sets"][0].startswith("strict") and f["sets"][1].startswith("recovery")
        assert (f["ablation"] == "no_plane") == (f["state"] == "qpos")
        assert (f["ablation"] == "no_mask") == (f["input_mode"] == "full_rgbd")
        assert f["input_mode"] != "masked_rgb_nodepth" and not f["no_aug"]
        assert f["dropout"] == (name != "no_dropout")
        if f["dropout"]:
            assert (f["drop_mask_p"], f["plane_sig_end"], f["plane_glitch_end"], f["plane_rand_frac"]) == (0.25, 8.0, 0.25, 0.5)
        else:
            assert (f["drop_mask_p"], f["plane_sig_end"], f["plane_glitch_end"], f["plane_rand_frac"]) == (0.0, 3.0, 0.03, 0.0)
        assert (f["drop_plane_p"], f["drop_start"], f["plane_sig"], f["plane_glitch_p"]) == (0.0, 0.5, 3.0, 0.03)
        if f["policy"] == "pi0":
            assert str(f["init"]).endswith("/pi0_base") and "/" in str(f["init"])
        else:
            assert f["init"] == config.hub()["base_weights"][f["policy"]]


def test_each_preset_changes_one_thing() -> None:
    """Exactly one switch (plus the flags it implies) differs from the baseline;
    drop_plane_p is 0 on both sides of the dropout switch and never changes."""
    base = config.preset_flags("baseline")
    expected = {
        "no_dropout": {"dropout", "drop_mask_p", "plane_sig_end", "plane_glitch_end", "plane_rand_frac"},
        "no_recovery": {"data", "sets", "set_frac"},
        "half_data": {"data"},
        "no_plane": {"ablation", "state"},
        "no_mask": {"ablation", "input_mode"},
        "no_smooth": {"sets"},
        "no_standoff": {"sets"},
        "smolvla": {"policy", "init"},   # smolvla_full is already on in the defaults
    }
    for name in PRESETS[1:]:
        f = config.preset_flags(name)
        changed = {k for k in base if f[k] != base[k]}
        assert changed == expected[name], (name, changed)


def test_data_variants_train_on_their_own_pairs() -> None:
    sets = config.data_config()["sets"]
    assert config.preset_flags("no_smooth")["sets"] == ["strict_nosmooth", "recovery_nosmooth"]
    assert config.preset_flags("no_standoff")["sets"] == ["strict_nostandoff", "recovery_nostandoff"]
    for name in ("no_smooth", "no_standoff"):
        for s in config.preset_flags(name)["sets"]:
            assert s in sets, s


def test_available_checkpoints_and_train_only_presets() -> None:
    rows = set(config.hub()["checkpoints"]) - {"pi0_base", "act"}
    assert rows <= set(PRESETS)
    assert set(PRESETS) - rows == {"no_plane", "no_mask", "no_smooth", "no_standoff"}


def test_unknown_preset_lists_alternatives() -> None:
    with pytest.raises(KeyError, match="baseline"):
        config.preset("nonexistent")
