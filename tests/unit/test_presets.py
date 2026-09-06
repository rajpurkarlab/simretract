"""Training presets resolve to exactly the launch configuration of the reference rows.

Each of the six reference presets is translated back into the environment
block and argument list of its reference launch configuration (the legacy
environment-variable form of the trainer knobs) and compared against those
literal sets.  Number formatting goes through ``%g`` on both sides so
``0.25``, ``8`` and ``2.5e-05`` compare as strings the way a shell would pass them.
"""

from __future__ import annotations

from typing import Any

import pytest

from simretract import config

pytestmark = pytest.mark.unit

REFERENCE = ("baseline", "no_recovery", "no_depth", "half_data", "smolvla", "act")
OPTIONAL = ("no_plane", "no_mask", "no_aug", "fastwam", "pi05", "pi0_fast")

#: Cache set names as recorded in norm.json of the released checkpoints (configs/data.yaml legacy_name).
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
        "SETS": ",".join(LEGACY_SET[s] for s in flags["sets"]),
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


# The launch configuration of every dropout row (environment block + trainer defaults).
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


def _without(d: dict[str, str], *keys: str) -> dict[str, str]:
    return {k: v for k, v in d.items() if k not in keys}


EXPECTED: dict[str, tuple[dict[str, str], dict[str, str]]] = {
    # pi0 + dropout baseline
    "baseline": (dict(COMMON_ENV), dict(COMMON_ARGS)),
    # A0: strict set only, no SET_FRAC (single set)
    "no_recovery": ({**_without(COMMON_ENV, "SET_FRAC"), "SETS": "rake_e5"}, dict(COMMON_ARGS)),
    # A3: single masked-RGB camera
    "no_depth": (dict(COMMON_ENV), {**COMMON_ARGS, "--input_mode": "masked_rgb_nodepth"}),
    # A7: half selection passed through SUBSET_JSON
    "half_data": ({**COMMON_ENV, "SUBSET_JSON": "<half_subset.json>"}, dict(COMMON_ARGS)),
    # SmolVLA-full: VLM + vision unfrozen, baseline optimizer, init lerobot/smolvla_base
    "smolvla": ({**COMMON_ENV, "SMOLVLA_FULL": "1"}, {**COMMON_ARGS, "--policy": "smolvla", "--init": "lerobot/smolvla_base"}),
    # ACT: baseline optimizer regime, from scratch (no --init)
    "act": ({**COMMON_ENV, "ACT_BASELINE_OPT": "1"}, {**_without(COMMON_ARGS, "--init"), "--policy": "act"}),
}


@pytest.mark.parametrize("name", REFERENCE)
def test_reference_preset_matches_reference_launch(name: str) -> None:
    env, args = to_legacy(config.preset_flags(name))
    want_env, want_args = EXPECTED[name]
    assert env == want_env, name
    assert args == want_args, name


def test_all_presets_exist_and_are_complete() -> None:
    names = config.preset_names()
    assert names[: len(REFERENCE)] == list(REFERENCE)
    assert set(names) == set(REFERENCE) | set(OPTIONAL)
    defaults = set(config.load_config("train/presets")["defaults"])
    for name in names:
        p = config.preset(name)
        assert p["name"] == name and p["description"]
        assert set(config.preset_flags(name)) == defaults, name
        assert "${" not in str(p), name


def test_reference_presets_carry_reference_rows() -> None:
    rows = {n: config.preset(n)["reference"] for n in REFERENCE}
    assert rows["baseline"] == {"grabbed": 50, "secured": 49, "held": 33, "kept": 23, "timeouts": 0}
    assert rows["no_recovery"] == {"grabbed": 47, "secured": 42, "held": 27, "kept": 17, "timeouts": 3}
    assert rows["no_depth"] == {"grabbed": 50, "secured": 47, "held": 37, "kept": 30, "timeouts": 0}
    assert rows["half_data"] == {"grabbed": 49, "secured": 45, "held": 35, "kept": 16, "timeouts": 1}
    assert rows["smolvla"] == {"grabbed": 50, "secured": 47, "held": 40, "kept": 9, "timeouts": 0}
    assert rows["act"] == {"grabbed": 42, "secured": 27, "held": 19, "kept": 10, "timeouts": 8}
    for name in OPTIONAL:
        assert "reference" not in config.preset(name)


def test_switch_semantics_agree_with_concrete_flags() -> None:
    """The four switches and the flags the trainer actually reads never disagree."""
    for name in config.preset_names():
        f = config.preset_flags(name)
        if f["data"] == "strict_only":
            assert f["sets"] == ["strict"] and f["set_frac"] is None
        else:
            assert f["sets"] == ["strict", "recovery"] and f["set_frac"] == [0.5, 0.5]
        assert (f["ablation"] == "no_plane") == (f["state"] == "qpos")
        assert (f["ablation"] == "no_mask") == (f["input_mode"] == "full_rgbd")
        assert (f["ablation"] == "no_depth") == (f["input_mode"] == "masked_rgb_nodepth")
        assert (f["ablation"] == "no_aug") == f["no_aug"]
        if f["dropout"]:
            assert (f["drop_mask_p"], f["plane_sig_end"], f["plane_glitch_end"], f["plane_rand_frac"]) == (0.25, 8.0, 0.25, 0.5)
        else:
            assert (f["drop_mask_p"], f["plane_sig_end"], f["plane_glitch_end"], f["plane_rand_frac"]) == (0.0, 3.0, 0.03, 0.0)
        assert (f["drop_plane_p"], f["drop_start"], f["plane_sig"], f["plane_glitch_p"]) == (0.0, 0.5, 3.0, 0.03)
        if f["policy"] == "pi0":
            assert str(f["init"]).endswith("/pi0_base") and "/" in str(f["init"])
        if f["policy"] == "act":
            assert f["init"] is None


def test_each_reference_ablation_changes_one_thing() -> None:
    base = config.preset_flags("baseline")
    derived = {"sets", "set_frac", "input_mode", "state", "init", "smolvla_full", "act_opt", "no_aug", "subset_json"}
    switches = {"policy", "data", "ablation", "dropout"}
    for name in REFERENCE[1:]:
        f = config.preset_flags(name)
        changed = {k for k in base if f[k] != base[k]}
        assert len(changed & switches) == 1, (name, changed)
        assert changed <= switches | derived, (name, changed)


def test_unknown_preset_lists_alternatives() -> None:
    with pytest.raises(KeyError, match="baseline"):
        config.preset("nonexistent")
