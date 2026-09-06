"""Named training rows (presets) and their resolution to trainer flags.

The canonical table is ``configs/train/presets.yaml`` (read through
``simretract.config.preset_flags``): ``defaults`` is the complete flag set of
the baseline row and every preset lists what it changes.  ``BUILTIN_PRESETS``
mirrors the four high-level switches of each row so that ``--help`` and the
unit tests work in a checkout whose configs are unavailable; the unit tests
assert the two agree whenever the yaml can be loaded.

A preset installs parser *defaults*: flags given explicitly on the command
line still win.  ``resolve(name)`` returns the fully finalized flag set
(after the ablation / data / dropout mapping of ``train.finalize_args``).
"""
from __future__ import annotations

import copy
from typing import Any

#: The high-level switches every preset carries and their baseline values.
PRESET_SWITCHES: dict[str, Any] = {
    "policy": "pi0",
    "data": "full",
    "ablation": "none",
    "dropout": True,
}

#: Switch-level view of the rows in configs/train/presets.yaml.
BUILTIN_PRESETS: dict[str, dict[str, Any]] = {
    # --- reference rows (docs/ABLATIONS.md) --------------------------------
    "baseline": {"policy": "pi0", "data": "full", "ablation": "none",
                 "dropout": True},
    "no_recovery": {"policy": "pi0", "data": "strict_only", "ablation": "none",
                    "dropout": True},
    "no_depth": {"policy": "pi0", "data": "full", "ablation": "no_depth",
                 "dropout": True},
    "half_data": {"policy": "pi0", "data": "half", "ablation": "none",
                  "dropout": True},
    "smolvla": {"policy": "smolvla", "data": "full", "ablation": "none",
                "dropout": True, "smolvla_full": True},
    "act": {"policy": "act", "data": "full", "ablation": "none",
            "dropout": True, "act_opt": "baseline"},
    # --- optional rows (no reference result; trained without dropout) ------
    "no_plane": {"policy": "pi0", "data": "full", "ablation": "no_plane",
                 "dropout": False},
    "no_mask": {"policy": "pi0", "data": "full", "ablation": "no_mask",
                "dropout": False},
    "no_aug": {"policy": "pi0", "data": "full", "ablation": "no_aug",
               "dropout": False},
    "fastwam": {"policy": "fastwam", "data": "full", "ablation": "none",
                "dropout": False, "chunk_h": 48, "video_t": 9, "video_ratio": 6},
    "pi05": {"policy": "pi05", "data": "full", "ablation": "none",
             "dropout": False},
    "pi0_fast": {"policy": "pi0_fast", "data": "full", "ablation": "none",
                 "dropout": False},
}

#: Rows with a reference result table, in table order.
REFERENCE_ROWS: tuple[str, ...] = ("baseline", "no_recovery", "no_depth",
                               "half_data", "smolvla", "act")


def _as_parser_defaults(flags: dict[str, Any]) -> dict[str, Any]:
    """Convert yaml-typed preset values to the parser's string conventions
    (``sets``/``set_frac`` lists -> comma strings, ``set_frac: null`` ->
    ``"none"`` = frame-proportional sampling)."""
    out = dict(flags)
    if isinstance(out.get("sets"), (list, tuple)):
        out["sets"] = ",".join(str(s) for s in out["sets"])
    if "set_frac" in out:
        sf = out["set_frac"]
        if sf is None:
            out["set_frac"] = "none"
        elif isinstance(sf, (list, tuple)):
            out["set_frac"] = ",".join(f"{float(x):g}" for x in sf)
    return out


def _yaml_presets() -> dict[str, dict[str, Any]] | None:
    """The preset table from ``configs/train/presets.yaml`` via
    ``simretract.config``, or ``None`` when it cannot be loaded."""
    try:
        from simretract import config as srconfig
        names = srconfig.preset_names()
        table = {n: _as_parser_defaults(srconfig.preset_flags(n)) for n in names}
    except Exception:  # noqa: BLE001 - any config problem -> builtin table
        return None
    return table or None


def load_presets(use_config: bool = True) -> dict[str, dict[str, Any]]:
    """Return ``{name: parser defaults}`` for every known preset (yaml table
    when loadable, else the builtin switch-level table)."""
    table = _yaml_presets() if use_config else None
    if table:
        return table
    return {n: {**PRESET_SWITCHES, **v} for n, v in copy.deepcopy(BUILTIN_PRESETS).items()}


def preset_names(use_config: bool = True) -> list[str]:
    """Names accepted by ``--preset``."""
    return list(load_presets(use_config))


def preset_flags(name: str, use_config: bool = True) -> dict[str, Any]:
    """Parser defaults a preset installs."""
    table = load_presets(use_config)
    if name not in table:
        raise KeyError(f"unknown preset {name!r}; known: {', '.join(table)}")
    return dict(table[name])


def resolve(name: str, extra_argv: list[str] | None = None,
            use_config: bool = True) -> dict[str, Any]:
    """Fully resolved trainer flags for a preset (what ``simretract-train
    --preset NAME`` runs with), as a plain dict for tests and docs."""
    from simretract.train import train as _train  # local: keeps import light

    args = _train.parse_args(["--preset", name, *(extra_argv or [])],
                             use_config=use_config)
    return vars(args)
