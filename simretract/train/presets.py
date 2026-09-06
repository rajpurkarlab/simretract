"""Named training rows (presets) and their resolution to trainer flags.

The canonical table is ``configs/train/presets.yaml`` (read through
``simretract.config.preset_flags``): ``defaults`` is the complete flag set of
the baseline row and every preset lists what it changes.  ``BUILTIN_PRESETS``
mirrors the switches of each row so that ``--help`` and the unit tests work
in a checkout whose configs are unavailable; the unit tests assert the two
agree whenever the yaml can be loaded.

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

#: Switch-level view of the rows in configs/train/presets.yaml, in table order.
#: Every row has the dropout rule on, except ``no_dropout``; every row is pi0,
#: except ``smolvla`` (the SmolVLA backbone on the baseline flags and data); the
#: ``no_smooth`` / ``no_standoff`` rows keep the baseline flags and train on
#: the no-smoothing / no-standoff cache pairs (configs/data.yaml).
BUILTIN_PRESETS: dict[str, dict[str, Any]] = {
    "baseline": {"policy": "pi0", "data": "full", "ablation": "none",
                 "dropout": True},
    "no_dropout": {"policy": "pi0", "data": "full", "ablation": "none",
                   "dropout": False},
    "no_recovery": {"policy": "pi0", "data": "strict_only", "ablation": "none",
                    "dropout": True},
    "half_data": {"policy": "pi0", "data": "half", "ablation": "none",
                  "dropout": True},
    "no_plane": {"policy": "pi0", "data": "full", "ablation": "no_plane",
                 "dropout": True},
    "no_mask": {"policy": "pi0", "data": "full", "ablation": "no_mask",
                "dropout": True},
    "no_smooth": {"policy": "pi0", "data": "full", "ablation": "none",
                  "dropout": True, "sets": "strict_nosmooth,recovery_nosmooth"},
    "no_standoff": {"policy": "pi0", "data": "full", "ablation": "none",
                    "dropout": True, "sets": "strict_nostandoff,recovery_nostandoff"},
    "smolvla": {"policy": "smolvla", "data": "full", "ablation": "none",
                "dropout": True, "smolvla_full": True},
}

#: Preset names in table order (the order of configs/train/presets.yaml).
PRESET_ORDER: tuple[str, ...] = tuple(BUILTIN_PRESETS)


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
