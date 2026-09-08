"""Real-robot deployment of a SimRetract policy.

Three localhost processes cooperate, all speaking the length-prefixed pickle
protocol of :mod:`simretract.serve.protocol`:

* the policy server (:mod:`simretract.serve.policy_server`), which owns the
  checkpoint and fits the tool plane from each query's depth + mask;
* the SAM 3 mask server (:mod:`simretract.real.mask_server`), which turns the
  wrist RGB stream into a single-object tool mask;
* the client (:mod:`simretract.real.client`), which captures the RealSense
  D405, drives the Trossen WidowX-AI follower and records every run.

This module holds the small, dependency-free helpers shared by the client,
the launcher and the utilities: loading ``configs/real/default.yaml`` and
turning a named preset into client flags.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

FOLLOWER_IP_PLACEHOLDER = "CHANGE_ME"
"""Value shipped in ``robot.follower_ip``; tools refuse to run while it is set."""

PRESET_KEYS = ("exec_h", "exec_min", "settle", "speed", "max_step", "smooth",
               "view", "plane_top_cut", "plane_top_cut_m")
"""Preset keys, each mapping onto the client flag of the same name."""


def repo_root() -> Path:
    """Repository root as configured by :mod:`simretract.config`.

    Falls back to the package location (the editable install layout) when the
    config module is unavailable, so ``--help`` and unit tests work anywhere.
    """
    try:
        from simretract import config
        return Path(config.repo_root())
    except Exception:  # config not importable yet (partial install)
        return Path(__file__).resolve().parents[2]


def default_runs_root() -> Path:
    """Root for real-run recordings: ``<eval>/real`` from :mod:`simretract.config`."""
    try:
        from simretract import config
        return Path(config.paths()["eval"]) / "real"
    except Exception:
        return repo_root() / "eval" / "real"


def default_config_path() -> Path:
    """Location of the shipped real-run configuration file."""
    return repo_root() / "configs" / "real" / "default.yaml"


def load_real_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load the real-run YAML config (``configs/real/default.yaml`` by default).

    Relative paths inside the ``mask`` section are resolved against the
    repository root so that the file can be used from any working directory.
    """
    p = Path(path) if path is not None else default_config_path()
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("robot", {})
    cfg.setdefault("services", {})
    cfg.setdefault("mask", {})
    cfg.setdefault("presets", {})
    ck = cfg["mask"].get("sam3_ckpt")
    if ck and not os.path.isabs(str(ck)):
        cfg["mask"]["sam3_ckpt"] = str(repo_root() / str(ck))
    return cfg


def follower_ip_is_set(ip: str | None) -> bool:
    """True when ``ip`` is a usable robot address (non-empty, not the placeholder)."""
    return bool(ip) and str(ip).strip().upper() != FOLLOWER_IP_PLACEHOLDER


def resolve_preset(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    """Return the preset ``name`` from ``cfg`` with every key of PRESET_KEYS present."""
    presets = cfg.get("presets") or {}
    if name not in presets:
        raise KeyError(f"unknown real-run preset {name!r}; "
                       f"available: {', '.join(sorted(presets)) or 'none'}")
    p = dict(presets[name])
    p.setdefault("plane_top_cut_m", 0.0)  # Older custom presets keep legacy behavior.
    missing = [k for k in PRESET_KEYS if k not in p]
    if missing:
        raise KeyError(f"preset {name!r} lacks keys {missing}")
    return p


def preset_client_args(preset: dict[str, Any]) -> list[str]:
    """Translate a preset dict into the matching client command-line flags."""
    args = [
        "--exec-h", str(int(preset["exec_h"])),
        "--exec-min", str(int(preset["exec_min"])),
        "--settle", str(float(preset["settle"])),
        "--speed", str(float(preset["speed"])),
        "--max-step", str(float(preset["max_step"])),
        "--smooth", str(float(preset["smooth"])),
        "--plane-top-cut", str(float(preset["plane_top_cut"])),
        "--plane-top-cut-m", str(float(preset.get("plane_top_cut_m", 0.0))),
    ]
    args.append("--view" if bool(preset["view"]) else "--no-view")
    return args
