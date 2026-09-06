"""Shared pytest configuration.

Modules that create an Isaac Sim ``SimulationApp`` (or import ``omni``) cannot
be collected outside the sim environment, so they are excluded from collection
here; the ``sim`` tier runs them as subprocesses through ``simretract-smoke``.

Every test carries the marker of its tier directory (``tests/unit`` ->
``unit``, ``tests/gpu`` -> ``gpu``, ``tests/sim`` -> ``sim``,
``tests/real_dry`` -> ``real_dry``, ``tests/release`` -> ``release``) so
``pytest -m <tier>`` selects a whole tier without per-module boilerplate.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Kit-side modules: never imported by the unit / gpu tiers.
collect_ignore = [
    "../simretract/sim/scene.py",
    "../simretract/sim/battery.py",
    "../simretract/sim/review.py",
    "../simretract/sim/recorder.py",
    "../simretract/sim/recovery_recorder.py",
    "../simretract/sim/tool_usd.py",
    "../simretract/real/gmask_lut.py",
    "../simretract/eval/sim_eval.py",
]


TIER_MARKERS = {"unit": "unit", "gpu": "gpu", "sim": "sim", "real_dry": "real_dry", "release": "release"}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark each test with its tier (the directory under ``tests/``)."""
    tests_dir = REPO / "tests"
    for item in items:
        try:
            rel = Path(str(item.fspath)).resolve().relative_to(tests_dir)
        except ValueError:
            continue
        marker = TIER_MARKERS.get(rel.parts[0]) if rel.parts else None
        if marker and not item.get_closest_marker(marker):
            item.add_marker(getattr(pytest.mark, marker))


@pytest.fixture
def repo_root() -> Path:
    return REPO


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Remove every SIMRETRACT_* override so tests see the tracked configuration."""
    for key in list(os.environ):
        if key.startswith("SIMRETRACT_"):
            monkeypatch.delenv(key, raising=False)
    return monkeypatch
