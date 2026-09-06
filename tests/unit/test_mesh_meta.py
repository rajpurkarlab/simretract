"""Reference checks of the local asset build against the tracked ``meta.json``.

``mesh_convert.compare_meta`` (after the conversion) and
``tool_usd.check_against_meta`` (after the USD build) are pure functions;
the tolerances are 0.1 mm on extents, 1% on volume, and for the collision
set (``tool_usd.check_parts``) 3 parts and 5% of the summed hull volume,
because CoACD splits the same shape differently for sub-micrometre
coordinate differences and from run to run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simretract.sim import mesh_convert, tool_usd

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def reference() -> dict:
    return json.loads((REPO / "assets/tools/retractor_rake/meta.json").read_text())


def test_identical_build_matches(reference: dict) -> None:
    assert mesh_convert.compare_meta(dict(reference), reference) == []


def test_rounding_noise_is_tolerated(reference: dict) -> None:
    new = dict(reference)
    new["extents_m"] = [x + 5e-5 for x in reference["extents_m"]]
    new["longest_m"] = reference["longest_m"] - 5e-5
    new["volume_m3"] = reference["volume_m3"] * 1.005
    assert mesh_convert.compare_meta(new, reference) == []


def test_each_reference_key_is_checked(reference: dict) -> None:
    for key, value, word in (("extents_m", [0.1, 0.1, 0.1], "extents_m"), ("longest_m", 0.3, "longest_m"),
                             ("volume_m3", reference["volume_m3"] * 1.1, "volume_m3"),
                             ("n_collision_parts", 3, "collision parts"),
                             ("collision_hull_volume_m3", reference["collision_hull_volume_m3"] * 1.2, "hull volume")):
        new = dict(reference)
        new[key] = value
        problems = mesh_convert.compare_meta(new, reference)
        assert len(problems) == 1 and word in problems[0], (key, problems)


def test_collision_set_tolerance(reference: dict) -> None:
    """The reference mesh splits into 20 or 21 parts with a hull volume
    within 2% from run to run on the reference machine; both pass."""
    n, hull = reference["n_collision_parts"], reference["collision_hull_volume_m3"]
    for dn, dv in ((0, 0.0), (-1, 0.02), (1, -0.02), (3, 0.049), (-3, -0.049)):
        new = dict(reference, n_collision_parts=n + dn, collision_hull_volume_m3=hull * (1 + dv))
        assert mesh_convert.compare_meta(new, reference) == [], (dn, dv)
        assert tool_usd.check_parts(n + dn, hull * (1 + dv), reference) is None
    assert "collision parts" in tool_usd.check_parts(n + 4, hull, reference)
    assert "hull volume" in tool_usd.check_parts(n, hull * 1.06, reference)
    assert tool_usd.check_parts(n, None, reference) is None            # no hull volume: count only
    assert tool_usd.check_parts(n, hull, {"n_collision_parts": n}) is None


def test_parts_check_skipped_without_coacd(reference: dict) -> None:
    new = dict(reference)
    new["n_collision_parts"] = 0
    assert mesh_convert.compare_meta(new, reference, check_parts_set=False) == []
    assert mesh_convert.compare_meta(new, reference, check_parts_set=True)


def test_watertightness_change_is_reported(reference: dict) -> None:
    new = dict(reference)
    new["volume_m3"] = None
    assert any("volume" in p for p in mesh_convert.compare_meta(new, reference))


def test_usd_summary_check(reference: dict) -> None:
    ok = {"extents_m": reference["extents_m"], "mesh_volume_m3": reference["volume_m3"],
          "n_parts": reference["n_collision_parts"], "hull_volume_m3": reference["collision_hull_volume_m3"]}
    assert tool_usd.check_against_meta(ok, reference) == []
    assert tool_usd.check_against_meta(dict(ok, n_parts=reference["n_collision_parts"] - 1), reference) == []
    bad = dict(ok, n_parts=reference["n_collision_parts"] + 5)
    assert tool_usd.check_against_meta(bad, reference)
    bad = dict(ok, hull_volume_m3=reference["collision_hull_volume_m3"] * 1.1)
    assert tool_usd.check_against_meta(bad, reference)
    bad = dict(ok, extents_m=[x + 1e-3 for x in reference["extents_m"]])
    assert tool_usd.check_against_meta(bad, reference)


def test_reference_meta_describes_the_recipe_build(reference: dict) -> None:
    """meta.json is the build of the recipe source with the recipe settings."""
    assert reference["select"] == "ovrsi_retractor_body"
    assert reference["scale"] == 0.01474075
    assert reference["pre_rotate"] == [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
    assert reference["align_z"] is True
    assert reference["coacd_threshold"] == 0.015
    assert reference["n_collision_parts"] == 21
    assert reference["collision_hull_volume_m3"] == 7.848e-06
    assert "source" not in reference


def test_setting_resolution_order() -> None:
    recipe = {"scale": 0.01474075, "pre_rotate": [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]}
    ref = {"scale": 0.01, "align_z": True, "pre_rotate": None}
    layers = [("recipe", recipe), ("meta", ref)]
    assert mesh_convert._setting(None, layers, "scale", 0.001) == (0.01474075, "recipe")
    assert mesh_convert._setting(0.5, layers, "scale", 0.001) == (0.5, "command line")
    assert mesh_convert._setting(None, [("recipe", None), ("meta", None)], "scale", 0.001) == (0.001, "default")
    assert mesh_convert._setting(None, layers, "align_z", False) == (True, "meta")
    assert mesh_convert._setting(False, layers, "align_z", False) == (False, "command line")
    assert mesh_convert._setting(None, layers, "pre_rotate", None)[1] == "recipe"
    assert mesh_convert._setting(None, [("meta", ref)], "pre_rotate", None) == (None, "default")
