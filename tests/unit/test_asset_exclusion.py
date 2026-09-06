"""The third-party tool geometry never enters the repository; the tracked tool
files are annotations only.

Checks what ``git add -A`` would commit (tracked plus untracked-but-not-ignored
paths), that the ignore rules cover the drop-in source folder
(``assets/source/<pack>/``), every geometry file the local build produces
and every geometry format dropped anywhere else, that ``meta.json`` carries
no source path, and that the tracked grasp-candidate database holds grasp
records only: centre, rotation, width, standoff, the grasp axis, approach,
quality and the two finger contact points ``p1`` / ``p2`` (sparse,
unconnected surface points; no faces, normals or vertex lists).
"""

from __future__ import annotations

import fnmatch
import json
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
TOOL_DIR = "assets/tools/retractor_rake"

GEOMETRY_GLOBS = (
    "assets/tools/*/visual.obj",
    "assets/tools/*/collision/*",
    "assets/tools/**/*.obj",
    "assets/tools/**/*.fbx",
    "assets/tools/**/*.stl",
    "assets/tools/**/*.glb",
    "assets/tools/**/*.ply",
    "assets/tools/**/*.blend",
    "assets/tools/**/*.max",
    "assets/tools/**/*.usd*",
    "assets/usd/*",
    "assets/source/*/*",
    "*.fbx",
    "*.blend",
    "*.max",
    "*.ply",
    "*.stl",
    "*.glb",
    "*.gltf",
    "*.usd",
    "*.usda",
    "*.usdc",
    "*.usdz",
)

#: Paths the local build writes, plus geometry dropped outside the build
#: directories; every one must be ignored.
BUILD_OUTPUTS = (
    "assets/source/vretractor/vretractor.fbx",
    "assets/source/vretractor/textures/vretractor_metal_Mat_BaseColor.png",
    "assets/source/other_pack/model.obj",
    "assets/source/other_pack/readme.txt",
    f"{TOOL_DIR}/canonical_slots.json",
    f"{TOOL_DIR}/visual.obj",
    f"{TOOL_DIR}/collision/coll_000.obj",
    f"{TOOL_DIR}/textures/albedo.png",
    f"{TOOL_DIR}/source.fbx",
    "assets/usd/retractor_rake.usd",
    "assets/usd/retractor_rake.usda",
    "downloads/pack.fbx",
    "assets/scans/tool.ply",
    "assets/meshes/tool.stl",
    "assets/meshes/tool.obj",
    "assets/retractor.obj",
    "assets/retractor.glb",
    "scene.usdc",
    "export/tool.gltf",
    "simretract/tool.obj",
)

#: Tracked annotations of the tool (no geometry).
TRACKED = (
    f"{TOOL_DIR}/meta.json",
    "assets/source/README.md",
    "configs/assets/retractor_rake.yaml",
    "assets/grasp_candidates/retractor_rake.json",
    "assets/gmask_lut.npz",
    "assets/gripper/finger_left_visual.obj",
    "assets/gripper/finger_right_visual.obj",
    "assets/gripper/collision_left/coll_000.obj",
    "assets/gripper/collision_right/coll_011.obj",
)

POSE_KEYS = {"center", "R", "width", "standoff"}
POSE_EXTRA_KEYS = {"kind", "axis", "approach", "quality", "p1", "p2"}


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout


def test_build_outputs_are_ignored() -> None:
    out = subprocess.run(["git", "-C", str(REPO), "check-ignore", *BUILD_OUTPUTS],
                         capture_output=True, text=True).stdout.split()
    missing = sorted(set(BUILD_OUTPUTS) - set(out))
    assert not missing, f"not ignored by .gitignore: {missing}"


def test_annotations_are_not_ignored() -> None:
    out = subprocess.run(["git", "-C", str(REPO), "check-ignore", *TRACKED],
                         capture_output=True, text=True).stdout.split()
    assert not out, f"tracked annotations would be ignored: {out}"


def test_no_geometry_would_be_committed() -> None:
    paths = [p for p in _git("ls-files", "--cached", "--others", "--exclude-standard", "-z").split("\0") if p]
    assert paths, "git ls-files returned nothing; is this a checkout?"
    hits = [p for p in paths if any(fnmatch.fnmatch(p, g) for g in GEOMETRY_GLOBS)]
    assert not hits, f"tool geometry would be committed: {hits}"


def test_meta_json_is_a_reference_without_provenance_path() -> None:
    meta = json.loads((REPO / TOOL_DIR / "meta.json").read_text())
    assert "source" not in meta
    for key in ("name", "select", "scale", "pre_rotate", "align_z", "coacd_threshold", "extents_m", "longest_m",
                "volume_m3", "n_collision_parts", "collision_hull_volume_m3", "suggested_tip", "suggested_bend"):
        assert key in meta, key
    assert len(meta["extents_m"]) == 3 and len(meta["suggested_tip"]) == 3
    assert "/" not in str(meta["select"]) and "/" not in str(meta["name"])


def _assert_pose_records(records: list[dict], where: str) -> None:
    assert records, where
    for i, rec in enumerate(records):
        keys = set(rec)
        assert POSE_KEYS <= keys, f"{where}[{i}]: keys {sorted(keys)}"
        assert keys <= POSE_KEYS | POSE_EXTRA_KEYS, f"{where}[{i}]: unexpected keys {sorted(keys - POSE_KEYS - POSE_EXTRA_KEYS)}"
        assert len(rec["center"]) == 3 and len(rec["R"]) == 3 and all(len(r) == 3 for r in rec["R"])


def test_tracked_tool_dir_holds_meta_only() -> None:
    """Under assets/tools/<tool>/ the release tracks meta.json and nothing else."""
    paths = [p for p in _git("ls-files", "--cached", "--others", "--exclude-standard", "-z").split("\0") if p]
    tool_files = sorted(p for p in paths if p.startswith("assets/tools/"))
    assert tool_files == [f"{TOOL_DIR}/meta.json"], tool_files
    assert not (REPO / TOOL_DIR / "canonical_slots.json").exists()


def test_tracked_grasp_files_hold_poses_only() -> None:
    db = json.loads((REPO / "assets/grasp_candidates/retractor_rake.json").read_text())
    assert set(db) == {"tool", "flat", "antipodal", "washer"}, sorted(db)
    for kind in ("flat", "antipodal", "washer"):
        _assert_pose_records(db[kind], kind)
