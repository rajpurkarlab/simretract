"""Release gates: objective checks that must hold before a public push.

This tier fails until the release is prepared; it is not part of
``scripts/smoke_test.sh all``. Each test names one gate and the file that
closes it: the code LICENSE, the license fields and terms, the hub
organisation and checksums, no tool geometry tracked, and the release
statement in the README.
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.release

REPO = Path(__file__).resolve().parents[2]

#: The release statement; it appears verbatim in README.md.
RELEASE_WORDING = (
    "We release our code, policy checkpoints, and robot-action/grasp-point datasets. "
    "The third-party retractor asset by HQ3DMOD must be obtained separately under its "
    "publisher's license. Asset acquisition and local preparation instructions are provided "
    "in this repository."
)

#: Files that would carry the third-party tool geometry; none may be tracked.
FORBIDDEN_GLOBS = (
    "assets/tools/*/visual.obj",
    "assets/tools/*/collision/*",
    "assets/tools/**/*.obj",
    "assets/tools/**/*.fbx",
    "assets/tools/**/*.stl",
    "assets/tools/**/*.glb",
    "assets/tools/**/*.gltf",
    "assets/tools/**/*.ply",
    "assets/tools/**/*.blend",
    "assets/tools/**/*.max",
    "assets/tools/**/*.usd*",
    "assets/tools/**/textures/*",
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


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _would_be_committed() -> list[str]:
    """Tracked plus untracked-but-not-ignored paths: what `git add -A` would commit."""
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        capture_output=True, check=True).stdout
    return [rel for rel in out.decode("utf-8", "surrogateescape").split("\0") if rel]


def test_license_file_present_and_placeholder_removed() -> None:
    assert (REPO / "LICENSE").is_file(), "add the code LICENSE file (see LICENSING.md)"
    body = _read("LICENSE")
    assert "TO BE CHOSEN" not in body.upper() and "PLACEHOLDER" not in body.upper(), "LICENSE is a placeholder"


def test_licensing_documents_present() -> None:
    for name in ("LICENSING.md", "LICENSE-CHECKPOINTS.md", "LICENSE-DATA.md"):
        assert (REPO / name).is_file(), f"{name} is missing"
    readme = _read("README.md")
    assert "LICENSING.md" in readme, "README does not point at LICENSING.md"


def test_release_terms_chosen() -> None:
    for name in ("LICENSING.md", "LICENSE-CHECKPOINTS.md", "LICENSE-DATA.md"):
        assert "to be chosen" not in _read(name).lower(), f"{name}: the terms are still to be chosen"


def test_license_fields_filled_in() -> None:
    py = _read("pyproject.toml")
    assert re.search(r"^license\s*=", py, re.M), "pyproject.toml has no `license` entry"
    cff = yaml.safe_load(_read("CITATION.cff"))
    assert cff.get("license") and "SEE LICENSE" not in str(cff["license"]).upper(), "CITATION.cff license is a placeholder"
    assert "SURNAME" not in _read("CITATION.cff"), "CITATION.cff authors are placeholders"


def test_readme_carries_release_wording() -> None:
    assert RELEASE_WORDING in _read("README.md"), "README.md lacks the verbatim release statement"


def test_no_tool_geometry_tracked() -> None:
    """The third-party retractor mesh, its collision parts and the USD are built locally, never committed."""
    hits = [rel for rel in _would_be_committed()
            if any(fnmatch.fnmatch(rel, g) for g in FORBIDDEN_GLOBS)]
    assert not hits, f"tool geometry would be committed: {hits}"
    meta = yaml.safe_load(_read("assets/tools/retractor_rake/meta.json"))
    assert "source" not in meta, "meta.json must not carry the source path"
    under_source = [rel for rel in _would_be_committed() if rel.startswith("assets/source/")]
    assert under_source == ["assets/source/README.md"], f"assets/source/ must hold only its README: {under_source}"


def test_hub_configured_with_checksums() -> None:
    hub = yaml.safe_load(_read("configs/hub.yaml"))
    assert hub.get("org") not in (None, "", "YOUR_HF_ORG"), "configs/hub.yaml: org is the placeholder"
    for name, entry in (hub.get("checkpoints") or {}).items():
        assert entry.get("sha256"), f"configs/hub.yaml: checkpoints.{name}.sha256 is empty"
