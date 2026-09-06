"""No private infrastructure, personal paths, hosts or dated notes ship in the repository.

Every git-tracked and untracked-but-not-ignored file is scanned.  Text files
are scanned as-is; binary assets (.npz, .usd, .obj, .png, .npy, ...) are reduced
to their printable runs first, the way ``strings`` does.  The patterns are
assembled from fragments so this file does not trip its own check.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]

# (pattern, case-insensitive)
PATTERNS: list[tuple[str, bool]] = [
    ("/ho" + "me/", False),
    ("/m" + "nt/", False),
    ("/work" + "space", False),
    ("/scra" + "tch", False),
    ("ddd" + "w123", False),
    ("Drop" + "box", False),
    ("inspect_" + "data", False),
    ("192." + "168.", False),
    ("ss" + "h ", False),
    ("rsy" + "nc", False),
    ("neb" + "ius", True),
    ("run" + "pod", False),
    ("66." + "201.", False),
    ("89." + "169.", False),
    ("david" + "wang", False),
    ("ausc_" + "live", False),
    ("omni-retract-" + "issac", False),
    ("owner " + "20", False),
    ("20" + "26-", False),
    ("1038" + "lab", False),
    ("cg" + "trader", True),
    ("so7" + "si", True),
    ("5268" + "240", True),
    ("purch" + "as", True),
]

BINARY_SUFFIXES = {".npz", ".npy", ".usd", ".usda", ".usdc", ".obj", ".mtl", ".png", ".jpg", ".jpeg",
                   ".pt", ".safetensors", ".mp4", ".tar", ".gz", ".zip", ".pdf", ".woff", ".woff2"}
PRINTABLE_RUN = re.compile(rb"[\x20-\x7e]{6,}")


def repo_files() -> list[Path]:
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        capture_output=True,
        check=True,
    ).stdout
    files = []
    for rel in out.decode("utf-8", "surrogateescape").split("\0"):
        if not rel:
            continue
        p = REPO / rel
        if rel.startswith("third_party/"):
            continue  # submodule content is not ours to scan
        if p.is_file():
            files.append(p)
    return files


def text_of(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix.lower() in BINARY_SUFFIXES:
        return "\n".join(m.group(0).decode("ascii", "replace") for m in PRINTABLE_RUN.finditer(data))
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return "\n".join(m.group(0).decode("ascii", "replace") for m in PRINTABLE_RUN.finditer(data))


def find_leaks(text: str) -> list[str]:
    hits = []
    lowered = text.lower()
    for pat, ci in PATTERNS:
        hay = lowered if ci else text
        needle = pat.lower() if ci else pat
        idx = hay.find(needle)
        if idx >= 0:
            start = max(0, idx - 30)
            hits.append(f"{pat!r} near: {text[start: idx + len(pat) + 30]!r}")
    return hits


def test_detector_catches_every_pattern() -> None:
    """Positive control: each pattern is found when present, in either case for the case-insensitive one."""
    for pat, ci in PATTERNS:
        assert find_leaks("prefix " + pat + " suffix"), pat
        if ci:
            assert find_leaks("prefix " + pat.upper() + " suffix"), pat
    assert not find_leaks("a clean line with seeds 202606 and 400851")


def test_repository_has_files() -> None:
    files = repo_files()
    assert files, "git ls-files returned nothing; is this a checkout?"


@pytest.mark.parametrize("path", repo_files(), ids=lambda p: str(p.relative_to(REPO)))
def test_no_leak_strings(path: Path) -> None:
    hits = find_leaks(text_of(path))
    assert not hits, f"{path.relative_to(REPO)}: " + "; ".join(hits)


def test_no_ascii_violations_in_code_and_configs() -> None:
    """Python, YAML and shell files stay ASCII so they render everywhere."""
    bad = []
    for p in repo_files():
        if p.suffix in {".py", ".yaml", ".yml", ".sh", ".toml", ".txt", ".cff"}:
            if any(ord(ch) > 127 for ch in p.read_text(encoding="utf-8", errors="replace")):
                bad.append(str(p.relative_to(REPO)))
    assert not bad, bad
