"""Check shipped files for machine-specific paths and connection identities.

Scan first-party tracked and non-ignored files. Binary files are checked for
printable paths; embedded metadata and visible media require separate review.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]

RULES = {
    "user directory": re.compile(r"/(?:home|Users)/[^/\s\"']+"),
    "Windows user directory": re.compile(
        r"[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/][^\\/\s\"']+"
    ),
    "machine directory": re.compile(r"/(?:mnt|workspace|scratch)(?:/|\b)"),
    "email address": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "SSH destination": re.compile(r"\bssh\s+(?:-\S+\s+)*[^\s\"']+@[^\s\"']+"),
}

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


def find_leaks(text: str, *, binary: bool = False) -> list[str]:
    hits = []
    for label, pattern in RULES.items():
        if binary and not label.endswith("directory"):
            continue  # Compressed bytes can resemble short email addresses.
        match = pattern.search(text)
        if match:
            context = text[max(0, match.start() - 30):match.end() + 30]
            hits.append(f"{label} near: {context!r}")
    return hits


@pytest.mark.parametrize("sample", [
    "/" + "/".join(("home", "example", "project")),
    "/" + "/".join(("Users", "example", "project")),
    "C:" + chr(92) + chr(92).join(("Users", "example", "project")),
    "D:" + "/".join(("", "Users", "example", "project")),
    *["/" + "/".join((root, "project")) for root in ("mnt", "workspace", "scratch")],
    "example" + "@" + "example.invalid",
    "ssh " + "example" + "@" + "example-host",
])
def test_detector_catches_machine_identifiers(sample: str) -> None:
    assert find_leaks(sample)


def test_detector_allows_portable_references() -> None:
    assert not find_leaks("assets/gripper/LICENSE 127.0.0.1 configs/real/default.yaml")
    assert not find_leaks("https://github.com/Source-Robotics/SSG-48-adaptive-electric-gripper")


def test_binary_scan_checks_paths_without_treating_encoded_bytes_as_email() -> None:
    encoded_text = "3" + "@" + "HHu.jB"
    assert not find_leaks(encoded_text, binary=True)
    path = "/" + "/".join(("home", "example", "project"))
    assert find_leaks(path, binary=True)


def test_repository_has_files() -> None:
    files = repo_files()
    assert files, "git ls-files returned nothing; is this a checkout?"


@pytest.mark.parametrize("path", repo_files(), ids=lambda p: str(p.relative_to(REPO)))
def test_no_leak_strings(path: Path) -> None:
    binary = path.suffix.lower() in BINARY_SUFFIXES - {".obj", ".mtl", ".usda"}
    hits = find_leaks(text_of(path), binary=binary)
    assert not hits, f"{path.relative_to(REPO)}: " + "; ".join(hits)


def test_no_ascii_violations_in_code_and_configs() -> None:
    """Python, YAML and shell files stay ASCII so they render everywhere."""
    bad = []
    for p in repo_files():
        if p.suffix in {".py", ".yaml", ".yml", ".sh", ".toml", ".txt", ".cff"}:
            if any(ord(ch) > 127 for ch in p.read_text(encoding="utf-8", errors="replace")):
                bad.append(str(p.relative_to(REPO)))
    assert not bad, bad
