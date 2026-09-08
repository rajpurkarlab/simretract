"""Exercise setup's final validation without installing or importing ML packages."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize("failed_roles", [[], ["sim", "policy"]])
def test_setup_reports_all_import_failures(tmp_path, failed_roles):
    repo = tmp_path / "checkout"
    (repo / "envs").mkdir(parents=True)
    (repo / "configs").mkdir()
    setup = repo / "envs" / "setup.sh"
    source = Path(__file__).resolve().parents[2] / "envs" / "setup.sh"
    shutil.copyfile(source, setup)
    fake_python = tmp_path / "fake-python"
    # Pip is a no-op. Run the actual embedded validation with stubbed imports
    # so the test covers both the Python exit status and the shell aggregation.
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import importlib, importlib.metadata, os, sys\n"
        "if sys.argv[1:3] == ['-m', 'pip']:\n"
        "    sys.exit(0)\n"
        "assert sys.argv[1] == '-', sys.argv\n"
        "role = sys.argv[2]\n"
        "def fake_import(name):\n"
        "    if role in os.environ['SETUP_TEST_FAILED_ROLES'].split(','):\n"
        "        raise ImportError('synthetic missing dependency for ' + role)\n"
        "    return object()\n"
        "importlib.import_module = fake_import\n"
        "importlib.metadata.version = lambda name: 'test'\n"
        "sys.argv = sys.argv[1:]\n"
        "exec(compile(sys.stdin.read(), '<setup-validation>', 'exec'))\n"
    )
    fake_python.chmod(0o755)
    result = subprocess.run(
        ["bash", str(setup), "--reuse",
         *(f"{role}={fake_python}" for role in ("sim", "policy", "mask"))],
        env={**os.environ, "SETUP_TEST_FAILED_ROLES": ",".join(failed_roles)},
        capture_output=True, text=True, timeout=15,
    )
    output = result.stdout + result.stderr
    # A failure in an earlier role must not hide validation of later roles.
    for role, marker in (("sim", "isaacsim"), ("policy", "lerobot"), ("mask", "ultralytics")):
        expected = "FAILED" if role in failed_roles else "ok"
        assert f"import {marker}: {expected}" in output, output
    if failed_roles:
        assert result.returncode != 0, output
        assert "environment validation failed for: sim policy" in result.stderr
        assert "== done." not in output
    else:
        assert result.returncode == 0, output
        assert "== done." in result.stdout
        assert "environment validation failed" not in output
