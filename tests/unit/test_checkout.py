"""The checkout must work without borrowing missing modules from another install."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tomllib


REPO = Path(__file__).resolve().parents[2]


def test_cli_targets_and_evaluation_are_in_this_checkout():
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    modules = {entry.split(":")[0] for entry in project["scripts"].values()}
    modules.update(("simretract.eval.driver", "simretract.eval.score", "simretract.eval.sim_eval"))
    for name in modules:
        spec = importlib.util.find_spec(name)
        assert spec is not None and Path(spec.origin).resolve().is_relative_to(REPO), name
    assert (REPO / "configs/eval/sim.yaml").is_file()
    result = subprocess.run(["git", "check-ignore", "simretract/eval/driver.py",
                             "configs/eval/sim.yaml"], cwd=REPO, capture_output=True)
    assert result.returncode == 1, result.stdout.decode()


def test_storage_override_does_not_relocate_wrapper_code(tmp_path):
    env = dict(os.environ, SIMRETRACT_ROOT=str(tmp_path), SIMRETRACT_POLICY_PYTHON=sys.executable)
    env.pop("SIMRETRACT_REPO", None)
    proc = subprocess.run(["bash", str(REPO / "scripts/run_real.sh"), "weights", "--help"],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0, proc.stderr
    assert "simretract-run-real" in proc.stdout
