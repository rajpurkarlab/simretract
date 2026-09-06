"""Real-robot dry-run tier (hardware optional).

Skips unless the SAM 3 weights, a robot address and a RealSense D405 are
present; then runs ``simretract-run-real ... dry`` (perception + inference,
the arm is read-only) for a few ticks with the ACT smoke checkpoint.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from simretract.cli import smoke

pytestmark = pytest.mark.real_dry


def _skip_unless_hardware() -> tuple[Path, str]:
    from simretract import config

    cfg = config.load_config("real/default")
    ip = str((cfg.get("robot") or {}).get("follower_ip", "CHANGE_ME"))
    if ip in {"", "CHANGE_ME"} and not os.environ.get("SIMRETRACT_FOLLOWER_IP"):
        pytest.skip("robot.follower_ip is not set")
    sam3 = Path(str((cfg.get("mask") or {}).get("sam3_ckpt", "")))
    if not sam3.is_absolute():
        sam3 = config.repo_root() / sam3
    if not sam3.is_file():
        pytest.skip(f"SAM 3 weights not found at {sam3}")
    rs = pytest.importorskip("pyrealsense2")
    if not list(rs.context().query_devices()):
        pytest.skip("no RealSense device connected")
    return sam3, os.environ.get("SIMRETRACT_FOLLOWER_IP", ip)


def test_client_dry_run() -> None:
    sam3, ip = _skip_unless_hardware()
    work = smoke.work_dir()
    log = work / "logs" / "real_dry.log"
    log.unlink(missing_ok=True)
    ckpt = smoke.ensure_act_smoke_checkpoint(work, work / "logs" / "act_smoke.log")
    rc = smoke.run_logged([sys.executable, "-m", "simretract.cli.run_real", "--ckpt", ckpt,
                           "--follower-ip", ip, "--sam3-ckpt", sam3, "dry",
                           "--max-ticks", "30", "--no-view", "--yes"], log, timeout=1200)
    assert rc == 0, smoke.tail(log)
