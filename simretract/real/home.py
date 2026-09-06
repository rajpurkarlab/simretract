"""Bring the follower arm back to the simulation home pose.

All six joints go to zero with the same slow micro-step ramp the client
uses; the gripper opening is left untouched. Connecting clears a controller
error; after a CAN feedback-loss fault, power-cycle the controller first if
the connect itself still raises.

    python -m simretract.real.home [--follower-ip A.B.C.D] [--yes]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np

from simretract.real import (FOLLOWER_IP_PLACEHOLDER, follower_ip_is_set,
                             load_real_config)

log = logging.getLogger("simretract.real.home")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m simretract.real.home",
        description="Ramp the follower arm to the simulation home pose (joints 0, gripper unchanged).")
    ap.add_argument("--follower-ip", dest="follower_ip", default=None,
                    help="follower controller IP; default from configs/real/default.yaml")
    ap.add_argument("--config", default=None, help="real-run YAML (default configs/real/default.yaml)")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    return ap


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    a = build_parser().parse_args(argv)
    model = "wxai_v0"
    if a.follower_ip is None:
        cfg = load_real_config(a.config)
        a.follower_ip = (cfg.get("robot") or {}).get("follower_ip")
        model = (cfg.get("robot") or {}).get("model", model)
    if not follower_ip_is_set(a.follower_ip):
        raise SystemExit(f"robot.follower_ip is {FOLLOWER_IP_PLACEHOLDER!r}: set it in "
                         "configs/real/default.yaml or pass --follower-ip")
    from simretract.real.client import Arm
    arm = Arm(a.follower_ip, model)
    cur = arm.read()
    log.info("present: %s", np.round(cur, 3))
    home = cur.copy()
    home[:6] = 0.0
    n = max(1, int(np.ceil(np.abs(home[:6] - cur[:6]).max() / 0.02)))
    log.info("ramp to home in %d micro-steps (~%.1fs); gripper stays at %.3f", n, 0.1 * n, cur[6])
    if not a.yes:
        if input("type go to move: ").strip() != "go":
            arm.cleanup()
            sys.exit("aborted")
    arm.enable_position_mode()
    try:
        for k in range(1, n + 1):
            arm.goto(cur + (home - cur) * k / n, 0.1)
        time.sleep(0.3)
        log.info("at home: %s", np.round(arm.read(), 3))
    finally:
        arm.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
