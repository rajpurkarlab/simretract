"""Live SAM 3 tracking preview window (no policy, no motion).

Continuous loop: D405 frame (58 degree vertical crop, as the client sees
it) -> mask server -> overlay window. Red is the SAM 3 tool mask; green is
the gripper LUT mask at the current opening when an arm address is given
(the arm is only read). Press q in the window or Ctrl-C to quit.

    python -m simretract.real.live_view [--mask-port 8766] [--follower-ip A.B.C.D]
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import logging
import os

import cv2
import numpy as np

from simretract.real import follower_ip_is_set
from simretract.real.client import (D405, default_lut_path, env_default,
                                    validate_server_capabilities)
from simretract.serve.protocol import PolicyClient

log = logging.getLogger("simretract.real.live_view")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m simretract.real.live_view",
                                 description="Live SAM 3 mask overlay from the wrist camera.")
    ap.add_argument("--host", default=env_default(str, "127.0.0.1", "SIMRETRACT_HOST"))
    ap.add_argument("--mask-port", dest="mask_port", type=int,
                    default=env_default(int, 8766, "SIMRETRACT_MASK_PORT", "MASK_PORT"))
    ap.add_argument("--follower-ip", dest="follower_ip", default=None,
                    help="follower controller IP for the gripper overlay (optional; arm is read-only)")
    ap.add_argument("--arm-model", dest="arm_model", default="wxai_v0")
    ap.add_argument("--lut", default=None, help="gripper LUT (default assets/gmask_lut.npz)")
    return ap


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    a = build_parser().parse_args(argv)
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise SystemExit("live view needs a display (DISPLAY is not set)")
    with ExitStack() as preflight:
        masker = PolicyClient(a.host, a.mask_port)
        preflight.callback(masker.close)
        validate_server_capabilities({}, masker.request({"op": "ping"}), 0)
        masker.request({"op": "reset"})
        with np.load(a.lut or str(default_lut_path())) as lut:
            lut_open, lut_masks = lut["openings"], lut["masks"]
        gmask = None
        if follower_ip_is_set(a.follower_ip):
            try:
                from simretract.real.client import Arm
                arm = Arm(a.follower_ip, a.arm_model)
                try:
                    w = float(arm.read()[6])
                finally:
                    arm.cleanup()
                gmask = lut_masks[int(np.argmin(np.abs(lut_open - w)))] > 0
                log.info("gripper mask at opening %.4f", w)
            except Exception as e:
                log.warning("no arm (%s); gripper overlay off", e)
        cam = D405(warmup_frames=10)
        preflight.callback(cam.stop)
        cv2.namedWindow("SAM3 live", cv2.WINDOW_NORMAL)
        preflight.callback(cv2.destroyAllWindows)
        cv2.resizeWindow("SAM3 live", 1280, 480)
        log.info("window open; press q to quit")
        preflight.pop_all()
    try:
        while True:
            rgb, _ = cam.frame()
            req = {"rgb_bytes": rgb.tobytes(), "shape": rgb.shape}
            if gmask is not None:
                req["gmask_bytes"] = (gmask.astype(np.uint8) * 255).tobytes()
            m = np.frombuffer(masker.request(req), np.uint8).reshape(480, 640) > 0
            ov = rgb.copy()
            ov[m] = (0.4 * ov[m] + 0.6 * np.array([255, 60, 60])).astype(np.uint8)
            if gmask is not None:
                ov[gmask] = (0.6 * ov[gmask] + 0.4 * np.array([60, 255, 60])).astype(np.uint8)
            masked = np.zeros_like(rgb)
            vis = m | (gmask if gmask is not None else False)
            masked[vis] = rgb[vis]
            panel = np.hstack([ov, masked])
            cv2.putText(panel, f"tool px {int(m.sum())}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)
            cv2.imshow("SAM3 live", cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        masker.close()
        cv2.destroyAllWindows()
        log.info("closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
