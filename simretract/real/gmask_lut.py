"""Render the gripper-silhouette lookup table used by the real client.

The wrist camera is rigid to the last link, so the gripper's appearance in
the wrist view depends only on the gripper opening (both carriages mirror
one commanded value). This tool sweeps the opening over its range in the
headless simulator and stores the semantic gripper mask per step; at
deployment the client looks the mask up by the measured carriage position.
No tool and no hand are present, so the mask contains gripper pixels only,
exactly like the simulated gripper-mask channel.

Isaac Sim (Kit) module: run under the simulation interpreter, e.g.
``OMNI_KIT_ACCEPT_EULA=YES <sim_python> -m simretract.real.gmask_lut``
or via ``simretract-build-assets --lut``.

Output: ``assets/gmask_lut.npz`` with ``openings`` (N,) float64 achieved
openings in metres and ``masks`` (N, 480, 640) u8 0/255.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

N_STEPS = 45
"""Opening resolution: 45 steps over 0..0.044 m is about 1 mm."""
OPEN_MAX = 0.044


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m simretract.real.gmask_lut",
        description="Render the gripper-mask LUT (headless Isaac Sim).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out", default=None, help="output .npz (default assets/gmask_lut.npz)")
    ap.add_argument("--preview", default=None,
                    help="optional preview strip PNG of five openings")
    ap.add_argument("--steps", type=int, default=N_STEPS, help="number of openings")
    ap.add_argument("--open-max", dest="open_max", type=float, default=OPEN_MAX,
                    help="maximum gripper opening (m)")
    return ap


def render_lut(app, steps: int, open_max: float, out: str, preview: str | None) -> None:
    """Sweep the gripper and save the LUT inside the live SimulationApp ``app``."""
    import omni.replicator.core as rep
    import omni.timeline
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.core.utils.semantics import add_update_semantics
    from pxr import UsdGeom

    from simretract.sim import scene as SL

    controller = SL.import_controller()
    RobotType, TrossenAIController = controller.RobotType, controller.TrossenAIController

    stage_utils.create_new_stage()
    usd_stage = stage_utils.get_current_stage()
    UsdGeom.SetStageMetersPerUnit(usd_stage, 1.0)
    UsdGeom.SetStageUpAxis(usd_stage, UsdGeom.Tokens.z)
    SL.setup_physics_scene(usd_stage, hz=500.0)
    SL.add_lights(usd_stage)
    SL.add_ground(usd_stage)
    stage_utils.add_reference_to_stage(usd_path=str(SL.robot_usd_path()), path="/World/wxai")
    SL.swap_fingers(usd_stage)
    SL.set_wrist_intrinsics(usd_stage)
    SL.set_gripper_drive(usd_stage)
    for gp in ("link_6", "carriage_left", "carriage_right"):
        add_update_semantics(usd_stage.GetPrimAtPath(f"/World/wxai/{gp}"), "gripper")

    robot = TrossenAIController(
        robot_path="/World/wxai", robot_type=RobotType.WXAI,
        arm_dof_indices=[0, 1, 2, 3, 4, 5], gripper_dof_index=6,
        default_dof_positions=[0.0] * 6 + [open_max, open_max])
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(90):
        app.update()

    rp = rep.create.render_product(f"/World/wxai/{SL.WRIST_CAM_REL}", (640, 480))
    a_seg = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": False})
    a_seg.attach(rp)

    openings = np.linspace(0.0, open_max, steps)
    masks = np.zeros((steps, 480, 640), np.uint8)
    for k, w in enumerate(openings):
        robot.set_gripper_position(float(w))
        for _ in range(60):  # let the drive settle at the target opening
            app.update()
        ach = float(np.asarray(robot.get_dof_positions()).ravel()[6])
        rep.orchestrator.step(rt_subframes=8)
        timeline.play()
        seg = a_seg.get_data()
        ids = np.asarray(seg["data"]).squeeze()
        id2c = {int(i): v.get("class", "") for i, v in seg["info"]["idToLabels"].items()}
        gids = [i for i, c in id2c.items() if c == "gripper"]
        if not gids:
            raise RuntimeError(f"gripper semantics missing at step {k}")
        m = (np.isin(ids, gids) * 255).astype(np.uint8)
        masks[k] = m
        openings[k] = ach  # store the achieved opening, not the target
        print(f"[LUT] {k + 1}/{steps} target {w:.4f} achieved {ach:.4f} px {int((m > 0).sum())}",
              flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, openings=openings, masks=masks)
    print(f"[LUT] saved {out}", flush=True)
    if preview:
        from PIL import Image
        idx = np.linspace(0, steps - 1, 5).round().astype(int)
        strip = np.concatenate([masks[i] for i in idx], axis=1)
        os.makedirs(os.path.dirname(os.path.abspath(preview)), exist_ok=True)
        Image.fromarray(strip).save(preview)
        print(f"[LUT] preview -> {preview}", flush=True)


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    from simretract import config
    config.require_kit_eula()
    if a.out is None:
        a.out = str(config.paths()["assets"] / "gmask_lut.npz")
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    try:
        render_lut(app, a.steps, a.open_max, a.out, a.preview)
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
