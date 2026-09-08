"""Closed-loop simulation evaluator (runs inside Isaac Sim).

For every seed the evaluator rebuilds the recorder-identical scene
(presented tool pose, operator-hand occluder, lighting, tool mass) and lets
a policy served by ``simretract.serve.policy_server`` drive the arm at
25 Hz simulation time: capture RGB / depth / tool mask / gripper mask from
the wrist camera, query the server, execute ``--exec-h`` of the 50-step
chunk as raw joint targets (EMA-smoothed), re-query.  The episode ends when
the gripper close completes (left carriage stalled for ``STALL_N``
consecutive frames with both pads in contact) or after ``--max-steps``
control ticks.

After a close the operator hand releases the tool on camera; the release
drop, the deviation of the tool pose from the PRESENTED pose (tip
translation, rotation, swing/twist split) and the load battery verdict are
recorded in ``result_<seed>.json`` together with the per-frame joint
states in ``ep_<seed>/meta.npz`` and an optional review video.

Benchmark protocol: this evaluator sends ``plane_top_cut: 0.0`` with every
query (no top cut) and uses the test hand style; the real-robot client
applies its own deployment defaults.

Launch the server first; the evaluator pings it before booting Kit.  One
Kit process per seed is the robust pattern (see ``simretract.eval.driver``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from simretract.serve.protocol import PolicyClient

#: Fixed language prompt (recorded in meta.npz; part of the model contract).
TASK = "take over the tool from the operator and hold its pose"
#: Physics steps per control tick: 500 Hz / 20 = 25 Hz.
CONTROL_DECIM = 20
#: Wire chunk length.
H = 50
#: Left-carriage motion below this (m/frame) counts as stalled.
STALL_EPS = 8e-4
#: Consecutive stalled + both-pads-contacting frames = close complete.
STALL_N = 12
#: Gripper command range (m).
GRIPPER_MAX = 0.044
#: Release drop (mm) above which the tool counts as dropped on release.
DROP_MM = 20.0
#: Tool-centre-point offset from the end-effector frame (m).
EE_TO_TCP = np.array([0.1487 - 0.1055, 0.0, 0.0])
#: Perturbed-start variants: (tick, joint-space |dq| rad, gripper offset m).
PERTURB_CONF = [(50, 0.06, 0.0), (50, 0.12, 0.0), (50, 0.20, -0.02),
                (110, 0.06, 0.0), (110, 0.12, 0.0), (110, 0.20, -0.02)]


def _env(name: str, default: str) -> str:
    return os.environ.get(f"SIMRETRACT_{name}", os.environ.get(name, default))


@dataclass
class EvalConfig:
    """Everything the evaluator needs for one process."""

    seeds: list[int]
    tag: str
    out: Path
    tool: str = "retractor_rake"
    host: str = "127.0.0.1"
    port: int = 8767
    max_steps: int = 450
    exec_h: int = 50
    smooth: float = 0.45
    skip_search: bool = True
    tries: int = 10
    mu: float = 0.4
    battery: bool = True
    perturb_variant: int | None = None
    keep_frames: bool = False
    review_mp4: bool = True
    hand_style: str = "test"
    plane_top_cut: float = 0.0

    @property
    def suffix(self) -> str:
        return f"_v{self.perturb_variant}" if self.perturb_variant is not None else ""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="simretract.eval.sim_eval",
        description="Closed-loop policy evaluation in Isaac Sim (one process, "
                    "one or more seeds). Flags fall back to the environment "
                    "variable named in their help.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--seeds", default=_env("SEED_LIST", ""),
                   help="comma-separated seeds to evaluate [env SEED_LIST]")
    p.add_argument("--seed0", type=int, default=int(_env("SEED0", "90000")),
                   help="first seed when --seeds is empty [env SEED0]")
    p.add_argument("--n-seeds", type=int, default=int(_env("SEEDS", "4")),
                   help="number of seeds from --seed0 when --seeds is empty [env SEEDS]")
    p.add_argument("--tag", default=_env("CKPT_TAG", ""),
                   help="run tag recorded in every result (required) [env CKPT_TAG]")
    p.add_argument("--out", default=_env("EVAL_OUT", ""),
                   help="output directory (default: <paths.eval>/<tag>) [env EVAL_OUT]")
    p.add_argument("--tool", default=_env("TOOL", "retractor_rake"), help="tool name [env TOOL]")
    p.add_argument("--host", default=_env("HOST", "127.0.0.1"), help="policy server host [env HOST]")
    p.add_argument("--port", type=int, default=int(_env("PORT", "8767")),
                   help="policy server port [env PORT]")
    p.add_argument("--max-steps", type=int, default=int(_env("MAX_STEPS", "450")),
                   help="control ticks (25 Hz) before the episode times out [env MAX_STEPS]")
    p.add_argument("--exec-h", type=int, default=int(_env("EXEC_H", "50")),
                   help="actions executed from each chunk before re-querying [env EXEC_H]")
    p.add_argument("--smooth", type=float, default=float(_env("SMOOTH", "0.45")),
                   help="EMA weight on the previous target (0 = off) [env SMOOTH]")
    p.add_argument("--skip-search", dest="skip_search", action="store_true",
                   default=_env("SKIP_SEARCH", "1") == "1",
                   help="trust the seed's feasibility; no scripted grasp search, "
                        "grasp_pose zeros on the wire [env SKIP_SEARCH]")
    p.add_argument("--search", dest="skip_search", action="store_false",
                   help="run the silent scripted grasp search first")
    p.add_argument("--tries", type=int, default=int(_env("TRIES", "10")),
                   help="candidate pool size / scripted tries [env TRIES]")
    p.add_argument("--mu", type=float, default=float(_env("MU", "0.4")),
                   help="tool contact friction [env MU]")
    p.add_argument("--no-battery", dest="battery", action="store_false",
                   default=_env("NO_BATTERY", "0") != "1",
                   help="gravity-only tier: no load battery after release [env NO_BATTERY]")
    p.add_argument("--perturb-variant", type=int, default=None, choices=range(6),
                   help="perturbed-start benchmark variant [env PERTURB_VARIANT]")
    p.add_argument("--keep-frames", action="store_true", default=_env("KEEP_FRAMES", "0") == "1",
                   help="keep the per-frame PNGs [env KEEP_FRAMES]")
    p.add_argument("--no-review-mp4", dest="review_mp4", action="store_false",
                   default=_env("REVIEW_MP4", "1") == "1",
                   help="skip the 2x2 review video [env REVIEW_MP4]")
    p.add_argument("--hand-style", default=_env("HAND_STYLE", "test"), choices=["test", "train"],
                   help="operator-hand occluder distribution; the benchmark uses "
                        "'test' [env HAND_STYLE]")
    p.add_argument("--plane-top-cut", type=float, default=float(_env("PLANE_TOP_CUT", "0")),
                   help="plane top-cut fraction sent with every query (protocol: 0)")
    return p


def config_from_args(a: argparse.Namespace) -> EvalConfig:
    from simretract import config

    if not a.tag:
        build_parser().error("--tag is required")
    seeds = [int(s) for s in str(a.seeds).split(",") if s.strip()]
    if not seeds:
        seeds = [a.seed0 + i for i in range(a.n_seeds)]
    pv = a.perturb_variant
    if pv is None and _env("PERTURB_VARIANT", "") != "":
        pv = int(_env("PERTURB_VARIANT", ""))
    out = Path(a.out) if a.out else Path(config.paths()["eval"]) / a.tag
    return EvalConfig(
        seeds=seeds, tag=a.tag, out=out, tool=a.tool, host=a.host, port=a.port,
        max_steps=a.max_steps, exec_h=a.exec_h, smooth=a.smooth,
        skip_search=a.skip_search, tries=a.tries, mu=a.mu, battery=a.battery,
        perturb_variant=pv, keep_frames=a.keep_frames, review_mp4=a.review_mp4,
        hand_style=a.hand_style, plane_top_cut=a.plane_top_cut)


# ------------------------------------------------------- scene-draw contract
def sample_presented_pose(rng: np.random.Generator):
    """Presented tool pose: position in the box x U(.35,.50) y U(-.12,.12)
    z U(.15,.30); tilt U(0,1.4 rad) about a random horizontal axis on top of
    a 180 deg x-flip; spin U(-pi,pi).  Draw order is shared with the
    recorder so a seed reproduces its training scene.  Returns
    (pos, Rotation, tilt_deg)."""
    from scipy.spatial.transform import Rotation

    pos = np.array([rng.uniform(0.35, 0.50), rng.uniform(-0.12, 0.12),
                    rng.uniform(0.15, 0.30)])
    base = Rotation.from_euler("x", np.pi)
    theta = rng.uniform(0.0, 1.4)
    phi = rng.uniform(0.0, 2 * np.pi)
    psi = rng.uniform(-np.pi, np.pi)
    r = (Rotation.from_rotvec(theta * np.array([np.cos(phi), np.sin(phi), 0]))
         * base * Rotation.from_euler("z", psi))
    return pos, r, float(np.degrees(theta))


def load_candidates(path: Path, kind: str) -> list[dict]:
    """Grasp candidates of one kind in the tool frame; standoff clipped to
    the reachable [0.05, 0.09] m band."""
    with open(path) as f:
        db = json.load(f)
    return [{"kind": kind, "center": np.array(d["center"]),
             "R": np.array(d["R"]), "width": float(d["width"]),
             "standoff": float(np.clip(d["standoff"], 0.05, 0.09))}
            for d in db[kind]]


# ----------------------------------------------------------------- the world
class EvalWorld:
    """Stage, robot, tool and helpers for one Kit process.  Built once; the
    scene assembly order matches the recorder so seeds reproduce the
    training scenes."""

    def __init__(self, app: Any, cfg: EvalConfig) -> None:
        import omni.timeline
        import isaacsim.core.experimental.utils.stage as stage_utils
        from isaacsim.core.experimental.prims import RigidPrim
        from isaacsim.core.utils.semantics import add_update_semantics
        from omni.physx import get_physx_interface
        from pxr import UsdGeom

        from simretract import config
        from simretract.sim import scene as SL
        from controller import RobotType, TrossenAIController  # vendor controller

        self.app = app
        self.cfg = cfg
        self.SL = SL
        from simretract.sim.tool_usd import require_tool_usd

        assets = Path(config.paths()["assets"])
        self.tool_usd = require_tool_usd(cfg.tool)   # built locally, never shipped (docs/ASSETS.md)
        with open(assets / "tools" / cfg.tool / "meta.json") as f:
            self.tip_local = np.array(json.load(f)["suggested_tip"])
        cand_path = assets / "grasp_candidates" / f"{cfg.tool}.json"
        self.flat = load_candidates(cand_path, "flat")
        self.anti = load_candidates(cand_path, "antipodal")

        stage_utils.create_new_stage()
        self.stage = stage_utils.get_current_stage()
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        SL.setup_physics_scene(self.stage, hz=500.0)
        SL.add_lights(self.stage)
        SL.add_ground(self.stage)
        stage_utils.add_reference_to_stage(usd_path=str(SL.robot_usd_path()), path="/World/wxai")
        SL.swap_fingers(self.stage)
        SL.set_wrist_intrinsics(self.stage)
        SL.spawn_tool(self.stage, kinematic=False, usd=str(self.tool_usd))
        SL.set_pose(self.stage, "/World/tool", (0.7, 0.45, 0.3), (1, 0, 0, 0))
        SL.set_gripper_drive(self.stage)
        SL.set_contact_friction(self.stage, cfg.mu)
        add_update_semantics(self.stage.GetPrimAtPath("/World/tool"), "tool")
        for gp in ("link_6", "carriage_left", "carriage_right"):
            add_update_semantics(self.stage.GetPrimAtPath(f"/World/wxai/{gp}"), "gripper")

        self.robot = TrossenAIController(
            robot_path="/World/wxai", robot_type=RobotType.WXAI,
            arm_dof_indices=[0, 1, 2, 3, 4, 5], gripper_dof_index=6,
            default_dof_positions=[0.0] * 6 + [GRIPPER_MAX, GRIPPER_MAX])
        self.timeline = omni.timeline.get_timeline_interface()
        self.contact_mon = SL.ContactMonitor(self.stage, "/World/tool")
        self.tool_rb = RigidPrim("/World/tool")
        self.hand = SL.OperatorHand(self.tool_rb)
        self.step_count = [0]
        self._sub = get_physx_interface().subscribe_physics_step_events(
            lambda dt: self.step_count.__setitem__(0, self.step_count[0] + 1))
        self.timeline.play()
        self.robot.set_gripper_position(GRIPPER_MAX)
        for _ in range(90):
            app.update()
        self.wrist = f"/World/wxai/{SL.WRIST_CAM_REL}"
        self.carr_l = "/World/wxai/carriage_left"
        self.carr_r = "/World/wxai/carriage_right"

    # -- primitive motions -------------------------------------------------
    def reset_arm(self) -> None:
        self.robot.set_gripper_position(GRIPPER_MAX)
        for _ in range(40):
            self.app.update()
        self.robot.reset_to_default_pose()
        for _ in range(50):
            self.app.update()

    def rehold(self, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
        self.tool_rb.set_world_poses(np.asarray(pos).reshape(1, 3),
                                     np.asarray(quat_wxyz).reshape(1, 4))
        self.tool_rb.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))
        self.hand.hold(pos, quat_wxyz)
        for _ in range(90):
            self.app.update()

    def move_ee(self, p0, q0, p1, q1, probes) -> str | None:
        from scipy.spatial.transform import Rotation, Slerp

        dist = float(np.linalg.norm(p1 - p0))
        n_steps = max(2, int(dist / (0.12 / 60.0)))
        sl = Slerp([0, 1], Rotation.concatenate([q0, q1]))
        for k in range(n_steps + 1):
            a = k / n_steps
            q = sl(a).as_quat()
            self.robot.set_end_effector_pose(p0 + a * (p1 - p0),
                                             np.array([q[3], q[0], q[1], q[2]]))
            self.app.update()
            if self.contact_mon.contacting_bodies():
                return "tool_contact"
            if k % 2 == 0 and self.SL.clearance_violations(self.tool_rb, probes, phase="reach"):
                return "occluder_interference"
        for j in range(30):
            self.app.update()
            if self.contact_mon.contacting_bodies():
                return "tool_contact"
            if j % 2 == 0 and self.SL.clearance_violations(self.tool_rb, probes, phase="reach"):
                return "occluder_interference"
        return None

    def run_candidate(self, cand, Rg_mat, grasp_w, probes, tool_pos, tool_quat):
        """Silent scripted try (recorder-identical): finds the candidate the
        recorder would have recorded for this seed."""
        from scipy.spatial.transform import Rotation

        R_grasp = Rotation.from_matrix(Rg_mat)
        approach = Rg_mat[:, 0]
        grasp_ee = grasp_w - R_grasp.apply(EE_TO_TCP)
        pre_ee = grasp_ee - approach * cand["standoff"]
        self.reset_arm()
        self.rehold(tool_pos, tool_quat)
        _d, ee_p, ee_q = self.robot.get_current_state()
        ee_p = np.asarray(ee_p).ravel()[:3]
        eq = np.asarray(ee_q).ravel()
        R_now = Rotation.from_quat([eq[1], eq[2], eq[3], eq[0]])

        why = self.move_ee(ee_p, R_now, pre_ee, R_grasp, probes)
        if why:
            return False, {"reject": why}
        why = self.move_ee(pre_ee, R_grasp, grasp_ee, R_grasp, probes)
        if why:
            return False, {"reject": why}
        qg = R_grasp.as_quat()
        for _ in range(40):
            self.robot.set_end_effector_pose(grasp_ee, np.array([qg[3], qg[0], qg[1], qg[2]]))
            self.app.update()
        _d, ee_now, _q = self.robot.get_current_state()
        err = float(np.linalg.norm(np.asarray(ee_now).ravel()[:3] - grasp_ee))
        if err > 0.012:
            return False, {"reject": "unreachable"}

        tgt, contact_q, k = GRIPPER_MAX, None, 0
        while tgt > 0.0:
            tgt -= 0.0009
            self.robot.set_gripper_position(max(tgt, 0.0))
            self.app.update()
            k += 1
            if k % 5 == 0 and self.SL.clearance_violations(self.tool_rb, probes, phase="close"):
                return False, {"reject": "close_interference"}
            touching = self.contact_mon.contacting_bodies()
            if self.carr_l in touching and self.carr_r in touching:
                contact_q = float(np.asarray(self.robot.get_dof_positions()).ravel()[-2])
                break
        if contact_q is None:
            return False, {"reject": "no_clamp"}
        self.robot.set_gripper_position(0.0)
        for _ in range(120):
            self.app.update()
        qc = np.asarray(self.robot.get_dof_positions()).ravel()[-2:]
        gap = self.SL.finger_base_gap() + float(qc.sum())
        if abs(gap - cand["width"]) > 0.0012:
            return False, {"reject": "width_mismatch"}

        p_before = np.asarray(self.tool_rb.get_world_poses()[0]).ravel()
        self.hand.release()
        for _ in range(90):
            self.app.update()
        p_after = np.asarray(self.tool_rb.get_world_poses()[0]).ravel()
        drop = float((p_before[2] - p_after[2]) * 1000)
        return drop < DROP_MM, {"gap_mm": gap * 1000, "reach_err_mm": err * 1000,
                                "drop_mm": drop, "grasp_w": grasp_w.tolist(),
                                "R_grasp": Rg_mat.tolist()}


class EvalCapture:
    """25 Hz sim-time wrist-camera capture with persistent annotators.  Keeps
    the latest frame arrays for the policy query and the per-frame states
    for meta.npz."""

    def __init__(self, world: EvalWorld, ep_dir: Path) -> None:
        import omni.replicator.core as rep

        self.rep = rep
        self.world = world
        self.ep_dir = ep_dir
        ep_dir.mkdir(parents=True, exist_ok=True)
        self.rp = rep.create.render_product(world.wrist, (640, 480))
        self.a_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self.a_dep = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        self.a_seg = rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation", init_params={"colorize": False})
        for a in (self.a_rgb, self.a_dep, self.a_seg):
            a.attach(self.rp)
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.tool_ps: list[np.ndarray] = []
        self.tool_qs: list[np.ndarray] = []
        self.latest: dict | None = None
        self.t = 0
        self.next_boundary = (world.step_count[0] // CONTROL_DECIM + 1) * CONTROL_DECIM

    def maybe_frame(self) -> None:
        from PIL import Image

        w = self.world
        if w.step_count[0] < self.next_boundary:
            return
        self.next_boundary = (w.step_count[0] // CONTROL_DECIM + 1) * CONTROL_DECIM
        self.rep.orchestrator.step(rt_subframes=8)
        w.timeline.play()  # orchestrator.step leaves the timeline paused
        rgb = np.asarray(self.a_rgb.get_data())[..., :3].copy()
        dep = np.nan_to_num(np.asarray(self.a_dep.get_data()), posinf=0.0).squeeze()
        dep_mm = np.clip(dep * 1000.0, 0, 65535).astype(np.uint16)
        seg = self.a_seg.get_data()
        ids = np.asarray(seg["data"]).squeeze()
        id2c = {int(k): v.get("class", "") for k, v in seg["info"]["idToLabels"].items()}
        tool_ids = [i for i, c in id2c.items() if c == "tool"]
        grip_ids = [i for i, c in id2c.items() if c == "gripper"]
        msk = (np.isin(ids, tool_ids) * 255).astype(np.uint8)
        gmk = (np.isin(ids, grip_ids) * 255).astype(np.uint8)
        Image.fromarray(rgb).save(self.ep_dir / f"rgb_{self.t:04d}.png")
        Image.fromarray(dep_mm).save(self.ep_dir / f"depth_{self.t:04d}.png")
        Image.fromarray(msk).save(self.ep_dir / f"mask_{self.t:04d}.png")
        Image.fromarray(gmk).save(self.ep_dir / f"gmask_{self.t:04d}.png")
        self.latest = {"rgb": rgb, "depth_mm": dep_mm, "mask": msk, "gmask": gmk}
        self.states.append(np.asarray(w.robot.get_dof_positions()).ravel()[:7].astype(np.float32))
        self.actions.append(np.asarray(w.robot.get_dof_position_targets()).ravel()[:7].astype(np.float32))
        tp, tq = w.tool_rb.get_world_poses()
        self.tool_ps.append(np.asarray(tp).ravel().astype(np.float32))
        self.tool_qs.append(np.asarray(tq).ravel().astype(np.float32))
        self.t += 1

    def step_to_frame(self) -> None:
        """Advance physics to the next 25 Hz boundary, then capture once."""
        while self.world.step_count[0] < self.next_boundary:
            self.world.app.update()
        self.maybe_frame()

    def close(self) -> None:
        for a in (self.a_rgb, self.a_dep, self.a_seg):
            a.detach()
        self.rp.destroy()


# --------------------------------------------------------------- one seed
def run_seed(world: EvalWorld, cfg: EvalConfig, client: PolicyClient, seed: int) -> dict:
    """Evaluate one seed; returns the result row (also written to disk)."""
    from scipy.spatial.transform import Rotation

    from simretract.sim.sampling import sample_occluder_spec

    SL = world.SL
    app = world.app
    robot, tool_rb, hand, contact_mon = world.robot, world.tool_rb, world.hand, world.contact_mon
    sfx = cfg.suffix

    # scene draw order is recorder-identical: the same seed gives the same
    # scene AND the same candidate pool ordering (shared rng stream)
    rng = np.random.default_rng(seed)
    tool_pos, R_tool, tilt = sample_presented_pose(rng)
    spec = sample_occluder_spec(seed, world.tip_local, hand_style=cfg.hand_style)
    probes = SL.author_occluder(world.stage, spec)
    light_spec = SL.randomize_lighting(world.stage, np.random.default_rng(seed ^ 0x4C494748))
    mass = float(rng.uniform(0.06, 0.16))
    tool_rb.set_masses(np.array([mass]))
    q_tool = R_tool.as_quat()
    tool_quat = np.array([q_tool[3], q_tool[0], q_tool[1], q_tool[2]])
    row: dict[str, Any] = {"seed": seed, "ckpt": cfg.tag, "tilt_deg": round(tilt),
                           "mass_g": round(mass * 1000, 1),
                           "occ_z": round(spec["hand"]["z_along_tool"], 3)}

    # scene pre-filters (recorder-identical)
    world.reset_arm()
    world.rehold(tool_pos, tool_quat)
    if contact_mon.contacting_bodies() or SL.clearance_violations(tool_rb, probes, phase="reach"):
        row["outcome"] = "scene_reject"
        return row

    hand_subs = spec["hand"]["subs"]

    def gates(c):
        Rg = R_tool.as_matrix() @ c["R"]
        if Rg[2, 0] > 0.3 or Rg[2, 2] < 0.0:
            return None
        gw = tool_pos + R_tool.apply(c["center"])
        if gw[2] < 0.08:
            return None
        for s in hand_subs:
            if np.linalg.norm(c["center"] - np.asarray(s["pos"])) < max(s["half"]) + 0.016:
                return None
        return Rg, gw

    def corridor(c, Rg, gw):
        approach = Rg[:, 0]
        pre = gw - approach * c["standoff"]
        pts_w = pre + np.linspace(0, 1, 8)[:, None] * (gw - pre)
        pts_t = R_tool.inv().apply(pts_w - tool_pos)
        mind = np.inf
        for s in hand_subs + spec["tip"]["subs"]:
            sq = np.asarray(s["quat_wxyz"])
            R_s = Rotation.from_quat([sq[1], sq[2], sq[3], sq[0]])
            local = R_s.inv().apply(pts_t - np.asarray(s["pos"]))
            d = np.linalg.norm(np.maximum(np.abs(local) - (np.asarray(s["half"]) + 0.006), 0.0), axis=1)
            mind = min(mind, float(d.min()))
        return mind

    pool = []
    for group in (world.flat, world.anti):
        for j in rng.permutation(len(group)):
            g = gates(group[j])
            if g is not None:
                cd = corridor(group[j], g[0], g[1])
                if cd > 0.025:
                    pool.append((cd, group[j], *g))
        if len(pool) >= cfg.tries:
            break
    pool.sort(key=lambda t: -t[0])

    # silent search for the recorder's winning candidate, or (protocol) trust
    # the seed's feasibility and send a zero grasp_pose the server slices away
    if cfg.skip_search:
        if not pool:
            row["outcome"] = "no_grasp"
            return row
        grasp_pose = np.zeros(7, np.float32)
    else:
        winner = None
        for _cd, cand, Rg, gw in pool[:cfg.tries]:
            ok, _info = world.run_candidate(cand, Rg, gw, probes, tool_pos, tool_quat)
            if ok:
                winner = (cand, Rg, gw)
                break
        if winner is None:
            row["outcome"] = "no_grasp"
            return row
        _cand, Rg, gw = winner
        grasp_pose = np.concatenate([gw, Rotation.from_matrix(Rg).as_quat()[[3, 0, 1, 2]]]).astype(np.float32)

    # policy episode: reset the scene exactly like the recorded pass
    world.reset_arm()
    world.rehold(tool_pos, tool_quat)
    ep_dir = cfg.out / f"ep_{seed}{sfx}"
    cap = EvalCapture(world, ep_dir)
    cap.step_to_frame()  # frame 0 = initial observation

    pv = cfg.perturb_variant
    p_tick, p_mag, p_grip = PERTURB_CONF[pv] if pv is not None else (-1, 0.0, 0.0)
    closed_at, stall, prev_qL, n_query = None, 0, None, 0
    close_armed = False
    chunk = None
    err = None
    p_done = False
    both = False
    prev_a = None
    ci = cfg.exec_h  # actions used from the current chunk; == exec_h -> query now
    try:
        for tick in range(cfg.max_steps):
            if pv is not None and tick == p_tick and not p_done:
                # driven shove: joint targets jump by a deterministic dq and
                # physics settles for ~0.5 s; the stale chunk is discarded so
                # the policy re-plans from the perturbed state
                p_done = True
                q_now = np.asarray(robot.get_dof_positions()).ravel()[:7]
                prng = np.random.default_rng(seed * 100 + pv)
                d = prng.normal(size=6) * np.array([0.8, 0.8, 0.8, 0.9, 0.9, 0.9])
                dq = d / np.linalg.norm(d) * p_mag
                robot.set_dof_position_targets((q_now[:6] + dq).reshape(1, 6).astype(np.float64),
                                               dof_indices=[0, 1, 2, 3, 4, 5])
                if p_grip != 0.0:
                    robot.set_gripper_position(float(np.clip(q_now[6] + p_grip, 0.0, GRIPPER_MAX)))
                for _ in range(12):
                    cap.step_to_frame()
                ci = cfg.exec_h
                row["perturb"] = {"tick": p_tick, "mag": p_mag, "grip": p_grip,
                                  "dq": [round(float(x), 4) for x in dq]}
            if ci >= cfg.exec_h:
                qpos = np.asarray(robot.get_dof_positions()).ravel()[:7].astype(np.float32)
                state14 = np.concatenate([qpos, grasp_pose])
                assert cap.latest is not None
                chunk = np.asarray(client.request(
                    {"rgb": cap.latest["rgb"], "depth_mm": cap.latest["depth_mm"],
                     "mask": cap.latest["mask"], "gmask": cap.latest["gmask"],
                     "state": state14, "plane_top_cut": float(cfg.plane_top_cut)}),
                    dtype=np.float32)
                assert chunk.shape == (H, 7), f"bad chunk {chunk.shape}"
                n_query += 1
                ci = 0
            a = chunk[ci]
            ci += 1
            if cfg.smooth > 0:
                a = (1 - cfg.smooth) * a + cfg.smooth * prev_a if tick > 0 else a
            prev_a = a
            robot.set_dof_position_targets(a[:6].reshape(1, 6).astype(np.float64),
                                           dof_indices=[0, 1, 2, 3, 4, 5])
            robot.set_gripper_position(float(np.clip(a[6], 0.0, GRIPPER_MAX)))
            cap.step_to_frame()

            # pre-close centering: when the command first drops below 40 mm,
            # record the shaft offset from the pad midpoint along the closing
            # axis and perpendicular to it; re-armed if the gripper re-opens
            g_cmd = float(np.clip(a[6], 0.0, GRIPPER_MAX))
            if g_cmd < 0.040 and not close_armed:
                close_armed = True
                try:
                    pl = SL.world_transform(world.stage, world.carr_l)[:3, 3]
                    pr = SL.world_transform(world.stage, world.carr_r)[:3, 3]
                    mid = 0.5 * (pl + pr)
                    ax = pr - pl
                    ax = ax / max(np.linalg.norm(ax), 1e-9)
                    tp, tq = tool_rb.get_world_poses()
                    tp = np.asarray(tp).ravel()
                    tqv = np.asarray(tq).ravel()
                    z_t = Rotation.from_quat([tqv[1], tqv[2], tqv[3], tqv[0]]).apply([0, 0, 1.0])
                    near = tp + ((mid - tp) @ z_t) * z_t      # nearest shaft point
                    dvec = near - mid
                    off_close = float(dvec @ ax)
                    off_perp = float(np.linalg.norm(dvec - off_close * ax))
                    row["center_off_mm"] = round(abs(off_close) * 1000, 1)
                    row["center_off_perp_mm"] = round(off_perp * 1000, 1)
                    row["close_start_tick"] = tick
                except Exception:
                    row["center_off_mm"] = None
            elif g_cmd > 0.042:
                close_armed = False

            qL = float(np.asarray(robot.get_dof_positions()).ravel()[-2])
            touching = contact_mon.contacting_bodies()
            both = world.carr_l in touching and world.carr_r in touching
            if prev_qL is not None and abs(qL - prev_qL) < STALL_EPS and both:
                stall += 1
            else:
                stall = 0
            prev_qL = qL
            if stall >= STALL_N:
                closed_at = tick + 1
                break
        # timeout fallback: carriage near-closed with both pads in contact is
        # a real grab whose jitter kept the stall counter at bay
        if closed_at is None and prev_qL is not None and prev_qL < 0.008 and both:
            closed_at = cfg.max_steps
    except (RuntimeError, ConnectionError, AssertionError) as e:
        err = repr(e)

    row.update({"grasp_pose": [round(float(v), 5) for v in grasp_pose],
                "queries": n_query, "ticks": cap.t})
    if err is not None:
        cap.close()
        row["outcome"] = f"server_error:{err}"
        row.update({"held": False, "strict": False})
    elif closed_at is None:
        cap.close()
        row["outcome"] = "timeout"
        row.update({"held": False, "strict": False})
    else:
        from simretract.sim.battery import run_battery

        # release on camera, then the verdict battery (also on camera)
        p_before = np.asarray(tool_rb.get_world_poses()[0]).ravel()
        hand.release()
        for _ in range(90):
            app.update()
            cap.maybe_frame()
        po_a, qo_a = tool_rb.get_world_poses()
        p_after = np.asarray(po_a).ravel()
        drop = float((p_before[2] - p_after[2]) * 1000)
        # grasp-phase deviation vs the PRESENTED pose: a policy that spins the
        # tool while grasping has already destroyed the pose the operator set,
        # which the battery's post-release reference cannot see
        q_now = np.asarray(qo_a).ravel()
        R_now = Rotation.from_quat([q_now[1], q_now[2], q_now[3], q_now[0]])
        g_r = float(np.degrees((R_now.inv() * R_tool).magnitude()))
        tip_pres = np.asarray(tool_pos) + R_tool.apply(world.tip_local)
        tip_now = p_after + R_now.apply(world.tip_local)
        g_t = float(np.linalg.norm(tip_now - tip_pres) * 1000)
        # swing-twist split about the tool long axis (tool-frame z)
        dR = (R_now.inv() * R_tool)
        rv = dR.as_rotvec()
        tw = float(np.degrees(abs(rv[2])))
        ax0 = R_tool.apply([0, 0, 1.0])
        ax1 = R_now.apply([0, 0, 1.0])
        sw = float(np.degrees(np.arccos(np.clip(ax0 @ ax1, -1, 1))))
        if drop < DROP_MM and not cfg.battery:
            # gravity-only tier: held = survived release; strict = pose within
            # 10 mm / 10 deg
            cap.close()
            verdict = {"note": "gravity_only", "left_gripper": False,
                       "grasp_r_deg": round(g_r, 1), "grasp_t_mm": round(g_t, 1),
                       "grasp_swing_deg": round(sw, 1), "grasp_twist_deg": round(tw, 1),
                       "q_pres": R_tool.as_quat().tolist(), "q_post": R_now.as_quat().tolist(),
                       "strict_pass": bool(g_r < 10.0 and g_t < 10.0)}
            held = True
            strict = bool(verdict["strict_pass"])
        elif drop < DROP_MM:
            verdict = run_battery(app, tool_rb, hand, world.stage, world.tip_local,
                                  on_step=cap.maybe_frame)
            verdict["grasp_r_deg"] = round(g_r, 1)
            verdict["grasp_t_mm"] = round(g_t, 1)
            verdict["grasp_swing_deg"] = round(sw, 1)
            verdict["grasp_twist_deg"] = round(tw, 1)
            verdict["q_pres"] = R_tool.as_quat().tolist()
            verdict["q_post"] = R_now.as_quat().tolist()
            held = not verdict["left_gripper"]
            strict = bool(verdict["strict_pass"]) and g_r < 10.0 and g_t < 10.0
            cap.close()
        else:
            cap.close()
            verdict = {"strict_pass": False, "left_gripper": True, "note": "dropped_on_release"}
            held, strict = False, False
        row.update({"outcome": "completed", "closed_at": closed_at,
                    "drop_mm": round(drop, 1), "held": held, "strict": strict,
                    "verdict": verdict})

    if cap.t > 0:
        np.savez(ep_dir / "meta.npz",
                 states=np.asarray(cap.states), actions=np.asarray(cap.actions),
                 tool_pos_t=np.asarray(cap.tool_ps), tool_quat_t=np.asarray(cap.tool_qs),
                 grasp_pose=grasp_pose, seed=seed, task=TASK, fps=25,
                 tool=cfg.tool, ckpt=cfg.tag, result=json.dumps(row),
                 lighting=json.dumps(light_spec))
        if cfg.review_mp4:
            from simretract.sim.review import make_review_grid

            grid_mp4 = cfg.out / f"ep_{seed}{sfx}_review.mp4"
            make_review_grid(str(ep_dir), str(grid_mp4))
            print(f"[M5] review grid -> {grid_mp4}", flush=True)
        if not cfg.keep_frames:
            for fn in os.listdir(ep_dir):
                if fn.endswith(".png"):
                    os.remove(ep_dir / fn)
    return row


def main(argv: list[str] | None = None) -> int:
    from simretract import config

    cfg = config_from_args(build_parser().parse_args(argv))
    config.require_kit_eula()
    cfg.out.mkdir(parents=True, exist_ok=True)

    client = PolicyClient(cfg.host, cfg.port)  # fail fast BEFORE the slow Kit boot
    pong = client.ping()
    print(f"[M5] server ok: {pong}", flush=True)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    config.ensure_trossen_on_path()
    world = EvalWorld(app, cfg)

    results = []
    for seed in cfg.seeds:
        t0 = time.time()
        row = run_seed(world, cfg, client, seed)
        results.append(row)
        print(f"[M5] {json.dumps(row)}", flush=True)
        # scene rejections and empty candidate pools leave no result file:
        # the driver treats such seeds as missing and retries them
        if "grasp_pose" in row:
            with open(cfg.out / f"result_{seed}{cfg.suffix}.json", "w") as f:
                json.dump(row, f, indent=1)
        print(f"[M5] seed {seed} {row['outcome']} in {time.time() - t0:.0f}s", flush=True)

    n_done = sum(1 for r in results if r.get("outcome") == "completed")
    n_held = sum(1 for r in results if r.get("held"))
    n_strict = sum(1 for r in results if r.get("strict"))
    n_scene = sum(1 for r in results if r.get("outcome") in ("scene_reject", "no_grasp"))
    summary = {"ckpt": cfg.tag, "tool": cfg.tool, "seeds": len(results),
               "scene_or_grasp_reject": n_scene, "completed": n_done,
               "held": n_held, "strict": n_strict}
    print(f"[M5] SUMMARY {json.dumps(summary)}", flush=True)
    with open(cfg.out / f"eval_log_{cfg.seeds[0]}.jsonl", "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(cfg.out / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("[M5] DONE", flush=True)
    sys.stdout.flush()
    os._exit(0)  # Kit's teardown intermittently hangs; the driver reaps by PID


if __name__ == "__main__":
    main()
