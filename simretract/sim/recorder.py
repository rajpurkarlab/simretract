"""Expert episode recorder: search silently, then record the winning grasp.

Per seed the scene is drawn deterministically (presented pose, occluders,
lighting, mass), grasp candidates are gated geometrically, a silent physical
search (or a stored winner from a previous set) picks the grasp, and the
winning approach is recorded at 25 Hz sim time as a raw episode:

    ep_<seed>/rgb_%04d.png      wrist RGB 640x480
    ep_<seed>/depth_%04d.png    uint16 millimetres
    ep_<seed>/mask_%04d.png     tool mask 0/255
    ep_<seed>/gmask_%04d.png    gripper mask 0/255
    ep_<seed>/omask_%04d.png    occluder mask 0/255
    ep_<seed>/meta.npz          states/actions/tool pose per frame + verdicts

Deterministic rejects leave an ``ep_<seed>.filtered`` marker (never retry);
a stalled episode leaves ``ep_<seed>.wedged``. One ``[M3] {json}`` line per
seed reports the outcome; ``[M3] DONE`` ends the run. Every knob of the
recording recipe lives in :class:`RecorderConfig`, whose defaults are the
production strict recipe; the CLI exposes each as a flag with an
environment fallback.

Run as ``<sim python> -m simretract.sim.recorder --out DIR --seeds ...``
(Isaac Sim boots inside ``main``).
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import json
import logging
import os
import shutil
import signal
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from simretract.sim.grasp_candidates import load_candidate_db, load_cands
from simretract.sim.sampling import (LIGHTING_STREAM_TAG, check_hand_style,
                                     sample_occluder_spec, sample_presented_pose)
from simretract.sim.tool_usd import DEFAULT_TOOL, require_eula, require_tool_usd, tip_local

log = logging.getLogger(__name__)

CONTROL_DECIM = 20          # physics steps per frame: 500 Hz / 20 = 25 fps
PHYSICS_HZ = 500.0
FRAME_RES = (640, 480)
RT_SUBFRAMES = 8
TASK = "take over the tool from the operator and hold its pose"
EE_TO_TCP = np.array([0.1487 - 0.1055, 0.0, 0.0])
STEP_NOMINAL_M = 0.12 / 60.0   # commanded TCP step per control update
OUTCOME_TAG = "[M3]"
DONE_LINE = "[M3] DONE"

# grasp-height bins used to stratify optional review videos
REVIEW_BINS = ((-0.03, "m6m3"), (0.0, "m3z0"), (0.03, "z0p3"), (None, "p3p6"))


class EpisodeStalled(Exception):
    """Raised by the per-episode watchdog."""


# ----------------------------------------------------------------- config
def _env(names: tuple[str, ...], default: Any, cast: Callable[[str], Any]) -> Any:
    """First environment variable of ``names`` that is set, cast; else default."""
    for n in names:
        v = os.environ.get(n)
        if v is not None and v != "":
            return cast(v)
    return default


def _env_bool(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclasses.dataclass
class RecorderConfig:
    """Recording recipe. Defaults = the production strict recipe.

    Field names double as CLI flags (``--standoff``) with environment
    fallbacks ``SIMRETRACT_<FIELD>`` and the legacy upper-case names.
    """
    tool: str = DEFAULT_TOOL
    mu: float = 0.4                 # tool + finger friction
    tries: int = 20                 # candidates physically tried in the silent search
    hand_style: str = "train"       # occluder hand distribution: train | test
    blend: bool = True              # single Bezier reach (True) or two straight segments
    blend_rot_frac: float = 0.7     # orientation slerp completes over this path fraction
    standoff: float = 0.03          # Bezier ends this far short of the grasp (m), then straight
    ctrl_back: float = 0.06         # Bezier control point behind the pre-grasp (m)
    fixed_standoff: float = 0.06    # straight reach only: fixed pre-grasp distance (0 = candidate's)
    speed_scale: float = 0.9        # scales the commanded step size of the blended reach
    joint_rate_deg: float = 2.5     # per-joint rate limit per update (time-scaling, blended reach)
    decel_arc: float = 0.0          # ramp the step down over the last N m of curve (0 = off)
    curve_end: float = 0.25         # step scale at the end of the deceleration ramp
    entry_speed: float = 1.0        # speed factor of the straight final entry
    tangent_orient: bool = False    # orient the approach axis along the path tangent
    interfere_gate: bool = True     # reject when the presented tool moves before the close
    interfere_t_mm: float = 3.0
    interfere_r_deg: float = 3.0
    keep_mode: str = "pose"         # pose: drop<20 mm and pose delta <10 mm/10 deg; battery: strict_pass
    skip_battery: bool = True       # skip the unrecorded load battery
    picker: str = "height"          # height: nearest to a target grasp height; clearance: max corridor
    z_lo: float = -0.05             # target grasp-height range along the tool axis (m)
    z_hi: float = 0.05
    z_strict: bool = False          # hard-reject candidates outside [z_lo, z_hi]
    upright_min: float = 0.0        # reject grips with the camera-side axis below this
    pert_frac_lo: float = 0.25      # recovery: where along the reach the excursion starts
    pert_frac_hi: float = 0.75
    review_every: int = 0           # write a review mp4 for every N-th kept episode (0 = off)
    episode_timeout_s: int = 0      # per-episode watchdog (0 = off)

    # CLI spellings that differ from the field name (the campaign driver owns
    # ``--tries`` for launches per item)
    FLAG_NAMES = {"tries": "search-tries"}

    @classmethod
    def flag_name(cls, field: str) -> str:
        return cls.FLAG_NAMES.get(field, field.replace("_", "-"))

    @classmethod
    def attr_name(cls, field: str) -> str:
        return cls.flag_name(field).replace("-", "_")

    @staticmethod
    def env_names(field: str) -> tuple[str, ...]:
        legacy = {"hand_style": (), "picker": (), "review_every": ("SAMPLE_EVERY",)}
        names = (f"SIMRETRACT_{field.upper()}",)
        return names + legacy.get(field, (field.upper(),))

    @classmethod
    def from_env(cls) -> "RecorderConfig":
        """Defaults overridden by environment variables (legacy names included)."""
        cfg = cls()
        for f in dataclasses.fields(cls):
            default = getattr(cfg, f.name)
            if f.name == "hand_style":
                # legacy switch: ERA4_HAND=1 selected the train-style hand
                v = _env(("SIMRETRACT_HAND_STYLE",), None, str)
                if v is None:
                    e4 = _env(("ERA4_HAND",), None, str)
                    v = None if e4 is None else ("train" if e4 == "1" else "test")
                if v is not None:
                    setattr(cfg, f.name, v)
                continue
            if f.name == "picker":
                v = _env(("SIMRETRACT_PICKER",), None, str)
                if v is None:
                    e4 = _env(("E4_PICK",), None, str)
                    v = None if e4 is None else ("height" if e4 == "1" else "clearance")
                if v is not None:
                    setattr(cfg, f.name, v)
                continue
            cast = _env_bool if isinstance(default, bool) else type(default)
            setattr(cfg, f.name, _env(cls.env_names(f.name), default, cast))
        return cfg

    @classmethod
    def recipe_defaults(cls, kind: str) -> dict[str, Any]:
        """Recorder knobs of ``configs/generation/<kind>.yaml`` (strict |
        recovery) keyed by field name; legacy key spellings are mapped.
        Review videos stay opt-in (``review_every``) regardless of the file.
        Returns ``{}`` when the file is absent."""
        from simretract import config

        try:
            raw = config.generation_recipe(kind)
        except config.ConfigError:
            return {}
        names = {f.name for f in dataclasses.fields(cls)}
        out: dict[str, Any] = {}
        for k, v in raw.items():
            if k == "e4_pick":
                out["picker"] = "height" if v else "clearance"
            elif k in names and k != "review_every":
                out[k] = v
        return out

    def validate(self) -> "RecorderConfig":
        check_hand_style(self.hand_style)
        if self.keep_mode not in ("pose", "battery"):
            raise ValueError("keep_mode must be 'pose' or 'battery'")
        if self.picker not in ("height", "clearance"):
            raise ValueError("picker must be 'height' or 'clearance'")
        if self.joint_rate_deg <= 0:
            raise ValueError("joint_rate_deg must be > 0")
        return self

    @classmethod
    def add_args(cls, ap: argparse.ArgumentParser,
                 defaults: dict[str, Any] | None = None) -> None:
        """Add one flag per field. Precedence: flag > environment >
        ``defaults`` (recipe-specific overrides) > dataclass default."""
        base = cls.from_env()
        for name, value in (defaults or {}).items():
            if _env(cls.env_names(name), None, str) is None:
                setattr(base, name, value)
        g = ap.add_argument_group("recording recipe (env fallback: SIMRETRACT_<NAME> or legacy NAME)")
        for f in dataclasses.fields(cls):
            flag = "--" + cls.flag_name(f.name)
            default = getattr(base, f.name)
            doc = _FIELD_HELP.get(f.name, "")
            if isinstance(default, bool):
                g.add_argument(flag, action=argparse.BooleanOptionalAction, default=default,
                               help=f"{doc} (default: {default})")
            elif f.name == "hand_style":
                g.add_argument(flag, choices=("train", "test"), default=default,
                               help=f"{doc} (default: {default})")
            elif f.name == "keep_mode":
                g.add_argument(flag, choices=("pose", "battery"), default=default,
                               help=f"{doc} (default: {default})")
            elif f.name == "picker":
                g.add_argument(flag, choices=("height", "clearance"), default=default,
                               help=f"{doc} (default: {default})")
            else:
                g.add_argument(flag, type=type(default), default=default,
                               help=f"{doc} (default: {default})")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "RecorderConfig":
        kw = {f.name: getattr(args, cls.attr_name(f.name)) for f in dataclasses.fields(cls)}
        return cls(**kw).validate()

    def to_cli_args(self) -> list[str]:
        """Flags reproducing this config on a recorder command line."""
        out: list[str] = []
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            flag = self.flag_name(f.name)
            if isinstance(v, bool):
                out.append(f"--{flag}" if v else f"--no-{flag}")
            else:
                out += [f"--{flag}", str(v)]
        return out

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_FIELD_HELP = {
    "tool": "tool name under assets/tools and assets/usd",
    "mu": "physics-material friction of tool and finger pads",
    "tries": "max candidates physically tried in the silent search",
    "hand_style": "occluder hand distribution (train = training sets, test = frozen test seeds)",
    "blend": "single quadratic-Bezier reach; --no-blend = two straight segments via the pre-grasp",
    "blend_rot_frac": "orientation slerp completes over this fraction of the blended path",
    "standoff": "blended reach ends this far short of the grasp along the approach axis, then enters straight (m)",
    "ctrl_back": "Bezier control point this far behind the pre-grasp on the approach axis (m)",
    "fixed_standoff": "straight reach only: fixed pre-grasp distance (m); 0 = candidate's own distance",
    "speed_scale": "global scale of the commanded TCP step of the blended reach",
    "joint_rate_deg": "per-joint rate limit per update; faster IK steps are subdivided in time",
    "decel_arc": "ramp the step size down over the last N metres of curve (0 = constant speed)",
    "curve_end": "step scale reached at the end of the deceleration ramp",
    "entry_speed": "speed factor of the straight final entry",
    "tangent_orient": "orient the gripper approach axis along the path tangent (nose-first)",
    "interfere_gate": "reject a try when the presented tool moves before the close",
    "interfere_t_mm": "interference gate translation threshold (mm)",
    "interfere_r_deg": "interference gate rotation threshold (deg)",
    "keep_mode": "keep rule: pose (drop < 20 mm and post-release pose within 10 mm / 10 deg) or battery",
    "skip_battery": "skip the unrecorded load battery after recording",
    "picker": "candidate picker: height (nearest to a target height in [z_lo, z_hi]) or clearance (max corridor)",
    "z_lo": "lower bound of the target grasp height along the tool axis (m)",
    "z_hi": "upper bound of the target grasp height along the tool axis (m)",
    "z_strict": "hard-reject candidates outside [z_lo, z_hi] (bin-targeted gap fill)",
    "upright_min": "reject candidates whose camera-side axis z component is below this",
    "pert_frac_lo": "recovery only: earliest reach fraction where the excursion starts",
    "pert_frac_hi": "recovery only: latest reach fraction where the excursion starts",
    "review_every": "write a review mp4 for every N-th kept episode (0 = never)",
    "episode_timeout_s": "per-episode watchdog in seconds; a stall leaves a .wedged marker (0 = off)",
}


# ----------------------------------------------------------------- capture
class EpisodeCapture:
    """25 Hz sim-time recorder: frames + state/action at physics-step
    boundaries, persistent annotators for the whole episode."""

    def __init__(self, session: "SimSession", ep_dir: str | os.PathLike):
        import omni.replicator.core as rep

        self.s = session
        self.rep = rep
        self.ep_dir = str(ep_dir)
        os.makedirs(self.ep_dir, exist_ok=True)
        self.rp = rep.create.render_product(session.wrist_cam, FRAME_RES)
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
        self.t = 0
        self.next_boundary = (session.step_count // CONTROL_DECIM + 1) * CONTROL_DECIM

    def _seg_ids(self):
        seg = self.a_seg.get_data()
        ids = np.asarray(seg["data"]).squeeze()
        id2c = {int(k): v.get("class", "")
                for k, v in seg["info"]["idToLabels"].items()}
        return ids, id2c

    def maybe_frame(self) -> None:
        from PIL import Image

        s = self.s
        if s.step_count < self.next_boundary:
            return
        self.next_boundary = (s.step_count // CONTROL_DECIM + 1) * CONTROL_DECIM
        self.rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
        s.timeline.play()
        rgb = np.asarray(self.a_rgb.get_data())[..., :3]
        Image.fromarray(rgb).save(os.path.join(self.ep_dir, f"rgb_{self.t:04d}.png"))
        dep = np.nan_to_num(np.asarray(self.a_dep.get_data()), posinf=0.0).squeeze()
        dep_mm = np.clip(dep * 1000.0, 0, 65535).astype(np.uint16)
        Image.fromarray(dep_mm).save(os.path.join(self.ep_dir, f"depth_{self.t:04d}.png"))
        ids, id2c = self._seg_ids()
        tool_ids = [i for i, c in id2c.items() if c == "tool"]
        if self.t == 0 and not tool_ids:
            # tool semantics occasionally drop (all-zero mask): re-stamp the
            # label and re-render before recording anything
            print(f"{OUTCOME_TAG} tool semantics missing at t=0 - restamping", flush=True)
            s.stamp_tool_semantics()
            self.rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
            s.timeline.play()
            ids, id2c = self._seg_ids()
            tool_ids = [i for i, c in id2c.items() if c == "tool"]
            if not tool_ids:
                raise RuntimeError("tool semantics unrecoverable - abort episode, "
                                   "do not record blind")
        grip_ids = [i for i, c in id2c.items() if c == "gripper"]
        occ_ids = [i for i, c in id2c.items() if c == "occluder"]
        Image.fromarray((np.isin(ids, tool_ids) * 255).astype(np.uint8)).save(
            os.path.join(self.ep_dir, f"mask_{self.t:04d}.png"))
        Image.fromarray((np.isin(ids, grip_ids) * 255).astype(np.uint8)).save(
            os.path.join(self.ep_dir, f"gmask_{self.t:04d}.png"))
        Image.fromarray((np.isin(ids, occ_ids) * 255).astype(np.uint8)).save(
            os.path.join(self.ep_dir, f"omask_{self.t:04d}.png"))
        self.states.append(np.asarray(
            s.robot.get_dof_positions()).ravel()[:7].astype(np.float32))
        self.actions.append(np.asarray(
            s.robot.get_dof_position_targets()).ravel()[:7].astype(np.float32))
        tp, tq = s.tool_rb.get_world_poses()
        self.tool_ps.append(np.asarray(tp).ravel().astype(np.float32))
        self.tool_qs.append(np.asarray(tq).ravel().astype(np.float32))
        self.t += 1

    def close(self) -> None:
        for a in (self.a_rgb, self.a_dep, self.a_seg):
            a.detach()
        self.rp.destroy()


# ----------------------------------------------------------------- session
@dataclasses.dataclass
class SceneDraw:
    """Everything one seed decides about the scene."""
    seed: int
    rng: np.random.Generator
    tool_pos: np.ndarray
    R_tool: Any
    tool_quat: np.ndarray      # wxyz
    tilt_deg: float
    spec: dict
    probes: list
    light_spec: dict
    mass_kg: float

    def row(self) -> dict[str, Any]:
        return {"seed": self.seed, "tilt_deg": round(self.tilt_deg),
                "mass_g": round(self.mass_kg * 1000, 1),
                "occ_z": round(self.spec["hand"]["z_along_tool"], 3)}


class SimSession:
    """The assembled simulation plus the expert motion primitives.

    Created once per process (Kit is expensive); ``run_seed`` style loops
    reuse it. Subclasses override the reach primitives for other recipes.
    """

    def __init__(self, simulation_app, cfg: RecorderConfig):
        import omni.timeline
        import isaacsim.core.experimental.utils.stage as stage_utils
        from isaacsim.core.experimental.prims import RigidPrim
        from isaacsim.core.utils.semantics import add_update_semantics
        from omni.physx import get_physx_interface
        from pxr import UsdGeom
        from scipy.spatial.transform import Rotation, Slerp

        from simretract.sim import scene

        self.app = simulation_app
        self.cfg = cfg
        self.scene = scene
        self.Rotation, self.Slerp = Rotation, Slerp
        self._add_update_semantics = add_update_semantics
        self.tip_local = tip_local(cfg.tool)
        db = load_candidate_db(cfg.tool)
        self.flat, self.anti = load_cands(db, "flat"), load_cands(db, "antipodal")
        self.joint_rate = np.radians(cfg.joint_rate_deg)
        self.interfere_t = cfg.interfere_t_mm / 1000.0
        self.presented: dict[str, Any] = {"p": None, "R": None}

        stage_utils.create_new_stage()
        self.stage = stage_utils.get_current_stage()
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        scene.setup_physics_scene(self.stage, hz=PHYSICS_HZ)
        scene.add_lights(self.stage)
        scene.add_ground(self.stage)
        stage_utils.add_reference_to_stage(usd_path=str(scene.robot_usd_path()), path="/World/wxai")
        scene.swap_fingers(self.stage)
        scene.set_wrist_intrinsics(self.stage)
        scene.spawn_tool(self.stage, kinematic=False, usd=require_tool_usd(cfg.tool))
        scene.set_pose(self.stage, "/World/tool", (0.7, 0.45, 0.3), (1, 0, 0, 0))
        scene.set_gripper_drive(self.stage)
        scene.set_contact_friction(self.stage, cfg.mu)
        self.stamp_tool_semantics()
        for gp in ("link_6", "carriage_left", "carriage_right"):
            add_update_semantics(self.stage.GetPrimAtPath(f"/World/wxai/{gp}"), "gripper")

        controller = scene.import_controller()
        self.robot = controller.TrossenAIController(
            robot_path="/World/wxai", robot_type=controller.RobotType.WXAI,
            arm_dof_indices=[0, 1, 2, 3, 4, 5], gripper_dof_index=6,
            default_dof_positions=[0.0] * 6 + [0.044, 0.044])
        self.timeline = omni.timeline.get_timeline_interface()
        self.contact_mon = scene.ContactMonitor(self.stage, "/World/tool")
        self.tool_rb = RigidPrim("/World/tool")
        self.hand = scene.OperatorHand(self.tool_rb)
        self.step_count = 0
        self._sub = get_physx_interface().subscribe_physics_step_events(self._on_step)
        self.timeline.play()
        self.robot.set_gripper_position(0.044)
        for _ in range(90):
            self.app.update()
        self.wrist_cam = f"/World/wxai/{scene.WRIST_CAM_REL}"

    def _on_step(self, _dt) -> None:
        self.step_count += 1

    def stamp_tool_semantics(self) -> None:
        self._add_update_semantics(self.stage.GetPrimAtPath("/World/tool"), "tool")

    def update(self, cap: EpisodeCapture | None = None) -> None:
        self.app.update()
        if cap:
            cap.maybe_frame()

    # ---------------------------------------------------------- scene draw
    def draw_scene(self, seed: int) -> SceneDraw:
        """Configure the scene for ``seed`` (pose, occluders, lighting, mass).
        The returned rng continues the seed stream (used by the picker)."""
        rng = np.random.default_rng(seed)
        tool_pos, R_tool, tilt = sample_presented_pose(rng)
        spec = sample_occluder_spec(seed, self.tip_local, self.cfg.hand_style)
        probes = self.scene.author_occluder(self.stage, spec)
        light_spec = self.scene.randomize_lighting(
            self.stage, np.random.default_rng(seed ^ LIGHTING_STREAM_TAG))
        mass = float(rng.uniform(0.06, 0.16))
        self.tool_rb.set_masses(np.array([mass]))
        q_tool = R_tool.as_quat()
        tool_quat = np.array([q_tool[3], q_tool[0], q_tool[1], q_tool[2]])
        return SceneDraw(seed, rng, tool_pos, R_tool, tool_quat, tilt, spec, probes,
                         light_spec, mass)

    def scene_rejected(self, sd: SceneDraw) -> bool:
        """Presented scene infeasible (tool in contact or occluder overlap)."""
        self.reset_arm()
        self.rehold(sd.tool_pos, sd.tool_quat)
        return bool(self.contact_mon.contacting_bodies()
                    or self.scene.clearance_violations(self.tool_rb, sd.probes, phase="reach"))

    # ---------------------------------------------------------- primitives
    def reset_arm(self) -> None:
        self.robot.set_gripper_position(0.044)
        for _ in range(40):
            self.app.update()
        self.robot.reset_to_default_pose()
        for _ in range(50):
            self.app.update()

    def rehold(self, pos, quat) -> None:
        self.tool_rb.set_world_poses(np.asarray(pos).reshape(1, 3),
                                     np.asarray(quat).reshape(1, 4))
        self.tool_rb.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))
        self.hand.hold(pos, quat)
        for _ in range(90):
            self.app.update()

    def standoff_of(self, c: dict) -> float:
        cfg = self.cfg
        return cfg.fixed_standoff if (cfg.fixed_standoff > 0 and not cfg.blend) else c["standoff"]

    def tool_disturbed(self) -> bool:
        """Interference gate: the presented tool moved (contact through the
        yielding hand) before the close."""
        if not self.cfg.interfere_gate or self.presented["p"] is None:
            return False
        tp, tq = self.tool_rb.get_world_poses()
        tp = np.asarray(tp).ravel()
        if np.linalg.norm(tp - self.presented["p"]) > self.interfere_t:
            return True
        tqv = np.asarray(tq).ravel()
        R_now = self.Rotation.from_quat([tqv[1], tqv[2], tqv[3], tqv[0]])
        return np.degrees((R_now.inv() * self.presented["R"]).magnitude()) > self.cfg.interfere_r_deg

    def curve_step_scale(self, rem: float) -> float:
        cfg = self.cfg
        if cfg.decel_arc <= 0:
            return 1.0
        return cfg.curve_end + (1.0 - cfg.curve_end) * min(1.0, rem / cfg.decel_arc)

    def set_ee_smooth(self, p, q_wxyz, jr: dict, cap: EpisodeCapture | None = None) -> float:
        """Rate-limited pose command: when the IK solution moves any joint
        faster than the joint rate, the step is subdivided in joint space
        (time-scaling; the path shape is preserved). Returns the residual
        max joint delta (rad) between issued and IK target; the caller does
        the final update of the step."""
        robot = self.robot
        robot.set_end_effector_pose(p, q_wxyz)
        tgt = np.asarray(robot.get_dof_position_targets()).ravel().copy()
        if jr["q"] is None:
            jr["q"] = tgt[:6].copy()
        dq = tgt[:6] - jr["q"]
        n_sub = max(1, int(np.ceil(np.max(np.abs(dq)) / self.joint_rate)))
        q0 = jr["q"].copy()
        for i in range(1, n_sub + 1):
            jr["q"] = q0 + dq * (i / n_sub)
            cmd = tgt.copy()
            cmd[:6] = jr["q"]
            robot.set_dof_position_targets(cmd.reshape(1, -1))
            if i < n_sub:
                self.update(cap)
        return float(np.max(np.abs(tgt[:6] - jr["q"])))

    def _align_rot(self, a, b):
        """Minimal rotation taking unit vector a onto unit vector b."""
        R = self.Rotation
        a = a / (np.linalg.norm(a) + 1e-12)
        b = b / (np.linalg.norm(b) + 1e-12)
        v = np.cross(a, b)
        c = float(np.dot(a, b))
        s = float(np.linalg.norm(v))
        if s < 1e-8:
            if c > 0:
                return R.identity()
            perp = np.cross(a, [1.0, 0, 0])
            if np.linalg.norm(perp) < 1e-6:
                perp = np.cross(a, [0, 1.0, 0])
            return R.from_rotvec(np.pi * perp / np.linalg.norm(perp))
        return R.from_rotvec(v / s * np.arctan2(s, c))

    def move_ee_blend(self, p0, q0, p_ctrl, p1, q1, probes, cap=None):
        """Blended reach: quadratic Bezier from ``p0`` to the pre-grasp
        (``standoff`` short of ``p1`` on the approach axis, control point
        ``ctrl_back`` behind it on the same axis) so the curve arrives already
        pointing down the approach axis, then a straight constant-speed entry
        into ``p1``. Orientation slerps over the first ``blend_rot_frac`` of
        the curve. Returns a reject reason or None."""
        cfg = self.cfg
        R, Slerp = self.Rotation, self.Slerp
        app = p1 - p_ctrl
        an = float(np.linalg.norm(app))
        if cfg.standoff > 0 and an > 1e-6:
            axis = app / an
            stand = min(cfg.standoff, 0.8 * an)
            p_end = p1 - axis * stand
            p_ctrl = p_end - axis * cfg.ctrl_back
        else:
            stand = 0.0
            p_end = p1

        def bez(t):
            return ((1 - t) ** 2) * p0 + 2 * (1 - t) * t * p_ctrl + (t ** 2) * p_end

        def bez_tan(t):
            dv = 2 * (1 - t) * (p_ctrl - p0) + 2 * t * (p_end - p_ctrl)
            n = np.linalg.norm(dv)
            return dv / n if n > 1e-9 else (p1 - p0) / np.linalg.norm(p1 - p0)

        grasp_ax = q1.as_matrix()[:, 0]
        ts = np.linspace(0, 1, 400)
        pts = np.stack([bez(t) for t in ts])
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        length = float(seg.sum())
        step_nom = STEP_NOMINAL_M * cfg.speed_scale
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        sl = Slerp([0, 1], R.concatenate([q0, q1]))
        s_sched = [0.0]
        while s_sched[-1] < length:
            s_sched.append(min(length, s_sched[-1] + step_nom
                               * self.curve_step_scale(length - s_sched[-1])))
        jr = {"q": None}
        for k, s_arc in enumerate(s_sched):
            t = float(np.interp(s_arc, cum, ts))
            a = min(1.0, t / max(cfg.blend_rot_frac, 1e-6))
            if cfg.tangent_orient:
                R_tan = self._align_rot(grasp_ax, bez_tan(t)) * q1
                q = Slerp([0, 1], R.concatenate([q0, R_tan]))(a).as_quat()
            else:
                q = sl(a).as_quat()
            self.set_ee_smooth(bez(t), np.array([q[3], q[0], q[1], q[2]]), jr, cap)
            self.update(cap)
            if self.contact_mon.contacting_bodies():
                return "tool_contact"
            if k % 2 == 0:
                if self.scene.clearance_violations(self.tool_rb, probes, phase="reach"):
                    return "occluder_interference"
                if self.tool_disturbed():
                    return "interference_pose"
        if stand > 0:
            qg = q1.as_quat()
            qcmd = np.array([qg[3], qg[0], qg[1], qg[2]])
            n2 = max(2, int(stand / (step_nom * cfg.entry_speed)))
            for k in range(n2 + 1):
                pk = p_end + (p1 - p_end) * (k / n2)
                self.set_ee_smooth(pk, qcmd, jr, cap)
                self.update(cap)
                if self.contact_mon.contacting_bodies():
                    return "tool_contact"
                if k % 2 == 0 and self.tool_disturbed():
                    return "interference_pose"
        # finalize: the rate limiter may still be mid-way to the final IK
        # solution; keep stepping the command until it converges (open loop)
        qg = q1.as_quat()
        qf = np.array([qg[3], qg[0], qg[1], qg[2]])
        for _ in range(60):
            res = self.set_ee_smooth(p1, qf, jr, cap)
            self.update(cap)
            if self.contact_mon.contacting_bodies():
                return "tool_contact"
            if res < 1e-4:
                break
        for _ in range(30):
            self.update(cap)
            if self.contact_mon.contacting_bodies():
                return "tool_contact"
        return None

    def move_ee(self, p0, q0, p1, q1, probes, cap=None):
        """Straight constant-speed reach with orientation slerp."""
        dist = float(np.linalg.norm(p1 - p0))
        n_steps = max(2, int(dist / STEP_NOMINAL_M))
        sl = self.Slerp([0, 1], self.Rotation.concatenate([q0, q1]))
        for k in range(n_steps + 1):
            a = k / n_steps
            q = sl(a).as_quat()
            self.robot.set_end_effector_pose(p0 + a * (p1 - p0),
                                             np.array([q[3], q[0], q[1], q[2]]))
            self.update(cap)
            if self.contact_mon.contacting_bodies():
                return "tool_contact"
            if k % 2 == 0 and self.tool_disturbed():
                return "interference_pose"
            if k % 2 == 0 and self.scene.clearance_violations(self.tool_rb, probes, phase="reach"):
                return "occluder_interference"
        for j in range(30):
            self.update(cap)
            if self.contact_mon.contacting_bodies():
                return "tool_contact"
            if j % 2 == 0 and self.scene.clearance_violations(self.tool_rb, probes, phase="reach"):
                return "occluder_interference"
        return None

    # ---------------------------------------------------------- try
    def begin_try(self, Rg_mat, grasp_w, cand, tool_pos, tool_quat):
        """Reset, re-present the tool and compute the grasp frames.
        Returns (R_grasp, grasp_ee, pre_ee, ee_p, R_now)."""
        R = self.Rotation
        R_grasp = R.from_matrix(Rg_mat)
        approach = Rg_mat[:, 0]
        grasp_ee = grasp_w - R_grasp.apply(EE_TO_TCP)
        pre_ee = grasp_ee - approach * self.standoff_of(cand)
        self.reset_arm()
        self.rehold(tool_pos, tool_quat)
        qp = np.asarray(tool_quat).ravel()
        self.presented["p"] = np.asarray(tool_pos, float).copy()
        self.presented["R"] = R.from_quat([qp[1], qp[2], qp[3], qp[0]])
        _d, ee_p, ee_q = self.robot.get_current_state()
        ee_p = np.asarray(ee_p).ravel()[:3]
        eq = np.asarray(ee_q).ravel()
        R_now = R.from_quat([eq[1], eq[2], eq[3], eq[0]])
        return R_grasp, grasp_ee, pre_ee, ee_p, R_now

    def settle_at_grasp(self, grasp_ee, R_grasp, cap=None):
        """Hold the grasp pose for 40 updates; returns (reach_err_m, ee_now)."""
        qg = R_grasp.as_quat()
        for _ in range(40):
            self.robot.set_end_effector_pose(grasp_ee, np.array([qg[3], qg[0], qg[1], qg[2]]))
            self.update(cap)
        _d, ee_now, _q = self.robot.get_current_state()
        ee_now = np.asarray(ee_now).ravel()[:3]
        return float(np.linalg.norm(ee_now - grasp_ee)), ee_now

    def close_press_release(self, width_ref: float, probes, tool_pos, tool_quat, cap=None):
        """Quick close until both pads contact, press to 0 for 120 updates,
        width check, hand release, 90 updates settle. Returns
        (reject_or_None, measurements)."""
        R = self.Rotation
        tgt, contact_q, k = 0.044, None, 0
        while tgt > 0.0:
            tgt -= 0.0009  # ~54 mm/s sim: quick close, the press follows
            self.robot.set_gripper_position(max(tgt, 0.0))
            self.update(cap)
            k += 1
            if k % 5 == 0 and self.scene.clearance_violations(self.tool_rb, probes, phase="close"):
                return "close_interference", {}
            touching = self.contact_mon.contacting_bodies()
            if ("/World/wxai/carriage_left" in touching
                    and "/World/wxai/carriage_right" in touching):
                contact_q = float(np.asarray(self.robot.get_dof_positions()).ravel()[-2])
                break
        if contact_q is None:
            return "no_clamp", {}
        self.robot.set_gripper_position(0.0)
        for _ in range(120):
            self.update(cap)
        qc = np.asarray(self.robot.get_dof_positions()).ravel()[-2:]
        gap = self.scene.finger_base_gap() + float(qc.sum())
        if abs(gap - width_ref) > 0.0012:
            return "width_mismatch", {}

        p_before = np.asarray(self.tool_rb.get_world_poses()[0]).ravel()
        self.hand.release()
        for _ in range(90):
            self.update(cap)
        p_after, q_after = self.tool_rb.get_world_poses()
        p_after = np.asarray(p_after).ravel()
        drop = float((p_before[2] - p_after[2]) * 1000)
        qa = np.asarray(q_after).ravel()
        R_now = R.from_quat([qa[1], qa[2], qa[3], qa[0]])
        qp = np.asarray(tool_quat).ravel()
        R_pres = R.from_quat([qp[1], qp[2], qp[3], qp[0]])
        pose_t = float(np.linalg.norm(p_after - np.asarray(tool_pos))) * 1000
        pose_r = float(np.degrees((R_now.inv() * R_pres).magnitude()))
        return None, {"gap_m": gap, "drop_mm": drop, "pose_t_mm": round(pose_t, 1),
                      "pose_r_deg": round(pose_r, 1)}

    def run_candidate(self, cand, Rg_mat, grasp_w, probes, tool_pos, tool_quat, cap=None):
        """One try: reach -> close -> press -> release. Returns (ok, info)."""
        cfg = self.cfg
        R_grasp, grasp_ee, pre_ee, ee_p, R_now = self.begin_try(
            Rg_mat, grasp_w, cand, tool_pos, tool_quat)
        if cfg.blend:
            why = self.move_ee_blend(ee_p, R_now, pre_ee, grasp_ee, R_grasp, probes, cap)
            if why:
                return False, {"reject": why}
        else:
            why = self.move_ee(ee_p, R_now, pre_ee, R_grasp, probes, cap)
            if why:
                return False, {"reject": why}
            why = self.move_ee(pre_ee, R_grasp, grasp_ee, R_grasp, probes, cap)
            if why:
                return False, {"reject": why}
        err, ee_now = self.settle_at_grasp(grasp_ee, R_grasp, cap)
        if err > 0.012:
            return False, {"reject": "unreachable"}
        # grasp-point offset from the pad midline along the closing axis
        tcp_now = ee_now + R_grasp.apply(EE_TO_TCP)
        center_off_mm = float(abs((grasp_w - tcp_now) @ Rg_mat[:, 1])) * 1000
        why, m = self.close_press_release(cand["width"], probes, tool_pos, tool_quat, cap)
        if why:
            return False, {"reject": why}
        info = {"gap_mm": m["gap_m"] * 1000, "reach_err_mm": err * 1000,
                "center_off_mm": round(center_off_mm, 1),
                "drop_mm": m["drop_mm"], "grasp_w": grasp_w.tolist(),
                "R_grasp": Rg_mat.tolist(),
                "pose_t_mm": m["pose_t_mm"], "pose_r_deg": m["pose_r_deg"]}
        if cfg.keep_mode == "pose":
            ok = m["drop_mm"] < 20.0 and m["pose_t_mm"] < 10.0 and m["pose_r_deg"] < 10.0
            if not ok:
                info["reject"] = "pose_delta"
            return ok, info
        return m["drop_mm"] < 20.0, info

    # ---------------------------------------------------------- picker
    def make_gates(self, sd: SceneDraw, *, height_picker: bool, gate_m: float,
                   upright_min: float):
        """Geometric candidate gates for this scene: upright, camera-side
        axis, closing axis vs plate normal (height picker), grasp height
        above the table, hand keep-out. Returns gates(c) -> (Rg, gw) | None."""
        R_tool, tool_pos = sd.R_tool, sd.tool_pos
        hand_subs = sd.spec["hand"]["subs"]
        R_tool_m = R_tool.as_matrix()

        def gates(c):
            Rg = R_tool_m @ c["R"]
            if Rg[2, 0] > 0.3 or Rg[2, 2] < upright_min:
                return None
            if height_picker:
                n_pl = R_tool.apply([0, 1.0, 0])
                if abs(float(Rg[:, 1] @ n_pl)) < 0.966:   # >15 deg off the plate normal
                    return None
            gw = tool_pos + R_tool.apply(c["center"])
            if gw[2] < 0.08:
                return None
            for s in hand_subs:
                if (np.linalg.norm(c["center"] - np.asarray(s["pos"]))
                        < max(s["half"]) + gate_m):
                    return None
            return Rg, gw
        return gates

    def make_corridor(self, sd: SceneDraw):
        """Corridor clearance: min distance of 8 points from pre-grasp to
        grasp to the (6 mm inflated) occluder sub-shapes, tool frame."""
        R = self.Rotation
        R_tool, tool_pos = sd.R_tool, sd.tool_pos
        subs = sd.spec["hand"]["subs"] + sd.spec["tip"]["subs"]

        def corridor(c, Rg, gw):
            approach = Rg[:, 0]
            pre = gw - approach * self.standoff_of(c)
            pts_w = pre + np.linspace(0, 1, 8)[:, None] * (gw - pre)
            pts_t = R_tool.inv().apply(pts_w - tool_pos)
            mind = np.inf
            for s in subs:
                sq = np.asarray(s["quat_wxyz"])
                R_s = R.from_quat([sq[1], sq[2], sq[3], sq[0]])
                local = R_s.inv().apply(pts_t - np.asarray(s["pos"]))
                d = np.linalg.norm(
                    np.maximum(np.abs(local) - (np.asarray(s["half"]) + 0.006), 0.0), axis=1)
                mind = min(mind, float(d.min()))
            return mind
        return corridor

    def build_pool(self, sd: SceneDraw, *, height_picker: bool, tries: int,
                   upright_min: float, z_lo: float, z_hi: float, z_strict: bool):
        """Gated, ranked candidate pool for the silent search.

        height picker: clearance is pass/fail (gate 1.0 cm, corridor 1.5 cm),
        ranked by distance to a per-scene target height z* ~ U(z_lo, z_hi);
        clearance picker: gate 1.6 cm, corridor 2.5 cm, ranked by corridor
        clearance, the scan stops after ``tries`` survivors.
        Returns (pool [(clearance, cand, Rg, gw)], gates, corridor, corr_th).
        """
        rng = sd.rng
        gate_m = 0.010 if height_picker else 0.016
        corr_th = 0.015 if height_picker else 0.025
        z_target = rng.uniform(z_lo, z_hi) if height_picker else None
        gates = self.make_gates(sd, height_picker=height_picker, gate_m=gate_m,
                                upright_min=upright_min)
        corridor = self.make_corridor(sd)
        pool = []
        for group in (self.flat, self.anti):
            for j in rng.permutation(len(group)):
                g = gates(group[j])
                if g is not None:
                    cd = corridor(group[j], g[0], g[1])
                    if cd > corr_th:
                        pool.append((cd, group[j], *g))
            if not height_picker and len(pool) >= tries:
                break
        if height_picker:
            pool.sort(key=lambda t: abs(t[1]["center"][2] - z_target))
            if z_strict:
                # near the hand the gates often reject every in-band candidate
                # and the nearest survivor would spill into the adjacent bin
                pool = [t for t in pool if z_lo <= t[1]["center"][2] <= z_hi]
        else:
            pool.sort(key=lambda t: -t[0])
        return pool, gates, corridor, corr_th

    def match_stored_winner(self, meta_path: str | os.PathLike, sd: SceneDraw):
        """Match a previously validated grasp (``info.grasp_w`` / ``R_grasp``
        of a stored episode) to a candidate of the database. Returns
        (cand, Rg0, gw0, center_dist_m, stored_info) or None when no
        candidate lies within 5 mm."""
        om = np.load(meta_path, allow_pickle=True)
        oi = json.loads(str(om["info"]))
        gw0 = np.asarray(oi["grasp_w"], float)
        Rg0 = np.asarray(oi["R_grasp"], float)
        c_t = sd.R_tool.inv().apply(gw0 - sd.tool_pos)
        R_tool_m = sd.R_tool.as_matrix()
        best = None
        for c in self.flat + self.anti:
            dd = float(np.linalg.norm(np.asarray(c["center"]) - c_t))
            dR = float(np.linalg.norm((R_tool_m @ c["R"]) - Rg0))
            sc = dd + 0.02 * dR
            if best is None or sc < best[0]:
                best = (sc, c, dd)
        if best is None or best[2] >= 0.005:
            return None
        return best[1], Rg0, gw0, best[2], oi

    # ---------------------------------------------------------- output
    def run_battery(self):
        from simretract.sim.battery import run_battery

        return run_battery(self.app, self.tool_rb, self.hand, self.stage, self.tip_local)

    def save_meta(self, ep_dir: str, cap: EpisodeCapture, sd: SceneDraw, info: dict,
                  verdict: dict, mass_g: float) -> None:
        R = self.Rotation
        grasp_pose = np.concatenate([
            np.asarray(info["grasp_w"]),
            R.from_matrix(np.asarray(info["R_grasp"])).as_quat()[[3, 0, 1, 2]]]).astype(np.float32)
        np.savez(os.path.join(ep_dir, "meta.npz"),
                 states=np.asarray(cap.states), actions=np.asarray(cap.actions),
                 tool_pos_t=np.asarray(cap.tool_ps), tool_quat_t=np.asarray(cap.tool_qs),
                 grasp_pose=grasp_pose, seed=sd.seed, task=TASK, fps=25, tool=self.cfg.tool,
                 mass_g=mass_g, verdict=json.dumps(verdict), info=json.dumps(info),
                 lighting=json.dumps(sd.light_spec), occ_z=sd.spec["hand"]["z_along_tool"],
                 tilt_deg=sd.tilt_deg)


# ----------------------------------------------------------------- helpers
def emit(row: dict[str, Any]) -> None:
    """One outcome line the campaign driver parses."""
    print(f"{OUTCOME_TAG} {json.dumps(row)}", flush=True)


def locked_counter(path: str | os.PathLike) -> int:
    """Increment a shared counter file under an exclusive lock; returns the
    new value (parallel workers share one review sampling sequence)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        n = int((fh.read().strip() or "0")) + 1
        fh.seek(0)
        fh.truncate()
        fh.write(str(n))
        fcntl.flock(fh, fcntl.LOCK_UN)
    return n


def review_bin(z_tool: float) -> str:
    for edge, name in REVIEW_BINS:
        if edge is None or z_tool < edge:
            return name
    return REVIEW_BINS[-1][1]


def maybe_review(out_root: str, ep_dir: str, counter_name: str, mp4_name: str, every: int) -> None:
    """Write a review video for every ``every``-th call (shared counter)."""
    if every <= 0:
        return
    review_dir = os.path.join(out_root, "review")
    n = locked_counter(os.path.join(review_dir, counter_name))
    if (n - 1) % every == 0:
        from simretract.sim.review import make_review_grid

        mp4 = os.path.join(review_dir, mp4_name)
        make_review_grid(ep_dir, mp4)
        print(f"{OUTCOME_TAG} review grid -> {mp4}", flush=True)


def install_watchdog() -> None:
    def _alarm(_s, _f):
        raise EpisodeStalled("episode stalled")
    signal.signal(signal.SIGALRM, _alarm)


def arm_watchdog(seconds: int) -> None:
    signal.alarm(int(seconds) if seconds > 0 else 0)


def episode_done(ep_dir: str) -> bool:
    return (os.path.isfile(os.path.join(ep_dir, "meta.npz"))
            or os.path.isfile(ep_dir + ".filtered") or os.path.isfile(ep_dir + ".wedged"))


def parse_seed_list(text: str) -> list[int]:
    return [int(s) for s in text.replace(" ", "").split(",") if s]


def write_log(out_root: str, log_name: str, results: list[dict]) -> None:
    with open(os.path.join(out_root, log_name), "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


def finish(results: list[dict], out_root: str, log_name: str) -> None:
    """Summary + DONE line, then a hard exit: Kit's teardown can hang."""
    n_rec = sum(1 for r in results if r.get("outcome") == "recorded")
    print(f"{OUTCOME_TAG} SUMMARY recorded={n_rec}/{len(results)}", flush=True)
    write_log(out_root, log_name, results)
    print(DONE_LINE, flush=True)
    sys.stdout.flush()
    os._exit(0)


# ----------------------------------------------------------------- main loop
def record_seed(session: SimSession, seed: int, out_root: str,
                stored_winners: str | None) -> dict[str, Any]:
    """Search and record one seed; emits and returns the outcome row."""
    cfg = session.cfg
    sd = session.draw_scene(seed)
    row = sd.row()
    ep_dir = os.path.join(out_root, f"ep_{seed}")

    def done(r: dict[str, Any]) -> dict[str, Any]:
        emit(r)
        return r

    if session.scene_rejected(sd):
        row["outcome"] = "scene_reject"
        return done(row)

    height = cfg.picker == "height"
    pool, gates, corridor, corr_th = session.build_pool(
        sd, height_picker=height, tries=cfg.tries, upright_min=cfg.upright_min,
        z_lo=cfg.z_lo, z_hi=cfg.z_hi, z_strict=cfg.z_strict)

    winner = None
    if stored_winners:
        try:
            m = session.match_stored_winner(os.path.join(stored_winners, f"ep_{seed}", "meta.npz"), sd)
            if m is not None:
                c, Rg0, gw0, dist, _oi = m
                # re-validate against the CURRENT hand: a stored grasp's hand
                # clearance does not transfer between hand distributions
                g = gates(c)
                if g is not None and corridor(c, g[0], g[1]) > corr_th:
                    winner = (c, Rg0, gw0)
                    print(f"{OUTCOME_TAG} stored winner matched (d={dist * 1000:.1f}mm)", flush=True)
                else:
                    print(f"{OUTCOME_TAG} stored winner fails hand gates on current hand "
                          "-> fresh search", flush=True)
        except Exception as e:  # noqa: BLE001  (fall back to the search)
            print(f"{OUTCOME_TAG} stored-winner load failed: {e!r}", flush=True)
    if winner is None:
        for _cd, cand, Rg, gw in pool[:cfg.tries]:
            ok, info = session.run_candidate(cand, Rg, gw, sd.probes, sd.tool_pos, sd.tool_quat)
            if ok:
                winner = (cand, Rg, gw)
                break
    if winner is None:
        row["outcome"] = "no_grasp"
        return done(row)

    # recorded pass: the same checks re-run live
    cap = EpisodeCapture(session, ep_dir)
    try:
        ok, info = session.run_candidate(*winner, sd.probes, sd.tool_pos, sd.tool_quat, cap=cap)
    finally:
        cap.close()
    if not ok:
        row["outcome"] = f"record_divergence:{info.get('reject', 'dropped')}"
        if info.get("reject") in ("pose_delta", "interference_pose"):
            row["pose_t_mm"] = info.get("pose_t_mm")
            row["pose_r_deg"] = info.get("pose_r_deg")
            shutil.rmtree(ep_dir, ignore_errors=True)
            open(ep_dir + ".filtered", "w").close()   # deterministic: never retry
        return done(row)

    verdict = {"battery_skipped": True} if cfg.skip_battery else session.run_battery()
    session.save_meta(ep_dir, cap, sd, info, verdict, row["mass_g"])
    row.update({"outcome": "recorded", "frames": cap.t,
                "gap_mm": round(info["gap_mm"], 2), "verdict": verdict})
    done(row)
    zbin = review_bin(float(winner[0]["center"][2]))
    maybe_review(out_root, ep_dir, f"bin_{zbin}.count", f"ep_{seed}_{zbin}_review.mp4",
                 cfg.review_every)
    return row


def run(cfg: RecorderConfig, seeds: list[int], out_root: str, stored_winners: str | None) -> None:
    """Boot Isaac Sim and record ``seeds`` into ``out_root``."""
    require_eula()
    os.makedirs(out_root, exist_ok=True)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    session = SimSession(app, cfg)
    install_watchdog()
    results: list[dict] = []
    for seed in seeds:
        ep_dir = os.path.join(out_root, f"ep_{seed}")
        if episode_done(ep_dir):
            print(f"{OUTCOME_TAG} seed {seed} already done - skipping", flush=True)
            continue
        if os.path.isdir(ep_dir):  # stale partial from a killed run
            shutil.rmtree(ep_dir, ignore_errors=True)
        arm_watchdog(cfg.episode_timeout_s)
        try:
            row = record_seed(session, seed, out_root, stored_winners)
        except EpisodeStalled:
            shutil.rmtree(ep_dir, ignore_errors=True)
            open(ep_dir + ".wedged", "w").close()
            row = {"seed": seed, "outcome": "episode_timeout"}
            emit(row)
        finally:
            arm_watchdog(0)
        results.append(row)
    finish(results, out_root, f"recorder_log_{seeds[0] if seeds else 'empty'}.jsonl")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Expert episode recorder (runs inside Isaac Sim).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out", required=True, help="raw episode root (ep_<seed>/ dirs are written here)")
    ap.add_argument("--seeds", default=os.environ.get("SEED_LIST", ""),
                    help="comma-separated seed list (env SEED_LIST)")
    ap.add_argument("--seed-start", type=int, default=int(os.environ.get("SEED0", "0")),
                    help="first seed when --seeds is empty (env SEED0)")
    ap.add_argument("--n-seeds", type=int, default=int(os.environ.get("SEEDS", "0")),
                    help="number of consecutive seeds from --seed-start when --seeds is empty (env SEEDS)")
    ap.add_argument("--stored-winners", default=os.environ.get("STORED_WINNER_ROOT") or None,
                    help="episode root whose ep_<seed>/meta.npz grasps are reused (re-validated "
                         "against the current hand) instead of a physical search (env STORED_WINNER_ROOT)")
    RecorderConfig.add_args(ap)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = RecorderConfig.from_args(args)
    seeds = parse_seed_list(args.seeds) or [args.seed_start + i for i in range(args.n_seeds)]
    if not seeds:
        raise SystemExit("no seeds: pass --seeds or --seed-start/--n-seeds")
    run(cfg, seeds, args.out, args.stored_winners)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
