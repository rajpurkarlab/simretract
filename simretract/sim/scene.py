"""Scene assembly for the simulated handover (import only inside Isaac Sim).

Everything the recorders and the sim benchmark share: physics scene, light
rig with per-episode randomisation, ground, the follower arm with the
flat-pad fingers swapped in, the wrist camera intrinsics, the tool as a
dynamic rigid body, the pooled occluder prims (hand + tissue tip), contact
reporting, the compliant "operator hand" that presents the tool, and a few
one-off render helpers.

``pxr`` / ``omni`` modules only exist after ``SimulationApp`` has been
created, so import this module lazily from a ``main()`` that boots Kit first.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from simretract.sim.sampling import sample_occluder_spec  # noqa: F401  (re-export)
from simretract.sim.tool_usd import assets_dir

log = logging.getLogger(__name__)


def robot_usd_path() -> Path:
    """Follower-arm USD shipped with the vendor repository (submodule)."""
    from simretract import config

    return Path(config.trossen_robot_usd())


def import_controller():
    """Import the vendor ``controller`` module (TrossenAIController, RobotType).

    The vendor repository is not a package, so ``simretract.config`` puts its
    scripts directory on ``sys.path`` once. Returns the module.
    """
    from simretract import config

    config.ensure_trossen_on_path()
    import controller  # noqa: WPS433  (vendor module)

    return controller


GRIPPER_DIR = assets_dir() / "gripper"
WRIST_CAM_REL = "camera_link/camera_color_frame/Camera"
GRIPPER_RGBA = (0.07, 0.07, 0.075)

# link_6-frame TCP (fingertip-pad midpoint)
TCP_OFFSET_LINK6 = np.array([0.1487, 0.0, 0.0])

PART_COLORS = [
    (0.90, 0.30, 0.30), (0.30, 0.75, 0.35), (0.25, 0.45, 0.95),
    (0.95, 0.75, 0.20), (0.70, 0.35, 0.85), (0.25, 0.80, 0.80),
    (0.95, 0.50, 0.20), (0.55, 0.75, 0.25), (0.85, 0.35, 0.60),
    (0.40, 0.55, 0.75), (0.75, 0.65, 0.45), (0.60, 0.80, 0.55),
]


def setup_physics_scene(stage, hz: float = 240.0, cpu: bool = True):
    """Physics scene. CPU PhysX by default: the scene is small, contacts
    between the articulation and the tool are more reliable there than on
    the GPU pipeline, and CPU is friendlier to determinism."""
    scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
    api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    api.CreateTimeStepsPerSecondAttr(hz)
    if cpu:
        api.CreateEnableGPUDynamicsAttr(False)
        api.CreateBroadphaseTypeAttr("MBP")
    else:
        api.CreateGpuFoundLostAggregatePairsCapacityAttr(10 * 1024)
    return scene


def add_lights(stage, dome: float = 1000.0, distant: float | None = 3000.0) -> None:
    """Default dome + distant light; ``randomize_lighting`` re-configures the
    same prims per episode."""
    d = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    d.CreateIntensityAttr(dome)
    dl = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    dl.CreateIntensityAttr(3000.0 if distant is None else distant)


_LIGHT_POOL: dict = {}


def _ensure_light_pool(stage) -> None:
    """Fixed light rig configured per episode (prims are never deleted live)."""
    if _LIGHT_POOL:
        return
    d = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    key = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    fills = []
    for i in range(3):
        s = UsdLux.SphereLight.Define(stage, f"/World/FillLight_{i}")
        s.CreateRadiusAttr(0.10)
        xf = UsdGeom.Xformable(s)
        xf.ClearXformOpOrder()
        t = xf.AddTranslateOp()
        t.Set(Gf.Vec3d(0, 0, 2))
        fills.append((s, t))
    _LIGHT_POOL.update({"dome": d, "key": key, "fills": fills})


def randomize_lighting(stage, rng: np.random.Generator) -> dict:
    """Per-episode lighting randomisation: dome intensity/tint, key light
    direction/intensity/tint/softness, 0-3 coloured fill lights. Returns the
    drawn spec (stored in the episode meta)."""
    _ensure_light_pool(stage)
    warmth = float(rng.uniform(-1, 1))  # -1 cool .. +1 warm
    tint = np.clip(np.array([1.0 + 0.18 * warmth, 1.0,
                             1.0 - 0.22 * warmth]), 0, 1)
    dome_i = float(rng.uniform(250, 1800))
    d = _LIGHT_POOL["dome"]
    d.CreateIntensityAttr(dome_i)
    d.CreateColorAttr(Gf.Vec3f(*tint))

    key = _LIGHT_POOL["key"]
    key_i = float(rng.uniform(400, 6000))
    elev = float(rng.uniform(20, 75))
    azim = float(rng.uniform(0, 360))
    key.CreateIntensityAttr(key_i)
    key.CreateColorAttr(Gf.Vec3f(*np.clip(
        tint + rng.uniform(-0.08, 0.08, 3), 0, 1)))
    xf = UsdGeom.Xformable(key)
    xf.ClearXformOpOrder()
    xf.AddRotateXYZOp().Set(Gf.Vec3f(float(-elev), 0.0, float(azim)))
    # soft vs hard shadows: distant-light angle
    key.CreateAngleAttr(float(rng.uniform(0.2, 8.0)))

    n_fill = int(rng.integers(0, 4))
    spec = {"dome_i": dome_i, "warmth": warmth, "key_i": key_i,
            "key_elev": elev, "key_azim": azim, "n_fill": n_fill}
    for i, (s, t) in enumerate(_LIGHT_POOL["fills"]):
        if i < n_fill:
            r = float(rng.uniform(0.8, 1.8))
            a = float(rng.uniform(0, 2 * np.pi))
            z = float(rng.uniform(0.6, 1.8))
            t.Set(Gf.Vec3d(0.35 + r * np.cos(a), r * np.sin(a), z))
            s.CreateIntensityAttr(float(rng.uniform(2000, 30000)))
            s.CreateColorAttr(Gf.Vec3f(*np.clip(
                np.array([1, 1, 1]) + rng.uniform(-0.25, 0.25, 3), 0, 1)))
            UsdGeom.Imageable(s).MakeVisible()
        else:
            UsdGeom.Imageable(s).MakeInvisible()
    return spec


def add_ground(stage, z_top: float = 0.0, color=(0.45, 0.45, 0.48)) -> None:
    g = UsdGeom.Cube.Define(stage, "/World/ground")
    xf = UsdGeom.Xformable(g)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, z_top - 0.1))
    xf.AddScaleOp().Set(Gf.Vec3d(2.0, 2.0, 0.1))
    UsdPhysics.CollisionAPI.Apply(g.GetPrim())
    g.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def author_trimesh(stage, path: str, tm, color, smooth_normals: bool = False):
    """Author a trimesh as a UsdGeom.Mesh with inline points (no file refs)."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in tm.vertices])
    mesh.CreateFaceVertexCountsAttr([3] * len(tm.faces))
    mesh.CreateFaceVertexIndicesAttr(tm.faces.ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    if smooth_normals:
        mesh.CreateNormalsAttr([Gf.Vec3f(*n) for n in tm.vertex_normals])
        mesh.SetNormalsInterpolation("vertex")
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    b = tm.bounds
    mesh.CreateExtentAttr([Gf.Vec3f(*b[0]), Gf.Vec3f(*b[1])])
    return mesh


def make_physics_material(stage, path: str, friction: float, restitution: float = 0.0):
    pmat = UsdShade.Material.Define(stage, path)
    mapi = UsdPhysics.MaterialAPI.Apply(pmat.GetPrim())
    mapi.CreateStaticFrictionAttr(friction)
    mapi.CreateDynamicFrictionAttr(friction)
    mapi.CreateRestitutionAttr(restitution)
    return pmat


# Base gap between the pad clamp faces at qpos 0 with zero inset (CAD +
# carriage geometry: 2 * (0.023 - 0.02216)). The pads are inset so the fully
# closed face gap is ~1 mm once the measured ~0.8 mm carriage residual at
# full close is accounted for: 1.68 + 0.84 - 1.0 = 1.52 mm total inset.
FINGER_BASE_GAP_RAW = 0.00168
PAD_INSET = 0.00076  # per side, baked into the mesh vertices (not xform ops)


def finger_base_gap(pad_inset: float = PAD_INSET) -> float:
    return FINGER_BASE_GAP_RAW - 2.0 * pad_inset


def swap_fingers(stage, robot_path: str = "/World/wxai", friction: float = 0.8,
                 pad_inset: float = PAD_INSET) -> dict:
    """Disable the stock finger geometry and mount the flat-pad fingers on
    the carriages.

    Meshes attach CAD-aligned to the carriage bolt pattern plus a
    ``pad_inset`` shim toward the centreline (baked into the vertices so the
    physics parser sees identity transforms). Returns a report dict.
    """
    report = {}
    pmat = make_physics_material(stage, robot_path + "/finger_phys_mat", friction)

    # matte black finger material (non-reflective, like the real pads)
    fmat = UsdShade.Material.Define(stage, robot_path + "/finger_black_mat")
    fshader = UsdShade.Shader.Define(stage,
                                     robot_path + "/finger_black_mat/pbr")
    fshader.CreateIdAttr("UsdPreviewSurface")
    fshader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.015, 0.015, 0.016))
    fshader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    fshader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    fshader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
    fshader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.0, 0.0, 0.0))
    fmat.CreateSurfaceOutput().ConnectToSource(fshader.ConnectableAPI(),
                                               "surface")

    for side in ("left", "right"):
        stock = stage.GetPrimAtPath(f"{robot_path}/gripper_{side}")
        # deactivate the stock collision subtree entirely: collisionEnabled
        # False leaves the colliders active
        coll_scope = stage.GetPrimAtPath(f"{robot_path}/gripper_{side}/collisions")
        n_disabled = 0
        if coll_scope and coll_scope.IsValid():
            coll_scope.SetActive(False)
            n_disabled = 1
        UsdGeom.Imageable(stock).MakeInvisible()

        # inward shim: the left carriage sits at +y, pads face -y (and vice versa)
        inset_y = -pad_inset if side == "left" else pad_inset

        carriage = f"{robot_path}/carriage_{side}"
        vis_tm = trimesh.load(
            str(GRIPPER_DIR / f"finger_{side}_visual.obj"), force="mesh")
        vis_tm.apply_translation([0.0, inset_y, 0.0])
        vis = author_trimesh(stage, f"{carriage}/newfinger_visual", vis_tm,
                             (0.04, 0.04, 0.045), smooth_normals=True)
        UsdShade.MaterialBindingAPI.Apply(vis.GetPrim()).Bind(fmat)

        coll_dir = GRIPPER_DIR / f"collision_{side}"
        coll_files = sorted(f for f in os.listdir(coll_dir) if f.endswith(".obj"))
        UsdGeom.Scope.Define(stage, f"{carriage}/newfinger_colls")
        for i, fn in enumerate(coll_files):
            tm = trimesh.load(str(coll_dir / fn), force="mesh")
            tm.apply_translation([0.0, inset_y, 0.0])
            cpath = f"{carriage}/newfinger_colls/{os.path.splitext(fn)[0]}"
            cmesh = author_trimesh(stage, cpath, tm,
                                   PART_COLORS[i % len(PART_COLORS)])
            UsdPhysics.CollisionAPI.Apply(cmesh.GetPrim())
            mc = UsdPhysics.MeshCollisionAPI.Apply(cmesh.GetPrim())
            mc.CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
            UsdShade.MaterialBindingAPI.Apply(cmesh.GetPrim()).Bind(
                pmat, materialPurpose="physics")
            cmesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
        report[side] = {"stock_colls_disabled": n_disabled,
                        "new_colls_added": len(coll_files),
                        "pad_inset_mm": pad_inset * 1000}
    return report


def set_wrist_intrinsics(stage, robot_path: str = "/World/wxai", fovy_deg: float = 58.0,
                         aspect: float = 4.0 / 3.0) -> dict:
    """Wrist camera set to the real D405 field of view (58 degrees vertical)."""
    cam = UsdGeom.Camera(stage.GetPrimAtPath(f"{robot_path}/{WRIST_CAM_REL}"))
    f = float(cam.GetFocalLengthAttr().Get())
    v_ap = 2.0 * f * math.tan(math.radians(fovy_deg) / 2)
    h_ap = v_ap * aspect
    cam.CreateVerticalApertureAttr(v_ap)
    cam.CreateHorizontalApertureAttr(h_ap)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
    fovx = math.degrees(2 * math.atan((h_ap / 2) / f))
    return {"fovy_deg": fovy_deg, "fovx_deg": fovx, "focal_mm": f}


def spawn_tool(stage, path: str = "/World/tool", kinematic: bool = False,
               usd: str | os.PathLike | None = None):
    """Reference the tool USD (``usd`` is required) as a never-sleeping
    rigid body."""
    import isaacsim.core.experimental.utils.stage as stage_utils

    if usd is None:
        raise ValueError("spawn_tool needs the tool USD path (usd=...)")
    prim = stage_utils.add_reference_to_stage(usd_path=str(usd), path=path)
    if kinematic:
        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
    # never sleep: the hand's small capped restoring torque must always act
    # (a sleeping body freezes residual orientation errors)
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rb.CreateSleepThresholdAttr(0.0)
    rb.CreateStabilizationThresholdAttr(0.0)
    return prim


def set_pose(stage, prim_path: str, pos, quat_wxyz) -> None:
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
    w, x, y, z = [float(v) for v in quat_wxyz]
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))


def teleport(stage, prim_path: str, rigid_prim, pos, quat_wxyz) -> None:
    """Teleport a (possibly kinematic) rigid body: physics AND render pose.

    PhysX never writes kinematic bodies back to the renderer, and USD edits
    during play never reach PhysX, so both sides are written explicitly.
    """
    pos = np.asarray(pos, dtype=float)
    quat = np.asarray(quat_wxyz, dtype=float)
    rigid_prim.set_world_poses(pos.reshape(1, 3), quat.reshape(1, 4))
    set_pose(stage, prim_path, pos, quat)


def world_transform(stage, prim_path: str) -> np.ndarray:
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    m = np.array(cache.GetLocalToWorldTransform(
        stage.GetPrimAtPath(prim_path))).T
    return m  # 4x4 with translation in m[:3, 3]


_OCC_POOL: dict = {}  # path -> (gprim, translate_op, orient_op, scale_op)
_OCC_POOL_N = 4  # per group per kind


def _ensure_occluder_pool(stage, parent: str) -> str:
    """Create the fixed occluder prim pool ONCE. Prims are never deleted
    mid-session: removing prims the RTX renderer still references crashes
    the renderer."""
    root = f"{parent}/occluder"
    if _OCC_POOL:
        return root
    UsdGeom.Scope.Define(stage, root)
    from isaacsim.core.utils.semantics import add_update_semantics
    for group in ("hand", "tip"):
        for kind, cls in (("box", UsdGeom.Cube), ("sph", UsdGeom.Sphere)):
            for i in range(_OCC_POOL_N):
                path = f"{root}/{group}_{kind}_{i}"
                g = cls.Define(stage, path)
                # own class overrides the inherited "tool" label: the hand
                # must NOT appear in the tool mask (it occludes the tool)
                add_update_semantics(g.GetPrim(), "occluder")
                xf = UsdGeom.Xformable(g)
                xf.ClearXformOpOrder()
                t = xf.AddTranslateOp()
                t.Set(Gf.Vec3d(0, 0, 0))
                o = xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
                o.Set(Gf.Quatd(1, 0, 0, 0))
                s = xf.AddScaleOp()
                s.Set(Gf.Vec3d(0.001, 0.001, 0.001))
                UsdGeom.Imageable(g).MakeInvisible()
                _OCC_POOL[path] = (g, t, o, s)
    return root


def author_occluder(stage, spec: dict, parent: str = "/World/tool") -> list:
    """Configure the occluder + tip anchor from the fixed prim pool (visual
    only, riding the tool; no physics). Safe to call every episode.
    Returns the probe list [(pos, quat_wxyz, half_extents), ...]."""
    root = _ensure_occluder_pool(stage, parent)
    for g, _t, _o, _s in _OCC_POOL.values():
        UsdGeom.Imageable(g).MakeInvisible()
    probes = []
    for group in ("hand", "tip"):
        rgba = Gf.Vec3f(*spec[group]["rgba"][:3])
        counters = {"box": 0, "sph": 0}
        for sub in spec[group]["subs"]:
            kind = "box" if sub["kind"] == "box" else "sph"
            i = counters[kind]
            counters[kind] += 1
            if i >= _OCC_POOL_N:
                continue
            g, t, o, s = _OCC_POOL[f"{root}/{group}_{kind}_{i}"]
            t.Set(Gf.Vec3d(*[float(v) for v in sub["pos"]]))
            w, x, y, z = [float(v) for v in sub["quat_wxyz"]]
            o.Set(Gf.Quatd(w, x, y, z))
            s.Set(Gf.Vec3d(*[float(v) for v in sub["half"]]))
            g.CreateDisplayColorAttr([rgba])
            UsdGeom.Imageable(g).MakeVisible()
            probes.append((np.asarray(sub["pos"], dtype=float),
                           np.asarray(sub["quat_wxyz"], dtype=float),
                           np.asarray(sub["half"], dtype=float)))
    return probes


def clearance_violations(tool_rb, probes, margin: float = 0.006, phase: str = "reach") -> set:
    """Occluder clearance check (>= ``margin`` by default 6 mm).

    Each occluder sub-shape, inflated by ``margin``, is used as an oriented
    box overlap probe; any ROBOT collision shape inside is a violation.
    Fingers count during reach and are excluded during close.
    Returns the set of violating prim paths."""
    from omni.physx import get_physx_scene_query_interface
    from scipy.spatial.transform import Rotation as _R

    pos, quat = tool_rb.get_world_poses()
    tp = np.asarray(pos).ravel()
    tq = np.asarray(quat).ravel()  # wxyz
    r_tool = _R.from_quat([tq[1], tq[2], tq[3], tq[0]])
    sq = get_physx_scene_query_interface()
    bad = set()
    for lp, lq, half in probes:
        wp = tp + r_tool.apply(lp)
        r_w = r_tool * _R.from_quat([lq[1], lq[2], lq[3], lq[0]])
        qx = r_w.as_quat()  # xyzw
        hits = []

        def on_hit(hit):
            hits.append(hit.collision)
            return True

        sq.overlap_box(
            np.asarray(half + margin, dtype=np.float32),
            np.asarray(wp, dtype=np.float32),
            np.asarray(qx, dtype=np.float32), on_hit, False)
        for h in hits:
            h = str(h)
            if not h.startswith("/World/wxai"):
                continue
            if phase == "close" and ("carriage" in h or "gripper_" in h):
                continue
            bad.add(h)
    return bad


def set_contact_friction(stage, mu: float, tool_path: str = "/World/tool",
                         robot_path: str = "/World/wxai") -> None:
    """Set tool + finger physics-material friction for the whole trial."""
    for mpath in (f"{tool_path}/physics_mat", f"{robot_path}/finger_phys_mat"):
        prim = stage.GetPrimAtPath(mpath)
        if prim and prim.IsValid():
            api = UsdPhysics.MaterialAPI(prim)
            api.GetStaticFrictionAttr().Set(float(mu))
            api.GetDynamicFrictionAttr().Set(float(mu))


def set_gripper_drive(stage, robot_path: str = "/World/wxai", stiffness: float = 40000.0,
                      damping: float = 50.0, max_force: float = 120.0) -> None:
    """Gripper drive tuned for a firm flat-pad squeeze on a steel tool at
    worst-case friction: kp 40000 N/m, force cap 120 N. The close machine
    presses to target 0 (force equilibrium), so the cap bounds the squeeze.
    Call BEFORE the timeline starts."""
    joint = stage.GetPrimAtPath(f"{robot_path}/joints/left_carriage_joint")
    drive = UsdPhysics.DriveAPI.Apply(joint, "linear")
    old = (drive.GetStiffnessAttr().Get(), drive.GetDampingAttr().Get(),
           drive.GetMaxForceAttr().Get())
    drive.CreateStiffnessAttr(float(stiffness))
    drive.CreateDampingAttr(float(damping))
    drive.CreateMaxForceAttr(float(max_force))
    log.info("gripper drive: %s -> (%s, %s, %s)", old, stiffness, damping, max_force)


def close_to_contact_and_press(robot, app, contact_mon, robot_path: str = "/World/wxai",
                               rate: float = 0.00015, press_to: float = 0.0,
                               settle: int = 250) -> dict:
    """Gentle close ramp until BOTH pads contact the tool, then press to
    ``press_to`` (0 = until the tool stops the pads at the drive's force
    cap). Returns telemetry."""
    left = f"{robot_path}/carriage_left"
    right = f"{robot_path}/carriage_right"
    tgt, contact_q = 0.044, None
    while tgt > 0.0:
        tgt -= rate
        robot.set_gripper_position(max(tgt, 0.0))
        app.update()
        touching = contact_mon.contacting_bodies()
        if left in touching and right in touching:
            contact_q = float(
                np.asarray(robot.get_dof_positions()).ravel()[-2])
            break
    if contact_q is None:
        return {"contact": False}
    robot.set_gripper_position(press_to)
    qs = []
    for _ in range(settle):
        app.update()
        qs.append(np.asarray(robot.get_dof_positions()).ravel()[-2:].copy())
    qs = np.asarray(qs)
    q_final = qs[-1]
    jitter = float(qs[-100:].std(axis=0).max()) if len(qs) >= 100 else None
    return {"contact": True, "contact_qL_mm": contact_q * 1000,
            "gap_mm": (finger_base_gap() + float(q_final.sum())) * 1000,
            "qL_mm": float(q_final[0]) * 1000,
            "qR_mm": float(q_final[1]) * 1000,
            "jitter_mm": jitter * 1000 if jitter is not None else None}


class ContactMonitor:
    """Pollable contact reporting for one rigid body (the tool).

    After each physics step ``contacting_bodies()`` returns the set of
    rigid-body prim paths currently in contact with the monitored body
    (e.g. .../carriage_left). Apply BEFORE the timeline starts playing.
    """

    def __init__(self, stage, body_path: str):
        self.body_path = str(body_path)
        api = PhysxSchema.PhysxContactReportAPI.Apply(
            stage.GetPrimAtPath(self.body_path))
        api.CreateThresholdAttr(0.0)

    def contacting_bodies(self) -> set:
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools

        headers, _data = get_physx_simulation_interface().get_contact_report()
        touching = set()
        for h in headers:
            a0 = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
            a1 = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
            if a0 == self.body_path:
                touching.add(a1)
            elif a1 == self.body_path:
                touching.add(a0)
        return touching


class OperatorHand:
    """Invisible compliant 6-DoF "hand" holding the tool.

    A PD wrench toward a target pose, applied every physics step.
    Linear: stiff (k = 5000 N/m, ~0.15 mm sag under a 76 g tool).
    Rotational: progressive like a soft weld: a stiff elastic zone
    (k_rot = 8 N*m/rad, gravity sag ~0.1 deg) with the spring torque
    saturating at ``t_yield`` = 0.06 N*m, so clamp-level forces rotate the
    tool freely and it recovers when the load is removed. The tool's long
    (spin) axis has its own softer spring and yield because its inertia is
    ~200x smaller. ``hold()`` / ``release()`` toggle the hold.
    """

    def __init__(self, rigid_prim, k_lin: float = 5000.0, k_rot: float = 8.0,
                 t_yield: float = 0.06, k_spin: float = 0.02, t_yield_spin: float = 0.005,
                 lin_damp_ratio: float = 1.0, rot_damp_ratio: float = 1.0):
        from omni.physx import get_physx_interface
        from scipy.spatial.transform import Rotation as _R

        self._R = _R
        self.rp = rigid_prim
        self.k_lin = float(k_lin)
        self.k_rot = float(k_rot)          # transverse (tool x/y): tilt hold
        self.t_yield = float(t_yield)
        self.k_spin = float(k_spin)        # tool long axis: soft spin hold
        self.t_yield_spin = float(t_yield_spin)
        self._lin_damp_ratio = float(lin_damp_ratio)
        self._rot_damp_ratio = float(rot_damp_ratio)
        # gains that depend on mass/inertia are computed lazily on the first
        # physics step: tensor queries are only valid while playing
        self.kd_lin = None
        self.kd_rot = None
        self.mass = None
        self.inertia_diag = None
        self.target = None  # (pos(3), quat_wxyz(4)) or None
        # optional constant external push: (force(3) world, point_local(3))
        self.extra_push = None
        # optional constant pure torque (world, N*m): load-battery impulses
        self.extra_torque = None
        # previous-step orientation for a frame-proof numeric angular velocity
        self._prev_rot = None
        self._sub = get_physx_interface().subscribe_physics_step_events(
            self._on_step)

    def hold(self, pos, quat_wxyz) -> None:
        q = np.asarray(quat_wxyz, dtype=float)
        self.target = (np.asarray(pos, dtype=float), q / np.linalg.norm(q))
        self._prev_rot = None  # avoid a damping spike across teleports

    def release(self) -> None:
        self.target = None

    def _pose_vel(self):
        pos, quat = self.rp.get_world_poses()
        lin, ang = self.rp.get_velocities()
        return (np.asarray(pos).ravel(), np.asarray(quat).ravel(),
                np.asarray(lin).ravel(), np.asarray(ang).ravel())

    def rotation_error_deg(self) -> float:
        """Angle between current and target orientation (deg, full rotvec)."""
        if self.target is None:
            return 0.0
        pos, quat, _, _ = self._pose_vel()
        r_cur = self._R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        qt = self.target[1]
        r_tgt = self._R.from_quat([qt[1], qt[2], qt[3], qt[0]])
        return float(np.degrees(np.linalg.norm(
            (r_tgt * r_cur.inv()).as_rotvec())))

    def tilt_error_deg(self) -> float:
        """Angle between current and target tool LONG AXIS (deg)."""
        if self.target is None:
            return 0.0
        pos, quat, _, _ = self._pose_vel()
        r_cur = self._R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        qt = self.target[1]
        r_tgt = self._R.from_quat([qt[1], qt[2], qt[3], qt[0]])
        z_cur = r_cur.apply([0.0, 0.0, 1.0])
        z_tgt = r_tgt.apply([0.0, 0.0, 1.0])
        c = float(np.clip(np.dot(z_cur, z_tgt), -1.0, 1.0))
        return float(np.degrees(np.arccos(c)))

    def _ensure_gains(self) -> None:
        if self.kd_lin is not None:
            return
        m = float(np.asarray(self.rp.get_masses()).ravel()[0])
        inertias = np.asarray(self.rp.get_inertias()).reshape(-1, 9)[0]
        i_trans = float(np.mean([inertias[0], inertias[4]]))  # tool x/y
        i_spin = float(inertias[8])                           # tool long axis
        self.kd_lin = 2.0 * self._lin_damp_ratio * math.sqrt(self.k_lin * m)
        self.kd_rot = 2.0 * self._rot_damp_ratio * math.sqrt(
            self.k_rot * max(i_trans, 1e-8))
        self.kd_spin = 2.0 * self._rot_damp_ratio * math.sqrt(
            self.k_spin * max(i_spin, 1e-10))
        self.mass = m
        self.inertia_diag = i_trans
        log.info("OperatorHand gains ready: mass=%.1fg I_trans=%.2e I_spin=%.2e "
                 "kd_lin=%.2f kd_rot=%.4f kd_spin=%.5f", m * 1000, i_trans, i_spin,
                 self.kd_lin, self.kd_rot, self.kd_spin)

    def _on_step(self, dt) -> None:
        if (self.target is None and self.extra_push is None
                and self.extra_torque is None):
            return
        self._ensure_gains()
        pos, quat, lin, ang = self._pose_vel()
        r_cur = self._R.from_quat([quat[1], quat[2], quat[3], quat[0]])

        # frame-proof world angular velocity from consecutive orientations;
        # a step delta too large to be physical means a teleport: skip damping
        if self._prev_rot is not None and dt > 0:
            step_rotvec = (r_cur * self._prev_rot.inv()).as_rotvec()
            if np.linalg.norm(step_rotvec) < 0.1:
                w_world = step_rotvec / dt
            else:
                w_world = np.zeros(3)
        else:
            w_world = np.zeros(3)
        self._prev_rot = r_cur

        force = np.zeros(3)
        torque = np.zeros(3)
        if self.target is not None:
            pt, qt = self.target
            force += self.k_lin * (pt - pos) - self.kd_lin * lin
            r_tgt = self._R.from_quat([qt[1], qt[2], qt[3], qt[0]])
            # per-axis gains in the TOOL frame: the long (spin) axis has
            # far less inertia, stiff gains are explicit-unstable there
            e_local = r_cur.inv().apply((r_tgt * r_cur.inv()).as_rotvec())
            w_local = r_cur.inv().apply(w_world)
            t_local = np.zeros(3)
            # transverse (tilt) spring with progressive yield
            t_xy = self.k_rot * e_local[:2]
            m_xy = float(np.linalg.norm(t_xy))
            if m_xy > self.t_yield:
                t_xy *= self.t_yield / m_xy
            t_local[:2] = t_xy - self.kd_rot * w_local[:2]
            # soft spin hold with its own yield
            t_z = self.k_spin * e_local[2]
            t_z = float(np.clip(t_z, -self.t_yield_spin, self.t_yield_spin))
            t_local[2] = t_z - self.kd_spin * w_local[2]
            t_world = r_cur.apply(t_local)
            d_mag = float(np.linalg.norm(t_world))
            if d_mag > 0.5:  # bound one-step spikes (teleports, resets)
                t_world *= 0.5 / d_mag
            torque += t_world
        if self.extra_push is not None:
            f_ext, p_local = self.extra_push
            p_world = pos + r_cur.apply(np.asarray(p_local, dtype=float))
            force += np.asarray(f_ext, dtype=float)
            com = pos  # approximation: prim origin ~ COM (centroid-built mesh)
            torque += np.cross(p_world - com, np.asarray(f_ext, dtype=float))
        if self.extra_torque is not None:
            torque += np.asarray(self.extra_torque, dtype=float)
        self.rp.apply_forces_and_torques_at_pos(
            forces=force.reshape(1, 3), torques=torque.reshape(1, 3),
            positions=pos.reshape(1, 3))


_cam_counter = [0]


def make_camera(stage, eye, target, focal: float = 18.0) -> str:
    """Author a look-at camera under /World/cams; returns its prim path."""
    _cam_counter[0] += 1
    path = f"/World/cams/cam_{_cam_counter[0]:02d}"
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(focal)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
    view = Gf.Matrix4d()
    view.SetLookAt(Gf.Vec3d(*[float(v) for v in eye]),
                   Gf.Vec3d(*[float(v) for v in target]),
                   Gf.Vec3d(0, 0, 1))
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(view.GetInverse())
    return path


def capture(simulation_app, cam_path: str, out_path: str, res=(960, 720),
            rt_subframes: int = 16):
    """Render one RGB frame from ``cam_path`` to ``out_path`` (fresh annotator)."""
    import omni.replicator.core as rep
    from PIL import Image

    rp = rep.create.render_product(cam_path, res)
    ann = rep.AnnotatorRegistry.get_annotator("rgb")
    ann.attach(rp)
    for _ in range(10):
        simulation_app.update()
    rep.orchestrator.step(rt_subframes=rt_subframes)
    rep.orchestrator.step(rt_subframes=rt_subframes)
    data = ann.get_data()
    Image.fromarray(data[..., :3]).save(out_path)
    ann.detach()
    rp.destroy()
    return data
