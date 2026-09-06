"""Tool asset locations and the tool USD builder.

The module-level helpers are simulator-free (they only resolve paths and
read ``meta.json``) so the omni-free tools (campaign driver, assembly) can
use them. ``build_tool_usd`` authors ``assets/usd/<tool>.usd`` from
``assets/tools/<tool>/`` and must run inside Isaac Sim; ``main`` boots the
simulator itself.

The tool meshes and the USD are built locally from the user's licensed copy
of the asset (``simretract-build-assets``; ``docs/ASSETS.md``) and are never
part of the repository; only ``meta.json`` is tracked. Every build is
checked against that reference (visual extents, mesh volume, the convex
collision set) and reports ``[BUILD] check vs meta.json: OK``.

Tool USD contents: a rigid-body root with steel density, a physics
material (friction overridden per episode by the scene), the visual mesh
with a steel PBR shader, a scope of convex-hull collision parts (CoACD
output, rendered as guides only) and ``tip`` / ``bend`` marker xforms.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

STEEL_DENSITY_KG_M3 = 7900.0
DEFAULT_TOOL = "retractor_rake"

#: Tolerances of the collision-set check. CoACD splits a mesh differently
#: for sub-micrometre coordinate differences (another export of the same
#: model, another platform), so the number of convex parts is checked within
#: a few parts and the summed hull volume within a few percent.
PARTS_TOL = 3
HULL_VOLUME_REL_TOL = 0.05


def repo_root() -> Path:
    """Repository root from ``simretract.config``."""
    from simretract import config

    return Path(config.repo_root())


def assets_dir() -> Path:
    """The ``assets/`` directory (``configs/paths.yaml``)."""
    from simretract import config

    return Path(config.paths()["assets"])


def tool_dir(tool: str) -> Path:
    """``assets/tools/<tool>/`` (visual.obj, collision/, meta.json)."""
    return assets_dir() / "tools" / tool


def tool_usd_path(tool: str) -> Path:
    """Locally built USD for ``tool`` (``assets/usd/<tool>.usd``)."""
    return assets_dir() / "usd" / f"{tool}.usd"


def tool_meta(tool: str) -> dict[str, Any]:
    """Parsed ``assets/tools/<tool>/meta.json``."""
    with open(tool_dir(tool) / "meta.json") as f:
        return json.load(f)


BUILD_HINT = ("build it from your licensed copy of the asset with "
              "`simretract-build-assets --tool {tool}` (the source file is taken from "
              "assets/source/<pack>/ or the tool_source override; docs/ASSETS.md)")


def require_tool_assets(tool: str) -> Path:
    """``assets/tools/<tool>/`` with ``visual.obj``, ``collision/`` and
    ``meta.json`` present, else :class:`FileNotFoundError` naming the build command."""
    tdir = tool_dir(tool)
    missing = [n for n in ("visual.obj", "collision", "meta.json") if not (tdir / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"tool assets for {tool!r} are not built ({tdir}: missing {', '.join(missing)}); "
            + BUILD_HINT.format(tool=tool))
    return tdir


def require_tool_usd(tool: str) -> Path:
    """``assets/usd/<tool>.usd`` if it has been built, else
    :class:`FileNotFoundError` naming the build command."""
    usd = tool_usd_path(tool)
    if not usd.is_file():
        raise FileNotFoundError(f"tool USD {usd} is not built; " + BUILD_HINT.format(tool=tool))
    return usd


def tip_local(tool: str):
    """Suggested blade-tip point in the tool frame (numpy array)."""
    import numpy as np

    return np.array(tool_meta(tool)["suggested_tip"], dtype=float)


def build_tool_usd(tool: str, out_usd: Path) -> dict[str, Any]:
    """Author the tool USD. Requires a running SimulationApp.

    Returns a summary dict (parts, hull volume, mesh volume, prim names)
    used by the CLI and the smoke comparison.
    """
    import trimesh
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    from simretract.sim import scene

    tdir = require_tool_assets(tool)
    meta = tool_meta(tool)
    out_usd = Path(out_usd)
    out_usd.parent.mkdir(parents=True, exist_ok=True)
    if out_usd.exists():
        out_usd.unlink()

    stage = Usd.Stage.CreateNew(str(out_usd))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    root_name = tool
    root = UsdGeom.Xform.Define(stage, f"/{root_name}")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateDensityAttr(STEEL_DENSITY_KG_M3)

    pmat = UsdShade.Material.Define(stage, f"/{root_name}/physics_mat")
    mapi = UsdPhysics.MaterialAPI.Apply(pmat.GetPrim())
    mapi.CreateStaticFrictionAttr(0.8)
    mapi.CreateDynamicFrictionAttr(0.8)
    mapi.CreateRestitutionAttr(0.0)

    vis_tm = trimesh.load(str(tdir / "visual.obj"), force="mesh")
    vis = scene.author_trimesh(stage, f"/{root_name}/visual", vis_tm,
                               (0.62, 0.64, 0.67), smooth_normals=True)
    smat = UsdShade.Material.Define(stage, f"/{root_name}/steel_mat")
    sh = UsdShade.Shader.Define(stage, f"/{root_name}/steel_mat/pbr")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.62, 0.64, 0.67))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(1.0)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
    smat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(vis.GetPrim()).Bind(smat)

    UsdGeom.Scope.Define(stage, f"/{root_name}/collisions")
    coll_dir = tdir / "collision"
    coll_files = sorted(f for f in os.listdir(coll_dir) if f.endswith(".obj"))
    vol = 0.0
    for i, fn in enumerate(coll_files):
        tm = trimesh.load(str(coll_dir / fn), force="mesh")
        vol += float(tm.convex_hull.volume)
        cm = scene.author_trimesh(
            stage, f"/{root_name}/collisions/{os.path.splitext(fn)[0]}",
            tm, scene.PART_COLORS[i % len(scene.PART_COLORS)])
        UsdPhysics.CollisionAPI.Apply(cm.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(cm.GetPrim()).CreateApproximationAttr(
            UsdPhysics.Tokens.convexHull)
        UsdShade.MaterialBindingAPI.Apply(cm.GetPrim()).Bind(
            pmat, materialPurpose="physics")
        cm.CreatePurposeAttr(UsdGeom.Tokens.guide)

    for mname, key in (("tip", "suggested_tip"), ("bend", "suggested_bend")):
        mx = UsdGeom.Xform.Define(stage, f"/{root_name}/{mname}")
        mx.AddTranslateOp().Set(Gf.Vec3d(*meta[key]))

    stage.GetRootLayer().Save()
    summary = {
        "usd": str(out_usd), "n_parts": len(coll_files),
        "hull_volume_m3": vol, "meta_volume_m3": meta.get("volume_m3"),
        "mesh_volume_m3": float(vis_tm.volume) if vis_tm.is_watertight else None,
        "extents_m": [float(x) for x in vis_tm.extents],
        "prims": sorted(str(p.GetPath()) for p in stage.Traverse()),
    }
    log.info("built %s: %d parts, hull vol %.2f cm3 (~%.0f g), meta vol %s cm3",
             out_usd, len(coll_files), vol * 1e6, vol * STEEL_DENSITY_KG_M3 * 1000,
             None if meta.get("volume_m3") is None else round(meta["volume_m3"] * 1e6, 2))
    return summary


def check_parts(n_parts: int, hull_volume_m3: float | None, meta: dict[str, Any]) -> str | None:
    """The problem line when a collision set is outside the tolerance of the
    reference, else ``None``.

    ``n_collision_parts`` is compared within :data:`PARTS_TOL` parts and
    ``collision_hull_volume_m3`` (the summed convex-hull volume of the parts,
    when the reference records it) within :data:`HULL_VOLUME_REL_TOL`.
    """
    n_ref = int(meta.get("n_collision_parts", 0))
    if abs(int(n_parts) - n_ref) > PARTS_TOL:
        return (f"{n_parts} collision parts vs meta.json n_collision_parts {n_ref} (tolerance "
                f"{PARTS_TOL}; the CoACD build or threshold differs, the reference was made with "
                f"coacd==1.0.11)")
    ref_hull = meta.get("collision_hull_volume_m3")
    if ref_hull is not None and hull_volume_m3 is not None:
        if abs(float(hull_volume_m3) - float(ref_hull)) > HULL_VOLUME_REL_TOL * abs(float(ref_hull)):
            return (f"collision hull volume {float(hull_volume_m3) * 1e6:.3f} cm3 vs meta.json "
                    f"{float(ref_hull) * 1e6:.3f} cm3 (tolerance {HULL_VOLUME_REL_TOL:.0%})")
    return None


def check_against_meta(summary: dict[str, Any], meta: dict[str, Any], *,
                       tol_m: float = 1e-4, vol_rel_tol: float = 0.01) -> list[str]:
    """Differences between a built USD (``build_tool_usd`` summary) and the
    tool's reference ``meta.json``: visual extents to ``tol_m``, mesh volume
    to ``vol_rel_tol``, the collision set per :func:`check_parts`. Empty = OK."""
    problems = []
    ext_ref = [float(x) for x in meta.get("extents_m", [])]
    ext = [float(x) for x in summary["extents_m"]]
    if len(ext) != len(ext_ref) or any(abs(a - b) > tol_m for a, b in zip(ext, ext_ref)):
        problems.append(f"visual extents {ext} vs meta.json {ext_ref} (tolerance {tol_m} m)")
    v, v_ref = summary.get("mesh_volume_m3"), meta.get("volume_m3")
    if (v is None) != (v_ref is None):
        problems.append(f"mesh volume {v} vs meta.json {v_ref} (watertightness differs)")
    elif v is not None and abs(float(v) - float(v_ref)) > vol_rel_tol * abs(float(v_ref)) + 1e-12:
        problems.append(f"mesh volume {v} vs meta.json {v_ref} (tolerance {vol_rel_tol:.0%})")
    parts_problem = check_parts(int(summary["n_parts"]), summary.get("hull_volume_m3"), meta)
    if parts_problem:
        problems.append(parts_problem)
    return problems


def require_eula() -> None:
    """Fail fast with an actionable message when the Isaac Sim EULA is not
    accepted through the environment (Kit refuses to boot otherwise)."""
    from simretract import config

    try:
        config.require_kit_eula()
    except config.ConfigError as exc:
        raise SystemExit(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build assets/usd/<tool>.usd from assets/tools/<tool>/ (runs inside Isaac Sim) "
                    "and check it against the tool's reference meta.json.")
    ap.add_argument("--tool", default=DEFAULT_TOOL, help="tool name under assets/tools/")
    ap.add_argument("--out", default=None,
                    help="output USD path (default: assets/usd/<tool>.usd)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        require_tool_assets(args.tool)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    require_eula()

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        out = Path(args.out) if args.out else tool_usd_path(args.tool)
        summary = build_tool_usd(args.tool, out)
        print(f"[BUILD] {out}: {summary['n_parts']} parts, hull vol "
              f"{summary['hull_volume_m3'] * 1e6:.2f} cm3", flush=True)
        problems = check_against_meta(summary, tool_meta(args.tool))
        for p in problems:
            print(f"[BUILD] MISMATCH {p}", flush=True)
        print(f"[BUILD] check vs {tool_dir(args.tool) / 'meta.json'}: "
              f"{'OK' if not problems else 'FAIL'}", flush=True)
        rc = 1 if problems else 0
    finally:
        app.close()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
