"""Mesh / CAD -> tool asset directory (visual OBJ + CoACD collision parts).

Loads one tool from a STEP file (trimesh's OCCT backend, with gmsh and
FreeCAD fallbacks) or from a mesh/scene file (FBX / OBJ / STL / PLY / glTF);
multi-object files need ``--select <node>``. The mesh is scaled to metres
and, when the recipe says so, pre-rotated (``--pre-rotate``, applied
together with the scale before every other step), then repaired, centred on its
volumetric centroid, rotated so the blade axis is +z (``--align-z``),
exported as ``visual.obj`` and decomposed with CoACD into
``collision/coll_NNN.obj``. The decomposition reads the exported
``visual.obj`` back, so it depends only on that file and the CoACD build,
not on the trimesh version that produced it.

Which settings apply comes from three places, in this order: the command
line; the tool's recipe ``configs/assets/<tool>.yaml`` when the input file
matches its source (by sha256, or by file name with a warning: ``select``,
``scale``, ``pre_rotate`` for the source, ``align_z`` and
``coacd_threshold`` for the tool); the tracked ``meta.json``; the built-in
defaults. The input itself defaults to the recipe's drop-in file
``assets/source/<pack>/<file>`` unless ``tool_source`` overrides it
(``docs/ASSETS.md``).

``meta.json`` is the reference description of the tool: extents, longest
extent, volume, the collision set (number of convex parts and their summed
hull volume), the tip/bend points the scene uses and the settings of the
build that produced them. Converting into a tool directory that has a
``meta.json`` checks the fresh result against it: ``[MESH] check vs
meta.json: OK`` leaves the reference untouched; a mismatch exits non-zero
without touching it unless ``--accept-meta`` records the local result as
the new reference (a different source asset). With a recipe the exported
``visual.obj`` is also verified against the recipe fingerprints
(``[VERIFY]`` lines, :mod:`simretract.sim.asset_recipe`); ``--no-verify``
skips that for a different asset. No provenance path is ever written into
``meta.json``.

If a CAD import or the convex decomposition fails this reports the error
and exits non-zero; it never fabricates geometry. ``coacd`` and the FBX
loader (``pyassimp`` + the assimp shared library) are optional extras of the
simulator environment (``envs/sim.txt``).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

from simretract.sim import asset_recipe
from simretract.sim.tool_usd import check_parts, tool_dir

DEFAULT_SCALE = 0.001
DEFAULT_COACD_THRESHOLD = 0.05

#: Reference values checked after a conversion (see :func:`compare_meta`):
#: extents to 0.1 mm, volume to 1%, the collision set per ``tool_usd.check_parts``.
META_CHECK_KEYS = ("extents_m", "longest_m", "volume_m3", "n_collision_parts", "collision_hull_volume_m3")
EXTENT_TOL_M = 1e-4
VOLUME_REL_TOL = 0.01


def _ensure_assimp_libpath() -> None:
    """pyassimp needs libassimp on LD_LIBRARY_PATH; conda does not set it.
    Re-exec once with it set so the dynamic loader can find the library."""
    libdir = os.path.join(sys.prefix, "lib")
    if not os.path.isfile(os.path.join(libdir, "libassimp.so")):
        return  # not a conda assimp install; let pyassimp search normally
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if libdir not in cur.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = libdir + (os.pathsep + cur if cur else "")
        os.execv(sys.executable, [sys.executable] + sys.argv)


def _slug(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in text.strip()]
    s = "".join(keep)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def _load_fbx_or_scene(path: pathlib.Path, select: str | None):
    """Load one named object from an FBX/scene via pyassimp, baking world transforms."""
    try:
        import pyassimp
        import pyassimp.postprocess as pp
    except ImportError as exc:
        raise SystemExit("pyassimp is an optional extra of the sim environment (`pip install pyassimp` "
                         "plus the assimp shared library); it is needed to read FBX and other "
                         "multi-object scene files. See docs/ASSETS.md.") from exc
    import trimesh

    flags = pp.aiProcess_Triangulate | pp.aiProcess_JoinIdenticalVertices
    names: list[str] = []
    pick = None
    with pyassimp.load(str(path), processing=flags) as scene:

        def rec(node, parent_world):
            nonlocal pick
            world = parent_world.dot(np.array(node.transformation, dtype=float))
            if node.meshes and "$AssimpFbx$" not in node.name:
                names.append(node.name)
                if select is not None and node.name == select:
                    verts, faces, off = [], [], 0
                    for m in node.meshes:
                        v = np.asarray(m.vertices, dtype=float)
                        vw = (world.dot(np.c_[v, np.ones(len(v))].T)).T[:, :3]
                        verts.append(vw)
                        faces.append(np.asarray(m.faces, dtype=int) + off)
                        off += len(v)
                    pick = (np.vstack(verts), np.vstack(faces))
            for child in node.children:
                rec(child, world)

        rec(scene.rootnode, np.eye(4))
        if pick is None:
            avail = "\n  ".join(sorted(set(names)))
            if select is None:
                raise SystemExit(
                    f"This file has multiple objects; pass --select <name>. Available:\n  {avail}")
            raise SystemExit(f"object '{select}' not found. Available:\n  {avail}")
        verts, faces = pick
    return trimesh.Trimesh(vertices=verts, faces=faces, process=True)


def _load_step(path: pathlib.Path):
    """STEP via trimesh (cascadio/OCCT), then gmsh, then FreeCAD freecadcmd."""
    import trimesh

    try:
        mesh = trimesh.load(str(path), force="mesh")
        if mesh is not None and len(getattr(mesh, "vertices", [])) > 0:
            print("STEP loaded via trimesh (cascadio/OCCT).")
            return mesh
        print("trimesh returned empty geometry for STEP; trying fallbacks ...")
    except Exception as exc:  # noqa: BLE001
        print(f"trimesh STEP load failed: {exc!r} (install `cascadio`); trying fallbacks ...")

    try:
        import gmsh  # type: ignore

        with tempfile.TemporaryDirectory() as td:
            stl = os.path.join(td, "out.stl")
            gmsh.initialize()
            gmsh.open(str(path))
            gmsh.model.mesh.generate(2)
            gmsh.write(stl)
            gmsh.finalize()
            print("STEP loaded via gmsh.")
            return trimesh.load(stl, force="mesh")
    except Exception as exc:  # noqa: BLE001
        print(f"gmsh fallback unavailable/failed: {exc!r}")

    fc = shutil.which("freecadcmd") or shutil.which("FreeCADCmd")
    if fc:
        try:
            with tempfile.TemporaryDirectory() as td:
                stl = os.path.join(td, "out.stl")
                script = os.path.join(td, "conv.py")
                pathlib.Path(script).write_text(
                    "import Part, Mesh\n"
                    f"shape = Part.Shape(); shape.read(r'{path}')\n"
                    f"Mesh.Mesh(shape.tessellate(0.1)).write(r'{stl}')\n")
                subprocess.run([fc, script], check=True)
                print("STEP loaded via FreeCAD freecadcmd.")
                return trimesh.load(stl, force="mesh")
        except Exception as exc:  # noqa: BLE001
            print(f"FreeCAD fallback failed: {exc!r}")

    raise SystemExit(
        "No working STEP backend. Install one of: `pip install cascadio`, `gmsh`, or FreeCAD.")


def load_tool_mesh(path: pathlib.Path, select: str | None):
    """Load a tool mesh from any supported file, selecting a node if asked."""
    import trimesh

    ext = path.suffix.lower()
    if ext == ".fbx":
        _ensure_assimp_libpath()
        return _load_fbx_or_scene(path, select)
    if ext in (".step", ".stp"):
        return _load_step(path)
    obj = trimesh.load(str(path), force="scene" if select else "mesh")
    if isinstance(obj, trimesh.Scene):
        if select and select in obj.geometry:
            return obj.geometry[select]
        if select:
            raise SystemExit(
                f"'{select}' not in scene. Available:\n  " + "\n  ".join(obj.geometry))
        return obj.dump(concatenate=True)
    return obj


def process_mesh(mesh, scale: float, pre_rotate: np.ndarray | None = None):
    """Scale to metres (with the optional pre-rotation applied in the same
    step), repair, and centre on the volumetric centroid."""
    import trimesh

    m = mesh.copy()
    T = np.eye(4)
    T[:3, :3] = float(scale) * (np.eye(3) if pre_rotate is None else np.asarray(pre_rotate, dtype=float))
    m.apply_transform(T)
    m.merge_vertices()
    m.update_faces(m.unique_faces())
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(m)
    trimesh.repair.fill_holes(m)
    center = m.center_mass if m.is_watertight else m.bounding_box.centroid
    m.apply_translation(-np.asarray(center))
    return m, np.asarray(center)


def suggest_tip_bend(mesh):
    """Provisional tip (far blade end) and bend (the angle) in the tool frame."""
    V = np.asarray(mesh.vertices)
    X = V - V.mean(0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    axis = vt[0]
    proj = X.dot(axis)
    i_hi, i_lo = int(proj.argmax()), int(proj.argmin())

    def offaxis(i):
        return float(np.linalg.norm(X[i] - proj[i] * axis))

    tip_i = i_hi if offaxis(i_hi) >= offaxis(i_lo) else i_lo
    tip = V[tip_i]
    bend = V.mean(0) + 0.7 * proj[tip_i] * axis
    return np.asarray(tip), np.asarray(bend), axis


def align_blade_to_z(mesh):
    """Rotate the tool so its longest principal axis lies along +z with the
    tip end at +z (the frame convention the held scene expects)."""
    import trimesh

    m = mesh.copy()
    tip, _bend, axis = suggest_tip_bend(m)
    c = np.asarray(m.vertices).mean(0)
    if float(np.dot(np.asarray(tip) - c, axis)) < 0:   # point `axis` toward the tip end
        axis = -axis
    R = trimesh.geometry.align_vectors(axis, [0.0, 0.0, 1.0])
    m.apply_transform(R)
    center = m.center_mass if m.is_watertight else m.bounding_box.centroid
    m.apply_translation(-np.asarray(center))
    return m


def decompose(visual_obj: pathlib.Path, threshold: float, out_dir: pathlib.Path) -> tuple[list[pathlib.Path], float]:
    """CoACD convex decomposition of the exported visual mesh -> one OBJ per
    convex part, plus the summed convex-hull volume of the parts. Reading
    the OBJ back (instead of the in-memory mesh) makes the result a function
    of the file and the CoACD build only."""
    try:
        import coacd
    except ImportError as exc:
        raise SystemExit("CoACD is not installed (`pip install coacd`); pass --no-coacd "
                         "to skip the collision decomposition.") from exc
    import trimesh

    try:
        coacd.set_log_level("error")
    except Exception:  # noqa: BLE001
        pass
    mesh = trimesh.load(str(visual_obj), force="mesh", process=False)
    cmesh = coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    try:
        parts = coacd.run_coacd(cmesh, threshold=threshold)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"CoACD failed: {exc!r}")
    cdir = out_dir / "collision"
    cdir.mkdir(parents=True, exist_ok=True)
    for old in cdir.glob("coll_*.obj"):
        old.unlink()
    paths = []
    hull_volume = 0.0
    for i, (v, f) in enumerate(parts):
        part = trimesh.Trimesh(vertices=np.asarray(v), faces=np.asarray(f), process=False)
        hull_volume += float(part.convex_hull.volume)
        p = cdir / f"coll_{i:03d}.obj"
        part.export(p)
        paths.append(p)
    return paths, hull_volume


def reference_meta(out_dir: pathlib.Path) -> dict[str, Any] | None:
    """The tracked ``meta.json`` of a tool directory, or ``None``."""
    p = pathlib.Path(out_dir) / "meta.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def compare_meta(new: dict[str, Any], ref: dict[str, Any], *, check_parts_set: bool = True) -> list[str]:
    """Differences between a fresh conversion and the reference ``meta.json``.

    Extents and the longest extent must agree to 0.1 mm, the volume to 1%,
    the collision set within ``tool_usd.check_parts`` (skipped with
    ``check_parts_set=False``, i.e. after ``--no-coacd``). Returns one line
    per difference; empty means the local build reproduces the reference.
    """
    problems = []
    a, b = np.asarray(new.get("extents_m", []), float), np.asarray(ref.get("extents_m", []), float)
    if a.shape != b.shape or np.any(np.abs(a - b) > EXTENT_TOL_M):
        problems.append(f"extents_m {a.tolist()} vs reference {b.tolist()} (tolerance {EXTENT_TOL_M} m)")
    la, lb = float(new.get("longest_m") or 0.0), float(ref.get("longest_m") or 0.0)
    if abs(la - lb) > EXTENT_TOL_M:
        problems.append(f"longest_m {la} vs reference {lb} (tolerance {EXTENT_TOL_M} m)")
    va, vb = new.get("volume_m3"), ref.get("volume_m3")
    if (va is None) != (vb is None):
        problems.append(f"volume_m3 {va} vs reference {vb} (watertightness differs)")
    elif va is not None and abs(float(va) - float(vb)) > VOLUME_REL_TOL * abs(float(vb)) + 1e-12:
        problems.append(f"volume_m3 {va} vs reference {vb} (tolerance {VOLUME_REL_TOL:.0%})")
    if check_parts_set:
        parts_problem = check_parts(int(new.get("n_collision_parts", 0)),
                                    new.get("collision_hull_volume_m3"), ref)
        if parts_problem:
            problems.append(parts_problem)
    return problems


def convert(input_path: pathlib.Path, *, select: str | None, name: str,
            out_dir: pathlib.Path, scale: float, coacd_threshold: float = DEFAULT_COACD_THRESHOLD,
            no_coacd: bool = False, align_z: bool = False, accept_meta: bool = False,
            pre_rotate: np.ndarray | None = None,
            fingerprint: dict[str, Any] | None = None) -> tuple[dict, list[str]]:
    """Run the full conversion; return ``(meta, problems)``.

    ``problems`` lists the differences from the reference ``meta.json`` in
    ``out_dir`` and, when ``fingerprint`` (a recipe block) is given, from the
    recipe fingerprints of ``visual.obj`` (prefixed ``fingerprint:``); empty
    when there is no reference or the build reproduces it. The reference
    file is rewritten only when there is none or when ``accept_meta`` is
    set; the meshes are always written.
    """
    mesh = load_tool_mesh(input_path, select)
    m, center = process_mesh(mesh, scale, pre_rotate)
    if align_z:
        m = align_blade_to_z(m)
    extents = np.asarray(m.extents)
    longest = float(extents.max())

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    visual = out_dir / "visual.obj"
    m.export(visual)
    tip, bend, _ = suggest_tip_bend(m)
    parts, hull_volume = ([], None) if no_coacd else decompose(visual, coacd_threshold, out_dir)

    meta = {
        "name": name,
        "select": select,
        "scale": scale,
        "pre_rotate": None if pre_rotate is None else [[float(x) for x in row] for row in np.asarray(pre_rotate)],
        "align_z": bool(align_z),
        "coacd_threshold": None if no_coacd else coacd_threshold,
        "extents_m": [round(float(x), 5) for x in extents],
        "longest_m": round(longest, 5),
        "watertight": bool(m.is_watertight),
        "volume_m3": round(float(m.volume), 9) if m.is_watertight else None,
        "center_removed_pre_m": [round(float(x), 5) for x in center],
        "n_collision_parts": len(parts),
        "collision_hull_volume_m3": None if hull_volume is None else round(hull_volume, 9),
        "suggested_tip": [round(float(x), 5) for x in tip],
        "suggested_bend": [round(float(x), 5) for x in bend],
    }

    print("=" * 64)
    print(f"converted: {name}")
    print(f"  visual:    {visual}")
    print(f"  collision: {len(parts)} convex parts in {out_dir / 'collision'}"
          + ("" if no_coacd else f" (CoACD threshold {coacd_threshold}, hull volume "
                                 f"{hull_volume * 1e6:.3f} cm3)"))
    print(f"  extents:   {extents.round(4).tolist()} m   (longest {longest * 100:.1f} cm)"
          + ("  [blade axis aligned to +z]" if align_z else ""))
    print(f"  watertight={m.is_watertight}  volume={meta['volume_m3']} m^3")
    print(f"  suggested tip {tip.round(4).tolist()}  bend {bend.round(4).tolist()}")
    if not (0.02 <= longest <= 0.6):
        print(f"  WARNING: longest extent {longest:.3f} m is outside a typical hand-tool range "
              f"(0.02-0.6 m) - check --scale (a {0.2 / longest:.4g}x change lands it near 20 cm).")

    ref = reference_meta(out_dir)
    problems: list[str] = []
    meta_path = out_dir / "meta.json"
    if ref is None:
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"  wrote {meta_path} (no reference to check against)")
    else:
        problems = compare_meta(meta, ref, check_parts_set=not no_coacd)
        for p in problems:
            print(f"[MESH] MISMATCH {p}", flush=True)
        if not problems:
            print(f"[MESH] check vs {meta_path}: OK (reference kept)", flush=True)
        elif accept_meta:
            meta_path.write_text(json.dumps(meta, indent=2))
            print(f"[MESH] check vs {meta_path}: FAIL (--accept-meta: reference rewritten)", flush=True)
        else:
            print(f"[MESH] check vs {meta_path}: FAIL (reference kept; pass --accept-meta to record "
                  f"this build as the new reference)", flush=True)
    if fingerprint is not None:
        V, F = asset_recipe.load_obj(visual)
        fp_problems, notes = asset_recipe.compare_fingerprint(V, F, fingerprint)
        for n in notes:
            print(f"[VERIFY] {n}", flush=True)
        for p in fp_problems:
            print(f"[VERIFY] MISMATCH {p}", flush=True)
        print(f"[VERIFY] {visual} vs configs/assets/{name}.yaml: {'OK' if not fp_problems else 'FAIL'}",
              flush=True)
        problems += [f"fingerprint: {p}" for p in fp_problems]
    print("=" * 64)
    return meta, problems


def _setting(explicit, layers: list[tuple[str, dict[str, Any] | None]], key: str, default):
    """The CLI value, else the first of ``layers`` (label, mapping) that
    holds ``key``, else the default; returns ``(value, origin)``."""
    if explicit is not None:
        return explicit, "command line"
    for label, mapping in layers:
        if mapping is not None and mapping.get(key) is not None:
            return mapping[key], label
    return default, "default"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="CAD/mesh -> metres -> visual OBJ + CoACD collision, checked against the "
                    "tool's reference meta.json and recipe fingerprints.")
    ap.add_argument("--input", default=None,
                    help="STEP / FBX / OBJ / STL / glTF path (default: the tool_source override, else "
                         "the recipe's drop-in file assets/source/<pack>/<file>; docs/ASSETS.md)")
    ap.add_argument("--select", default=None,
                    help="object/node name for multi-object files (default: the recipe source when "
                         "the input matches it, else the reference meta.json)")
    ap.add_argument("--tool", default=None, help="tool name (default: slug of --select or stem)")
    ap.add_argument("--out-dir", default=None, help="output dir (default: assets/tools/<tool>)")
    ap.add_argument("--scale", type=float, default=None,
                    help="multiply coords -> metres (STEP mm=0.001; cm packs=0.01; default: the "
                         f"recipe source, else the reference meta.json, else {DEFAULT_SCALE})")
    ap.add_argument("--pre-rotate", default=None,
                    help="3x3 rotation applied with --scale before every other step, nine row-major "
                         "numbers or 'identity' (default: the recipe source, else the reference meta.json)")
    ap.add_argument("--coacd-threshold", type=float, default=None,
                    help="lower = more convex parts (default: the recipe, else the reference "
                         f"meta.json, else {DEFAULT_COACD_THRESHOLD})")
    ap.add_argument("--no-coacd", action="store_true", help="skip convex decomposition")
    ap.add_argument("--align-z", action="store_true", default=None, dest="align_z",
                    help="rotate so the blade's long axis is +z with the tip at +z "
                         "(default: the recipe, else the reference meta.json, else off)")
    ap.add_argument("--no-align-z", action="store_false", dest="align_z",
                    help="keep the source orientation even if the reference used --align-z")
    ap.add_argument("--accept-meta", action="store_true",
                    help="on a mismatch, record this build as the new reference meta.json "
                         "(implies --no-verify unless --verify is given)")
    ap.add_argument("--verify", action="store_true", default=None, dest="verify",
                    help="check the exported visual.obj against the recipe fingerprints "
                         "(default: on when the tool has a recipe)")
    ap.add_argument("--no-verify", action="store_false", dest="verify",
                    help="skip the recipe fingerprint check (a different asset)")
    args = ap.parse_args(argv)

    ref_dir = None
    if args.out_dir:
        ref_dir = pathlib.Path(args.out_dir)
    elif args.tool:
        ref_dir = tool_dir(args.tool)
    ref = reference_meta(ref_dir) if ref_dir is not None else None
    name = args.tool or (ref or {}).get("name")

    if not name and not args.input:
        raise SystemExit("pass --input FILE or --tool NAME (the tool's recipe names the drop-in "
                         "source; docs/ASSETS.md)")

    from simretract import config

    try:
        recipe = asset_recipe.load_recipe(name) if name else None
        inp, origin = asset_recipe.resolve_source(name or "", args.input)
    except config.ConfigError as exc:
        raise SystemExit(str(exc)) from exc
    if inp is None:
        raise SystemExit(f"no tool mesh: {origin}")
    if not inp.is_file():
        raise SystemExit(f"input not found: {inp}")
    if inp.suffix.lower() == ".fbx":
        _ensure_assimp_libpath()   # may re-exec: before any output
    print(f"input {inp} ({origin})")

    entry = None
    if recipe is not None:
        entry, how = asset_recipe.identify_source(inp, recipe)
        if how == "sha256":
            print(f"[SOURCE] {inp.name}: recipe source '{entry['label']}' (sha256 verified)", flush=True)
        elif how == "name":
            print(f"[SOURCE] {inp.name}: file name matches recipe source '{entry['label']}' but the "
                  f"sha256 differs (expected {entry['sha256']}, got {entry['actual_sha256']}; "
                  f"{entry['actual_size']} bytes vs {entry['size_bytes']}): another revision of the "
                  f"pack; the recipe settings are applied and the fingerprint check decides", flush=True)
        else:
            print(f"[SOURCE] {inp.name}: is not the source named in {asset_recipe.recipe_path(name).name}; "
                  f"settings come from the command line and the reference meta.json", flush=True)
    recipe_settings = asset_recipe.settings_for(entry, recipe) if recipe is not None else None
    layers = [(f"recipe source '{entry['label']}'" if entry else "recipe", recipe_settings),
              ("reference meta.json", ref)]

    select, _ = _setting(args.select, layers, "select", None)
    scale, _ = _setting(args.scale, layers, "scale", DEFAULT_SCALE)
    try:
        pre_rotate_raw, pre_origin = _setting(args.pre_rotate, layers, "pre_rotate", None)
        pre_rotate = asset_recipe.parse_pre_rotate(pre_rotate_raw)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    align_z, _ = _setting(args.align_z, layers, "align_z", False)
    threshold, _ = _setting(args.coacd_threshold, layers, "coacd_threshold", DEFAULT_COACD_THRESHOLD)
    name = name or _slug(select or inp.stem)
    out_dir = ref_dir if ref_dir is not None else tool_dir(name)
    print(f"settings: select={select!r} scale={float(scale)} "
          f"pre_rotate={'none' if pre_rotate is None else asset_recipe.format_pre_rotate(pre_rotate)} "
          f"({pre_origin}) align_z={bool(align_z)} coacd_threshold={float(threshold)}")

    verify = args.verify
    if verify is None:
        verify = recipe is not None and recipe.get("fingerprint") is not None and not args.accept_meta
    fingerprint = recipe.get("fingerprint") if (verify and recipe is not None) else None
    if verify and fingerprint is None:
        raise SystemExit(f"--verify: no recipe fingerprint for {name} (configs/assets/{name}.yaml)")

    _meta, problems = convert(inp, select=select, name=name, out_dir=out_dir, scale=float(scale),
                              coacd_threshold=float(threshold), no_coacd=args.no_coacd,
                              align_z=bool(align_z), accept_meta=args.accept_meta,
                              pre_rotate=pre_rotate, fingerprint=fingerprint)
    if args.accept_meta:
        problems = [p for p in problems if p.startswith("fingerprint:")]
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
