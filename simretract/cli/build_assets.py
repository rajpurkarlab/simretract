"""``simretract-build-assets``: build the simulation assets locally.

The retractor mesh is a third-party asset that is never part of the
repository: the user obtains it under its publisher's license, drops the
purchased folder under ``assets/source/<pack>/`` as downloaded, and this
command converts it into everything the simulation needs (``docs/ASSETS.md``).
Every step runs in the simulator environment as a subprocess (the tool USD
needs Isaac Sim; the converter and the samplers need trimesh/scipy which
that environment pins), so this entry point works from any environment::

    # the default: the whole build from the recipe (configs/assets/<tool>.yaml)
    simretract-build-assets --tool retractor_rake --accept-eula
    #   = 1. source file -> assets/tools/<tool>/{visual.obj, collision/}, checked against
    #        meta.json and the recipe fingerprints ([MESH], [VERIFY])
    #     2. tool USD, checked against meta.json                     -> assets/usd/<tool>.usd ([BUILD])
    #     3. the tracked grasp candidates re-sampled and compared     ([GRASPS])

    # the steps one by one
    simretract-build-assets --from-mesh                       # 1, source from the recipe or tool_source
    simretract-build-assets --from-mesh pack.fbx --select node --scale 0.01 --pre-rotate identity
    simretract-build-assets --verify                          # fingerprint check of the built visual.obj only
    simretract-build-assets --usd                             # 2
    simretract-build-assets --grasps --compare                # 3
    simretract-build-assets --grasps                          # re-sample and overwrite (a different asset)
    simretract-build-assets --lut                             # gripper silhouette LUT for the real-robot client

``--compare`` verifies without overwriting: the USD is built into a temporary
directory and the re-sampled grasp candidates are compared with the tracked
JSON. ``--select``, ``--scale``, ``--pre-rotate``, ``--align-z`` and
``--coacd-threshold`` default to the recipe source the input file matches
(by sha256), then to the tracked ``meta.json``; the conversion checks its
result against both and fails on a mismatch unless ``--accept-meta``
records the local build as the new reference (``--no-verify`` skips the
fingerprint check for a different asset).
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile

from simretract.sim.tool_usd import DEFAULT_TOOL

log = logging.getLogger(__name__)

#: ``--from-mesh`` without a value: take the path from the recipe / configuration.
FROM_CONFIG = "<recipe or tool_source>"


def _sim_python(explicit: str | None) -> str:
    if explicit:
        return explicit
    from simretract import config

    return str(config.interpreter("sim"))


def _run(python: str, module: str, args: list[str], accept_eula: bool) -> int:
    env = dict(os.environ)
    if accept_eula:
        env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    cmd = [python, "-m", module] + args
    log.info("running: %s", " ".join(cmd))
    return subprocess.run(cmd, env=env).returncode


def resolve_source(value: str | None, tool: str) -> str | None:
    """The mesh to convert: an explicit path, or (``--from-mesh`` without a
    value) the ``tool_source`` override else the recipe's drop-in file
    (``None`` when a conversion was not requested)."""
    if value is None:
        return None
    if value != FROM_CONFIG:
        return value
    from simretract import config
    from simretract.sim import asset_recipe

    try:
        path, origin = asset_recipe.resolve_source(tool)
    except config.ConfigError as exc:
        raise SystemExit(str(exc)) from exc
    if path is None:
        raise SystemExit(f"no tool mesh to convert: {origin}")
    log.info("source: %s (%s)", path, origin)
    return str(path)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Build the simulation assets locally from a licensed tool mesh: tool meshes, "
                    "tool USD, grasp candidates, gripper LUT (docs/ASSETS.md). With no action flag "
                    "the whole build runs: --from-mesh (recipe source), --usd, --grasps --compare.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--tool", default=DEFAULT_TOOL, help="tool name under assets/tools")
    ap.add_argument("--from-mesh", nargs="?", const=FROM_CONFIG, default=None, metavar="FILE",
                    help="convert a mesh/CAD file into assets/tools/<tool>/ (runs first); without "
                         "a value the file is the tool_source override (configs/envs.local.yaml or "
                         "SIMRETRACT_TOOL_SOURCE), else the recipe's assets/source/<pack>/<file>")
    ap.add_argument("--verify", action="store_true", default=None, dest="verify",
                    help="check visual.obj against the recipe fingerprints (part of --from-mesh by "
                         "default; alone, checks the mesh already built)")
    ap.add_argument("--no-verify", action="store_false", dest="verify",
                    help="--from-mesh: skip the fingerprint check (a different asset)")
    ap.add_argument("--usd", action="store_true",
                    help="build the tool USD and check it against meta.json")
    ap.add_argument("--grasps", action="store_true", help="sample the grasp-candidate database")
    ap.add_argument("--lut", action="store_true",
                    help="render the gripper silhouette LUT used by the real-robot client")
    ap.add_argument("--out", default=None,
                    help="output path of the single requested USD or grasp JSON")
    ap.add_argument("--compare", action="store_true",
                    help="verify without overwriting: build the USD in a temporary directory, "
                         "compare re-sampled grasp candidates with the tracked JSON")
    ap.add_argument("--plot", default=None, help="--grasps: write a scatter figure (.png)")
    ap.add_argument("--select", default=None,
                    help="--from-mesh: node name in a multi-object file (default: recipe, meta.json)")
    ap.add_argument("--scale", type=float, default=None,
                    help="--from-mesh: units -> metres (default: recipe, meta.json, else 0.001)")
    ap.add_argument("--pre-rotate", default=None, metavar="R",
                    help="--from-mesh: nine row-major numbers or 'identity', applied with --scale "
                         "before every other step (default: recipe, meta.json)")
    ap.add_argument("--coacd-threshold", type=float, default=None,
                    help="--from-mesh: CoACD concavity threshold (default: recipe, meta.json, else 0.05)")
    ap.add_argument("--no-coacd", action="store_true", help="--from-mesh: skip the decomposition")
    ap.add_argument("--align-z", action="store_true", default=None, dest="align_z",
                    help="--from-mesh: put the blade axis on +z (default: recipe, meta.json)")
    ap.add_argument("--no-align-z", action="store_false", dest="align_z",
                    help="--from-mesh: keep the source orientation")
    ap.add_argument("--accept-meta", action="store_true",
                    help="--from-mesh: on a mismatch, record the local build as the new meta.json")
    ap.add_argument("--accept-eula", action="store_true",
                    help="set OMNI_KIT_ACCEPT_EULA=YES for the simulator subprocess")
    ap.add_argument("--sim-python", default=None,
                    help="simulator interpreter (default: the sim_python recorded in "
                         "configs/envs.local.yaml by envs/setup.sh)")
    ap.add_argument("extra", nargs="*", help="extra arguments passed to the delegated module")
    return ap


def mesh_args(args: argparse.Namespace, source: str) -> list[str]:
    """Arguments of ``simretract.sim.mesh_convert`` for ``--from-mesh``."""
    a = ["--input", source, "--tool", args.tool]
    if args.select:
        a += ["--select", args.select]
    if args.scale is not None:
        a += ["--scale", str(args.scale)]
    if args.pre_rotate is not None:
        a += ["--pre-rotate", args.pre_rotate]
    if args.coacd_threshold is not None:
        a += ["--coacd-threshold", str(args.coacd_threshold)]
    if args.no_coacd:
        a.append("--no-coacd")
    if args.align_z is True:
        a.append("--align-z")
    elif args.align_z is False:
        a.append("--no-align-z")
    if args.accept_meta:
        a.append("--accept-meta")
    if args.verify is True:
        a.append("--verify")
    elif args.verify is False:
        a.append("--no-verify")
    return a


def main(argv: list[str] | None = None) -> int:
    """Run the requested actions in order: --from-mesh, --verify, --usd, --grasps, --lut.

    The actions are independent subprocesses; the first failure stops the
    sequence and its return code is returned. With no action flag the whole
    build runs from the recipe: conversion (with the fingerprint check), the
    tool USD in place, and the grasp-candidate comparison.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    if not (args.usd or args.grasps or args.lut or args.from_mesh or args.verify):
        args.from_mesh = FROM_CONFIG
        args.usd = True
        args.grasps = True
        grasps_compare = True
    else:
        grasps_compare = args.compare
    source = resolve_source(args.from_mesh, args.tool)
    py = _sim_python(args.sim_python)

    if source:
        rc = _run(py, "simretract.sim.mesh_convert", mesh_args(args, source) + args.extra,
                  args.accept_eula)
        if rc:
            return rc
    elif args.verify:
        from simretract.sim import asset_recipe

        rc = asset_recipe.main(["--tool", args.tool])
        if rc:
            return rc

    if args.usd:
        a = ["--tool", args.tool]
        if args.compare:
            tmp = tempfile.mkdtemp(prefix="simretract_usd_")
            a += ["--out", os.path.join(tmp, f"{args.tool}.usd")]
        elif args.out:
            a += ["--out", args.out]
        rc = _run(py, "simretract.sim.tool_usd", a + args.extra, args.accept_eula)
        if rc:
            return rc

    if args.grasps:
        a = ["--tool", args.tool]
        if grasps_compare:
            a.append("--compare")
        elif args.out and not args.usd:
            a += ["--out", args.out]
        if args.plot:
            a += ["--plot", args.plot]
        rc = _run(py, "simretract.sim.grasp_candidates", a + args.extra, args.accept_eula)
        if rc:
            return rc

    if args.lut:
        rc = _run(py, "simretract.real.gmask_lut", args.extra, args.accept_eula)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
