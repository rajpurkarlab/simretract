"""Tool asset recipes and the mesh fingerprint verifier.

A recipe, ``configs/assets/<tool>.yaml``, is the tracked description of how
the source file becomes the tool mesh under ``assets/tools/<tool>/``:

* ``source``: the pack to drop under ``assets/source/<pack>/`` as downloaded
  (file name, expected sha256 and size, listing and license), the node to
  take from it, the scale to metres and the pre-rotation applied with it
  before the converter's own steps;
* ``conversion``: the source-independent settings (``align_z``,
  ``coacd_threshold``);
* ``expected_meta``: the reference values the tracked ``meta.json`` holds;
* ``fingerprint``: what the built ``visual.obj`` must reproduce.

The fingerprint has an exact part and a tolerant part. ``face_sha256`` (the
int64 face array) is exact: a rebuild must produce the same triangles over
the same vertex order. ``vertex_sha256`` (float64 vertices rounded to
1e-6 m) is the exact vertex hash. ``vertex_grid_sha256`` is the identity
test: every coordinate is snapped to a 1 um grid whose phase (recorded per
coordinate in ``vertex_grid_phase``, one bit each) was chosen on the
released mesh so that no coordinate lies within 0.25 um of a snapping
boundary. A build reproduces that hash exactly when every coordinate is
within 0.25 um of the released mesh, and fails when any coordinate is off
by more than 0.75 um. The phase bits carry no shape information (they are
the second bit below the micrometre of each coordinate). Extents (0.1 um)
and the signed volume (1e-6 relative) are checked as readable summaries of
the same identity.

The module needs numpy and PyYAML only, so it runs in every role
environment: the converter calls it after the export (in the simulator
environment), ``simretract-build-assets --verify`` calls it on an existing
``visual.obj`` from the policy environment, and the unit tests exercise it
on synthetic meshes.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import math
import pathlib
from typing import Any

import numpy as np

from simretract import config

GRID_M = 1e-6                 # snapping grid of the tolerant vertex fingerprint
GRID_MARGIN_M = 0.25 * GRID_M  # every coordinate of the reference is at least this far from a snapping boundary
ROUND_DECIMALS = 6            # the exact vertex fingerprint: coordinates rounded to 1e-6 m
EXTENT_TOL_M = 1e-7
VOLUME_REL_TOL = 1e-6

FINGERPRINT_KEYS = ("n_vertices", "n_faces", "face_sha256", "vertex_sha256", "vertex_grid_sha256",
                    "vertex_grid_phase", "extents_m", "volume_m3", "watertight")
SOURCE_KEYS = ("file", "sha256", "size_bytes", "select", "scale")


# --------------------------------------------------------------------------- recipe


def recipe_path(tool: str) -> pathlib.Path:
    """``configs/assets/<tool>.yaml``."""
    return config.config_dir() / "assets" / f"{tool}.yaml"


def load_recipe(tool: str) -> dict[str, Any] | None:
    """The recipe of ``tool``, or ``None`` when the tool has none.

    Raises :class:`~simretract.config.ConfigError` when the file exists but
    lacks the keys the converter relies on.
    """
    p = recipe_path(tool)
    if not p.is_file():
        return None
    rec = config.load_yaml(p)
    if rec.get("tool") != tool:
        raise config.ConfigError(f"{p}: 'tool' is {rec.get('tool')!r}, expected {tool!r}")
    src = rec.get("source")
    if not isinstance(src, dict) or "pack" not in src or any(k not in src for k in SOURCE_KEYS):
        raise config.ConfigError(f"{p}: 'source' needs pack, {', '.join(SOURCE_KEYS)}")
    fp = rec.get("fingerprint")
    if fp is not None and any(k not in fp for k in FINGERPRINT_KEYS):
        missing = [k for k in FINGERPRINT_KEYS if k not in fp]
        raise config.ConfigError(f"{p}: fingerprint lacks {missing}")
    return rec


def source_entry(recipe: dict[str, Any]) -> dict[str, Any]:
    """The recipe's drop-in source with its ``label`` (the pack name)."""
    return dict(recipe["source"], label=str(recipe["source"]["pack"]))


def source_dir(recipe: dict[str, Any]) -> pathlib.Path:
    """``assets/source/<pack>/``: where the asset folder is dropped in."""
    return config.paths()["assets"] / "source" / str(recipe["source"]["pack"])


def default_source_path(recipe: dict[str, Any]) -> pathlib.Path:
    """``assets/source/<pack>/<file>``: the file the converter takes by default."""
    return source_dir(recipe) / str(recipe["source"]["file"])


def sha256_file(path: pathlib.Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def identify_source(path: pathlib.Path, recipe: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Match a file to the recipe source.

    Returns ``(entry, how)`` with ``how`` = ``"sha256"`` (size and digest
    match), ``"name"`` (file name matches but the digest does not: another
    revision of the pack) or ``"unknown"`` (``entry`` is ``None``).
    """
    path = pathlib.Path(path)
    size = path.stat().st_size
    digest = sha256_file(path)
    entry = source_entry(recipe)
    if digest == str(entry["sha256"]).lower() and int(entry["size_bytes"]) == size:
        return entry, "sha256"
    if path.name == str(entry["file"]):
        return dict(entry, actual_sha256=digest, actual_size=size), "name"
    return None, "unknown"


def resolve_source(tool: str, explicit: str | pathlib.Path | None = None) -> tuple[pathlib.Path | None, str]:
    """The file to convert for ``tool`` and where that choice came from.

    Order: an explicit path; the ``tool_source`` override
    (``SIMRETRACT_TOOL_SOURCE`` or ``configs/envs.local.yaml``); the recipe's
    drop-in file ``assets/source/<pack>/<file>`` when it exists. Returns
    ``(None, message)`` when nothing applies, the message naming the drop-in
    path to fill.
    """
    if explicit:
        return pathlib.Path(explicit).expanduser(), "command line"
    override = config.tool_source()
    if override is not None:
        return override, "tool_source override (SIMRETRACT_TOOL_SOURCE or configs/envs.local.yaml)"
    recipe = load_recipe(tool)
    if recipe is None:
        return None, (f"no recipe configs/assets/{tool}.yaml and no tool_source override; pass the "
                      f"file explicitly (docs/ASSETS.md)")
    p = default_source_path(recipe)
    if p.is_file():
        return p, f"recipe default assets/source/{recipe['source']['pack']}/{recipe['source']['file']}"
    return None, (f"no tool source: place the asset pack as downloaded under "
                  f"assets/source/{recipe['source']['pack']}/ (expected {p.name}), or set tool_source in "
                  f"configs/envs.local.yaml / SIMRETRACT_TOOL_SOURCE (docs/ASSETS.md)")


def parse_pre_rotate(value: Any) -> np.ndarray | None:
    """A 3x3 rotation from a recipe entry (nested list), a CLI string (nine
    numbers, comma or space separated) or ``None``/``"identity"``.

    The matrix must be orthonormal with determinant +1 (1e-6); it is applied
    to the source coordinates together with the scale, before every other
    conversion step.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "identity"}:
            return None
        nums = [float(x) for x in value.replace(",", " ").replace("[", " ").replace("]", " ").split()]
        if len(nums) != 9:
            raise ValueError(f"--pre-rotate needs 9 numbers (row-major 3x3), got {len(nums)}")
        R = np.asarray(nums, dtype=float).reshape(3, 3)
    else:
        R = np.asarray(value, dtype=float)
        if R.shape != (3, 3):
            raise ValueError(f"pre_rotate must be 3x3, got shape {R.shape}")
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-6) or abs(np.linalg.det(R) - 1.0) > 1e-6:
        raise ValueError(f"pre_rotate is not a proper rotation (orthonormal, det +1): {R.tolist()}")
    return R


def format_pre_rotate(R: np.ndarray | None) -> str:
    """The CLI form of a pre-rotation (inverse of :func:`parse_pre_rotate`)."""
    if R is None:
        return "identity"
    return ",".join(repr(float(x)) for x in np.asarray(R, dtype=float).reshape(-1))


def settings_for(entry: dict[str, Any] | None, recipe: dict[str, Any] | None) -> dict[str, Any]:
    """Conversion settings from the matched source entry and the recipe's
    source-independent block; keys absent from both are omitted."""
    out: dict[str, Any] = {}
    if entry is not None:
        out["select"] = entry.get("select")
        out["scale"] = float(entry["scale"])
        out["pre_rotate"] = parse_pre_rotate(entry.get("pre_rotate"))
    conv = (recipe or {}).get("conversion") or {}
    for key in ("align_z", "coacd_threshold"):
        if conv.get(key) is not None:
            out[key] = conv[key]
    return out


# --------------------------------------------------------------------------- fingerprints


def load_obj(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (float64, n x 3) and triangle faces (int64, m x 3) of a
    Wavefront OBJ in file order (what ``trimesh.load(..., process=False)``
    returns for the converter's export)."""
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    i = int(tok.split("/")[0])
                    idx.append(len(verts) + i if i < 0 else i - 1)
                if len(idx) != 3:
                    raise ValueError(f"{path}: face with {len(idx)} vertices; the tool mesh is triangulated")
                faces.append(idx)
    if not verts or not faces:
        raise ValueError(f"{path}: no vertices/faces found")
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def vertex_sha256(V: np.ndarray) -> str:
    """SHA-256 of the float64 vertices rounded to 1e-6 m (C-order bytes)."""
    v = np.round(np.ascontiguousarray(np.asarray(V, dtype=np.float64)), ROUND_DECIMALS)
    return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()


def face_sha256(F: np.ndarray) -> str:
    """SHA-256 of the int64 face array (C-order bytes)."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(F, dtype=np.int64)).tobytes()).hexdigest()


def grid_phase(V: np.ndarray) -> np.ndarray:
    """Per-coordinate phase bits of the reference mesh: 1 where the
    coordinate sits in the middle half of its 1 um cell (snapped with
    ``floor``), 0 where it sits near a cell boundary (snapped with
    ``round``). Either way the coordinate is at least 0.25 um from the
    boundary of the snapping it gets."""
    q = np.asarray(V, dtype=np.float64).reshape(-1) / GRID_M
    frac = q - np.floor(q)
    return ((frac >= 0.25) & (frac < 0.75)).astype(np.uint8)


def encode_phase(bits: np.ndarray) -> str:
    return base64.b64encode(np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()).decode("ascii")


def decode_phase(text: str, n_coords: int) -> np.ndarray:
    raw = base64.b64decode("".join(str(text).split()))
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    if len(bits) < n_coords or len(bits) - n_coords >= 8:
        raise ValueError(f"vertex_grid_phase holds {len(bits)} bits for {n_coords} coordinates")
    return bits[:n_coords]


def grid_codes(V: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Integer 1 um cell of every coordinate under its phase."""
    q = np.asarray(V, dtype=np.float64).reshape(-1) / GRID_M
    phase = np.asarray(phase, dtype=np.uint8).reshape(-1)
    if phase.shape != q.shape:
        raise ValueError(f"phase has {phase.size} bits for {q.size} coordinates")
    codes = np.where(phase == 1, np.floor(q), np.round(q))
    return codes.astype(np.int64)


def vertex_grid_sha256(V: np.ndarray, phase: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(grid_codes(V, phase)).tobytes()).hexdigest()


def signed_volume(V: np.ndarray, F: np.ndarray) -> float:
    """Volume enclosed by a closed, consistently wound triangle mesh (the
    sum of signed tetrahedra with the origin; what ``trimesh`` reports)."""
    a, b, c = (np.asarray(V, dtype=np.float64)[np.asarray(F)[:, i]] for i in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def is_closed(F: np.ndarray) -> bool:
    """Every undirected edge is shared by exactly two faces (``trimesh``'s
    ``is_watertight``)."""
    F = np.asarray(F, dtype=np.int64)
    edges = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def mesh_fingerprint(V: np.ndarray, F: np.ndarray, phase: np.ndarray | None = None) -> dict[str, Any]:
    """The fingerprint block of a mesh. With ``phase`` (the reference's
    bits) the grid hash is computed under it; without, the mesh is treated
    as the reference and its own phase is recorded."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    bits = grid_phase(V) if phase is None else np.asarray(phase, dtype=np.uint8)
    ext = V.max(axis=0) - V.min(axis=0)
    closed = is_closed(F)
    return {
        "n_vertices": int(len(V)),
        "n_faces": int(len(F)),
        "face_sha256": face_sha256(F),
        "vertex_sha256": vertex_sha256(V),
        "vertex_grid_sha256": vertex_grid_sha256(V, bits),
        "vertex_grid_phase": encode_phase(bits),
        "extents_m": [round(float(x), 9) for x in ext],
        "volume_m3": round(abs(signed_volume(V, F)), 15) if closed else None,
        "watertight": closed,
    }


def compare_fingerprint(V: np.ndarray, F: np.ndarray, expected: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Check a mesh against a recipe ``fingerprint`` block.

    Returns ``(problems, notes)``: an empty ``problems`` means the mesh is
    the released tool; ``notes`` say whether the vertices matched exactly or
    within the 0.25 um grid margin.
    """
    problems: list[str] = []
    notes: list[str] = []
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    n_v, n_f = int(expected["n_vertices"]), int(expected["n_faces"])
    if len(V) != n_v or len(F) != n_f:
        problems.append(f"{len(V)} vertices / {len(F)} faces vs recipe {n_v} / {n_f}")
        return problems, notes
    if face_sha256(F) != str(expected["face_sha256"]):
        problems.append("face_sha256 differs (different triangles or vertex order)")
    if vertex_sha256(V) == str(expected["vertex_sha256"]):
        notes.append("vertex_sha256: exact (coordinates identical to 1e-6 m)")
    else:
        phase = decode_phase(expected["vertex_grid_phase"], V.size)
        if vertex_grid_sha256(V, phase) == str(expected["vertex_grid_sha256"]):
            notes.append(f"vertex_sha256 differs, vertex_grid_sha256 OK: every coordinate within "
                         f"{GRID_MARGIN_M * 1e6:.2f} um of the released mesh")
        else:
            problems.append(f"vertex_grid_sha256 differs: some coordinate is more than "
                            f"{GRID_MARGIN_M * 1e6:.2f} um from the released mesh")
    ext = V.max(axis=0) - V.min(axis=0)
    ext_ref = np.asarray(expected["extents_m"], dtype=float)
    if np.any(np.abs(ext - ext_ref) > EXTENT_TOL_M):
        problems.append(f"extents_m {ext.tolist()} vs recipe {ext_ref.tolist()} (tolerance {EXTENT_TOL_M} m)")
    closed = is_closed(F)
    if closed != bool(expected["watertight"]):
        problems.append(f"watertight {closed} vs recipe {expected['watertight']}")
    vol_ref = expected.get("volume_m3")
    if closed and vol_ref is not None:
        vol = abs(signed_volume(V, F))
        if not math.isclose(vol, float(vol_ref), rel_tol=VOLUME_REL_TOL, abs_tol=0.0):
            problems.append(f"volume_m3 {vol} vs recipe {vol_ref} (tolerance {VOLUME_REL_TOL:.0e} relative)")
    return problems, notes


def verify_obj(path: pathlib.Path, recipe: dict[str, Any]) -> tuple[list[str], list[str]]:
    """:func:`compare_fingerprint` on an OBJ file against the recipe."""
    fp = recipe.get("fingerprint")
    if fp is None:
        raise config.ConfigError(f"configs/assets/{recipe['tool']}.yaml has no fingerprint block")
    V, F = load_obj(pathlib.Path(path))
    return compare_fingerprint(V, F, fp)


def report(tag: str, path: pathlib.Path, recipe: dict[str, Any]) -> int:
    """Print the ``[VERIFY]`` lines for ``path`` and return 0 (OK) or 1."""
    problems, notes = verify_obj(path, recipe)
    for n in notes:
        print(f"{tag} {n}", flush=True)
    for p in problems:
        print(f"{tag} MISMATCH {p}", flush=True)
    print(f"{tag} {path} vs {recipe_path(recipe['tool']).relative_to(config.repo_root())}: "
          f"{'OK' if not problems else 'FAIL'}", flush=True)
    return 1 if problems else 0


def fingerprint_yaml(fp: dict[str, Any]) -> str:
    """The ``fingerprint:`` block of a recipe for a mesh (phase folded)."""
    lines = ["fingerprint:"]
    for key in ("n_vertices", "n_faces", "face_sha256", "vertex_sha256", "vertex_grid_sha256"):
        lines.append(f"  {key}: {fp[key]}")
    lines.append("  vertex_grid_phase: >-")
    phase = str(fp["vertex_grid_phase"])
    lines += [f"    {phase[i:i + 76]}" for i in range(0, len(phase), 76)]
    lines.append(f"  extents_m: [{', '.join(repr(float(x)) for x in fp['extents_m'])}]")
    lines.append(f"  volume_m3: {fp['volume_m3']!r}")
    lines.append(f"  watertight: {str(bool(fp['watertight'])).lower()}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    from simretract.sim.tool_usd import DEFAULT_TOOL, tool_dir

    ap = argparse.ArgumentParser(
        description="Verify a built visual.obj against the tool's recipe fingerprints "
                    "(configs/assets/<tool>.yaml), or print the fingerprint block of a mesh.")
    ap.add_argument("--tool", default=DEFAULT_TOOL)
    ap.add_argument("--obj", default=None, help="mesh to check (default: assets/tools/<tool>/visual.obj)")
    ap.add_argument("--print-fingerprint", action="store_true",
                    help="print the recipe fingerprint block of --obj instead of verifying it")
    args = ap.parse_args(argv)
    obj = pathlib.Path(args.obj) if args.obj else tool_dir(args.tool) / "visual.obj"
    if not obj.is_file():
        raise SystemExit(f"{obj} is not built (docs/ASSETS.md)")
    if args.print_fingerprint:
        V, F = load_obj(obj)
        print(fingerprint_yaml(mesh_fingerprint(V, F)), end="")
        return 0
    recipe = load_recipe(args.tool)
    if recipe is None:
        raise SystemExit(f"no recipe {recipe_path(args.tool)}")
    return report("[VERIFY]", obj, recipe)


if __name__ == "__main__":
    raise SystemExit(main())
