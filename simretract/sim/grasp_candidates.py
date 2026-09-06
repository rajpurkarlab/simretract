"""Geometric grasp-candidate sampling and the cached candidate database.

The recorders never search the tool surface online: they draw from
``assets/grasp_candidates/<tool>.json``, a fixed list of grasp poses in the
tool frame produced once by the samplers below (flat-clamp, antipodal and
washer families). The recorders permute this list per seed, so the shipped
JSON is part of the data contract: regenerating it can change which grasp a
seed ends up with. ``--compare`` re-samples and checks the shipped file
instead of overwriting it. The candidates are grasp poses in the tool
frame (centre, rotation, width, standoff): released annotations that carry
no mesh geometry; the mesh they were sampled from is built locally
(``docs/ASSETS.md``).

Grasp frame axes (columns of ``R``) match the gripper/TCP frame:
  x = approach (the gripper points this way; the pre-grasp backs off here)
  y = close    (the antipodal line; finger motion axis)
  z = x cross y
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib
from typing import Any

import numpy as np

from simretract.sim.tool_usd import DEFAULT_TOOL, assets_dir, tool_dir

log = logging.getLogger(__name__)

N_GRASPS = 240
SAMPLER_SEED = 0
CANDIDATE_KINDS = ("flat", "antipodal", "washer")


def candidates_path(tool: str) -> pathlib.Path:
    """``assets/grasp_candidates/<tool>.json``."""
    return assets_dir() / "grasp_candidates" / f"{tool}.json"


def load_candidate_db(tool: str) -> dict[str, Any]:
    with open(candidates_path(tool)) as f:
        return json.load(f)


def load_cands(db: dict[str, Any], kind: str, standoff_clip=(0.05, 0.09)) -> list[dict]:
    """Candidate dicts as the recorders consume them: numpy center / R,
    width, and the candidate standoff clipped to the reach range."""
    return [{"kind": kind, "center": np.array(d["center"]),
             "R": np.array(d["R"]), "width": float(d["width"]),
             "standoff": float(np.clip(d["standoff"], *standoff_clip))}
            for d in db[kind]]


@dataclasses.dataclass
class GraspCandidate:
    center: np.ndarray    # grasp center, tool frame (m)
    axis: np.ndarray      # closing/antipodal axis (unit), tool frame
    approach: np.ndarray  # approach axis (unit), tool frame
    R: np.ndarray         # 3x3 grasp frame in tool coords [approach, axis, third]
    width: float          # contact separation (m)
    standoff: float       # pre-grasp backoff along -approach (m)
    quality: float        # antipodal alignment in [0, 1]
    p1: np.ndarray
    p2: np.ndarray

    def to_dict(self) -> dict:
        out = {}
        for k, v in dataclasses.asdict(self).items():
            out[k] = v.tolist() if isinstance(v, np.ndarray) else float(v)
        return out

    @staticmethod
    def from_dict(d: dict) -> "GraspCandidate":
        return GraspCandidate(
            center=np.array(d["center"]), axis=np.array(d["axis"]),
            approach=np.array(d["approach"]), R=np.array(d["R"]),
            width=float(d["width"]), standoff=float(d["standoff"]),
            quality=float(d["quality"]), p1=np.array(d["p1"]), p2=np.array(d["p2"]),
        )


def _perp_basis(a: np.ndarray):
    """Two orthonormal vectors spanning the plane perpendicular to unit vector a."""
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(a, ref)
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    return u, v


def _pose_from_roll(center, a, roll, standoff):
    u, v = _perp_basis(a)
    approach = np.cos(roll) * u + np.sin(roll) * v
    approach /= np.linalg.norm(approach)
    third = np.cross(approach, a)
    R = np.column_stack([approach, a, third])
    return approach, R


# --------------------------------------------------------------------------- #
# Washer / face-clamp grasps: for fenestrated tools (handles with through-holes)
# clamp the two broad faces of the flat plate with the pads spanning a hole and
# bearing on the surrounding annulus. The plate normal and the holes are
# detected from the mesh, so any looped/holed tool works without tuning.
# --------------------------------------------------------------------------- #
def _plate_axis_candidates(mesh) -> np.ndarray:
    """Candidate plate normals: eigenvectors of the area-weighted normal
    matrix, strongest first (a thin plate's broad faces dominate)."""
    n, a = mesh.face_normals, mesh.area_faces
    M = (n * a[:, None]).T @ n
    w, V = np.linalg.eigh(M)
    return V[:, np.argsort(w)[::-1]].T


def detect_plate_holes(mesh, plate_normal=None):
    """Return ``(plate_normal, holes)`` for a fenestrated tool; the axis with
    the most enclosed-hole area wins when none is given. ``(None, [])`` when
    there are no through-holes."""
    if plate_normal is not None:
        nrm = np.asarray(plate_normal, float)
        return nrm / np.linalg.norm(nrm), detect_through_holes(mesh, nrm)
    best = (None, [])
    best_area = 0.0
    for cand in _plate_axis_candidates(mesh):
        holes = detect_through_holes(mesh, cand)
        area = sum(np.pi * r * r for _, r in holes)
        if area > best_area:
            best_area, best = area, (cand, holes)
    return best


def detect_through_holes(mesh, normal: np.ndarray, *, cell: float = 0.0015,
                         min_area: float = 1.5e-5):
    """Interior through-holes of a plate by ray-casting a grid along its
    normal: open cells connected to the border are background, enclosed open
    cells are holes. Returns ``[(center_xyz, radius), ...]`` at mid-plane."""
    from scipy.ndimage import label

    u, v = _perp_basis(normal)
    V = mesh.vertices
    pu, pv, pn = V @ u, V @ v, V @ normal
    lo_u, hi_u, lo_v, hi_v = pu.min(), pu.max(), pv.min(), pv.max()
    nu = max(1, int(np.ceil((hi_u - lo_u) / cell)))
    nv = max(1, int(np.ceil((hi_v - lo_v) / cell)))
    gu = lo_u + (np.arange(nu) + 0.5) * cell
    gv = lo_v + (np.arange(nv) + 0.5) * cell
    UU, VV = np.meshgrid(gu, gv, indexing="ij")
    n_mid = float((pn.min() + pn.max()) / 2.0)
    base = (UU[..., None] * u + VV[..., None] * v
            + (pn.min() - cell) * normal).reshape(-1, 3)             # start below plate
    dirs = np.tile(normal, (base.shape[0], 1))
    solid = mesh.ray.intersects_any(base, dirs).reshape(nu, nv)

    lab, k = label(~solid)
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    holes = []
    for li in range(1, k + 1):
        if li in border:
            continue
        mask = lab == li
        area = int(mask.sum()) * cell * cell
        if area < min_area:
            continue
        ui, vi = np.where(mask)
        cu = lo_u + (ui.mean() + 0.5) * cell
        cv = lo_v + (vi.mean() + 0.5) * cell
        center = cu * u + cv * v + n_mid * normal
        holes.append((center, float(np.sqrt(area / np.pi))))
    return holes


def sample_flat_clamp_grasps(mesh, tip, *, n_grasps: int = 240,
                             n_poses_per_site: int = 4, min_tip_clearance: float = 0.04,
                             min_width: float = 0.0015, max_width: float = 0.044,
                             flat_align_deg: float = 15.0, standoff_range: tuple = (0.05, 0.09),
                             n_surface: int = 20000, dedup_grid: float = 0.008,
                             seed: int = 0) -> list[GraspCandidate]:
    """Flat-face clamps for a wide flat-pad gripper: both contact faces are
    locally flat and parallel (a plate-like region), so the pad seats fully
    and the jaw opening matches the local thickness. The closing axis is the
    local surface normal; the pad approaches edge-on (roll about it)."""
    import trimesh

    rng = np.random.default_rng(seed)
    tip = np.asarray(tip, float)
    pts, fidx = trimesh.sample.sample_surface(mesh, n_surface, seed=seed)
    N = mesh.face_normals[fidx]
    cos_t = np.cos(np.radians(flat_align_deg))
    seen, sites = set(), []
    for i in rng.permutation(len(pts)):
        p, n = pts[i], N[i]
        # cast across the plate (into the material, opposite the outward normal)
        org = (p - n * 1e-4)[None]
        locs, _ridx, tidx = mesh.ray.intersects_location(org, (-n)[None])
        if len(locs) == 0:
            continue
        d = (locs - p) @ (-n)                                  # distance along -n
        m = d > min_width
        if not m.any():
            continue
        j = np.where(m)[0][np.argmin(d[m])]                   # nearest valid opposing face
        p2, w = locs[j], float(d[j])
        if w > max_width:
            continue
        if float(mesh.face_normals[tidx[j]] @ (-n)) < cos_t:  # opposing face must be parallel & flat
            continue
        center = (p + p2) / 2.0
        if np.linalg.norm(center - tip) < min_tip_clearance:
            continue
        key = tuple(np.round(center / dedup_grid).astype(int))
        if key in seen:
            continue
        seen.add(key)
        sites.append((center, n.copy(), w, p.copy(), p2.copy()))
    rng.shuffle(sites)
    sites = sites[:max(1, int(np.ceil(n_grasps / max(1, n_poses_per_site))))]
    grasps: list[GraspCandidate] = []
    for center, axis, w, p1, p2 in sites:
        offset = float(rng.uniform(0, 2 * np.pi))
        for k in range(n_poses_per_site):
            roll = offset + 2 * np.pi * k / n_poses_per_site
            approach, R = _pose_from_roll(center, axis, roll, 0.0)
            grasps.append(GraspCandidate(
                center=center.copy(), axis=axis.copy(), approach=approach, R=R,
                width=float(w), standoff=float(rng.uniform(*standoff_range)),
                quality=1.0, p1=p1, p2=p2))
    rng.shuffle(grasps)
    return grasps[:n_grasps]


def _plate_thickness(mesh, normal, center, radius):
    """Median plate thickness on the annulus around a hole (ray entry->exit
    along the normal), probed in many directions/offsets; misses are skipped."""
    u, v = _perp_basis(normal)
    th = []
    for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        d = np.cos(ang) * u + np.sin(ang) * v
        for f in (1.3, 1.7, 2.2):
            org = (center + d * (radius * f) - normal * 0.05)[None]
            locs = mesh.ray.intersects_location(org, normal[None])[0]
            if len(locs) >= 2:
                p = locs @ normal
                th.append(p.max() - p.min())
    return float(np.median(th)) if th else float("nan")


def sample_washer_grasps(mesh, tip, *, n_poses_per_hole: int = 6,
                         standoff_range: tuple = (0.05, 0.09), min_tip_clearance: float = 0.04,
                         min_width: float = 0.0015, max_width: float = 0.044,
                         plate_normal=None, seed: int = 0) -> list[GraspCandidate]:
    """Washer-pinch candidates centred on each detected through-hole:
    closing axis = plate normal, the pad spans the hole and bears on the
    annulus."""
    rng = np.random.default_rng(seed)
    tip = np.asarray(tip, float)
    normal, holes = detect_plate_holes(mesh, plate_normal)
    if normal is None:
        return []                                # no fenestration -> no washer grasps
    grasps: list[GraspCandidate] = []
    for center, radius in holes:
        if np.linalg.norm(center - tip) < min_tip_clearance:
            continue
        w = _plate_thickness(mesh, normal, center, radius)
        if not (min_width <= w <= max_width):                        # must fit the jaws
            continue
        offset = float(rng.uniform(0, 2 * np.pi))
        for k in range(n_poses_per_hole):
            roll = offset + 2 * np.pi * k / n_poses_per_hole
            approach, R = _pose_from_roll(center, normal, roll, 0.0)
            grasps.append(GraspCandidate(
                center=center.copy(), axis=normal.copy(), approach=approach, R=R,
                width=float(w), standoff=float(rng.uniform(*standoff_range)),
                quality=1.0, p1=center + normal * w / 2, p2=center - normal * w / 2))
    return grasps


def sample_antipodal_grasps(
    mesh,
    tip,
    *,
    max_width: float = 0.044,
    min_width: float = 0.0015,
    min_tip_clearance: float = 0.04,
    n_surface: int = 10000,
    normal_thresh_deg: float = 22.0,
    n_poses_per_pair: int = 4,
    standoff_range: tuple = (0.05, 0.09),
    n_grasps: int = 240,
    dedup_grid: float = 0.006,
    seed: int = 0,
) -> list[GraspCandidate]:
    """Up to ``n_grasps`` 6-DoF antipodal candidates over the whole surface:
    surface point pairs whose outward normals nearly oppose and whose
    separation fits the jaws, each yielding several rolls about the closing
    axis. The only positional constraint is the tip keep-out."""
    import trimesh
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    tip = np.asarray(tip, float)
    pts, fidx = trimesh.sample.sample_surface(mesh, n_surface, seed=seed)
    N = mesh.face_normals[fidx]

    tree = cKDTree(pts)
    cos_t = np.cos(np.radians(normal_thresh_deg))

    # one best antipodal pair per query point, spatially deduped
    pairs = []
    seen = set()
    for i in rng.permutation(len(pts)):
        pi, ni = pts[i], N[i]
        best = None
        for j in tree.query_ball_point(pi, max_width):
            if j == i:
                continue
            pj, nj = pts[j], N[j]
            d = pj - pi
            w = float(np.linalg.norm(d))
            if w < min_width or w > max_width:
                continue
            d = d / w
            a1 = float(-ni @ d)
            a2 = float(nj @ d)
            if a1 < cos_t or a2 < cos_t or float(ni @ nj) > -cos_t:
                continue
            q = min(a1, a2)
            if best is None or q > best[0]:
                best = (q, pj, nj, d, w)
        if best is None:
            continue
        q, pj, nj, d, w = best
        center = (pi + pj) / 2.0
        if np.linalg.norm(center - tip) < min_tip_clearance:   # keep clear of the tip
            continue
        key = tuple(np.round(center / dedup_grid).astype(int))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((center, d, w, q, pi.copy(), pj.copy()))

    rng.shuffle(pairs)
    n_pairs = max(1, int(np.ceil(n_grasps / max(1, n_poses_per_pair))))
    pairs = pairs[:n_pairs]

    grasps: list[GraspCandidate] = []
    for center, a, w, q, p1, p2 in pairs:
        offset = float(rng.uniform(0, 2 * np.pi))
        for k in range(n_poses_per_pair):
            roll = offset + 2 * np.pi * k / n_poses_per_pair
            standoff = float(rng.uniform(*standoff_range))
            approach, R = _pose_from_roll(center, a, roll, standoff)
            grasps.append(GraspCandidate(
                center=center.copy(), axis=a.copy(), approach=approach, R=R,
                width=w, standoff=standoff, quality=q, p1=p1, p2=p2))
    rng.shuffle(grasps)
    return grasps[:n_grasps]


def load_tool(tool_dir_path: str | pathlib.Path):
    """(mesh, tip, bend, name) from a tool asset directory."""
    import trimesh

    tool_dir_path = pathlib.Path(tool_dir_path)
    if not (tool_dir_path / "visual.obj").is_file():
        raise SystemExit(f"{tool_dir_path / 'visual.obj'} is not built; the tool mesh is prepared "
                         "locally from your licensed copy of the asset with "
                         "`simretract-build-assets --from-mesh` (docs/ASSETS.md)")
    meta = json.loads((tool_dir_path / "meta.json").read_text())
    mesh = trimesh.load(tool_dir_path / "visual.obj", process=False)
    return mesh, np.array(meta["suggested_tip"]), np.array(meta["suggested_bend"]), meta["name"]


def build_candidate_db(tool: str, n_grasps: int = N_GRASPS, seed: int = SAMPLER_SEED) -> dict[str, Any]:
    """Sample all three families for ``tool`` and return the database dict."""
    mesh, tip, _bend, name = load_tool(tool_dir(tool))
    flat = sample_flat_clamp_grasps(mesh, tip, n_grasps=n_grasps, seed=seed)
    anti = sample_antipodal_grasps(mesh, tip, n_grasps=n_grasps, seed=seed)
    washer = sample_washer_grasps(mesh, tip, seed=seed)
    widths = np.array([g.width for g in flat + anti + washer])
    log.info("%s: flat=%d antipodal=%d washer=%d | width mm: min=%.2f med=%.2f max=%.2f",
             name, len(flat), len(anti), len(washer), widths.min() * 1000,
             np.median(widths) * 1000, widths.max() * 1000)
    return {"tool": tool,
            "flat": [g.to_dict() for g in flat],
            "antipodal": [g.to_dict() for g in anti],
            "washer": [g.to_dict() for g in washer]}


#: Tolerances of ``--compare``: positions and widths to 1 um; rotation-matrix
#: entries to 1e-4 (0.006 degrees). A build whose coordinates lie within
#: 0.25 um of the released mesh (the recipe's grid fingerprint) can move the
#: face normals of millimetre-sized triangles (the closing axis of the
#: flat-clamp family) by up to a few 1e-5 rad while every centre, width and
#: standoff stays; a different mesh or sampler changes the candidate set outright.
COMPARE_ATOL_M = 1e-6
COMPARE_ATOL_ROT = 1e-4


def compare_candidate_dbs(new: dict[str, Any], shipped: dict[str, Any],
                          atol: float = COMPARE_ATOL_M, atol_rot: float = COMPARE_ATOL_ROT) -> list[str]:
    """Differences between a freshly sampled database and the shipped one."""
    problems = []
    for kind in CANDIDATE_KINDS:
        a, b = new.get(kind, []), shipped.get(kind, [])
        if len(a) != len(b):
            problems.append(f"{kind}: count {len(a)} vs shipped {len(b)}")
            continue
        for i, (ga, gb) in enumerate(zip(a, b)):
            for key, tol in (("center", atol), ("R", atol_rot), ("width", atol), ("standoff", atol)):
                if not np.allclose(np.asarray(ga[key]), np.asarray(gb[key]), atol=tol):
                    problems.append(f"{kind}[{i}].{key} differs")
                    break
            if len(problems) >= 20:
                problems.append("... (more)")
                return problems
    return problems


def plot_candidates(db: dict[str, Any], tool: str, out_png: pathlib.Path) -> None:
    """Two-view scatter of the candidates with closing axes (QA figure)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mesh, tip, _bend, name = load_tool(tool_dir(tool))
    all_g = [(k, GraspCandidate.from_dict(d)) for k in CANDIDATE_KINDS for d in db[k]]
    V = mesh.vertices[np.random.default_rng(0).choice(
        len(mesh.vertices), min(4000, len(mesh.vertices)), replace=False)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    colors = {"flat": "#1f77b4", "antipodal": "#ff7f0e", "washer": "#2ca02c"}
    for ax, (i0, i1, l0, l1) in zip(axes, [(1, 2, "y", "z"), (0, 2, "x", "z")]):
        ax.scatter(V[:, i0], V[:, i1], s=1, c="0.75", zorder=1)
        for kind, g in all_g:
            ax.scatter(g.center[i0], g.center[i1], s=12, c=colors[kind], zorder=3)
            a = g.axis * g.width * 1.6
            ax.plot([g.center[i0] - a[i0], g.center[i0] + a[i0]],
                    [g.center[i1] - a[i1], g.center[i1] + a[i1]],
                    c=colors[kind], lw=0.5, alpha=0.5, zorder=2)
        ax.scatter([tip[i0]], [tip[i1]], marker="*", s=200, c="red", zorder=4)
        ax.set_xlabel(l0)
        ax.set_ylabel(l1)
        ax.set_aspect("equal")
    handles = [plt.Line2D([], [], marker="o", ls="", c=c, label=k) for k, c in colors.items()]
    handles.append(plt.Line2D([], [], marker="*", ls="", c="red", markersize=12,
                              label="tip (keep-out 4cm)"))
    axes[0].legend(handles=handles, loc="upper left", fontsize=9)
    fig.suptitle(f"{name}: grasp candidates (centers + closing axes)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Sample the grasp-candidate database for a tool (needs trimesh + scipy).")
    ap.add_argument("--tool", default=DEFAULT_TOOL)
    ap.add_argument("--n-grasps", type=int, default=N_GRASPS)
    ap.add_argument("--seed", type=int, default=SAMPLER_SEED)
    ap.add_argument("--out", default=None,
                    help="write the database here (default: assets/grasp_candidates/<tool>.json)")
    ap.add_argument("--compare", action="store_true",
                    help="do not write; re-sample and compare with the shipped database")
    ap.add_argument("--plot", default=None, help="optional scatter figure path (.png)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    db = build_candidate_db(args.tool, n_grasps=args.n_grasps, seed=args.seed)
    if args.plot:
        plot_candidates(db, args.tool, pathlib.Path(args.plot))
    if args.compare:
        shipped = load_candidate_db(args.tool)
        problems = compare_candidate_dbs(db, shipped)
        for p in problems:
            print(f"[GRASPS] MISMATCH {p}", flush=True)
        print(f"[GRASPS] compare vs {candidates_path(args.tool)}: "
              f"{'OK' if not problems else 'FAIL'}", flush=True)
        return 1 if problems else 0
    out = pathlib.Path(args.out) if args.out else candidates_path(args.tool)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(db, f)
    print(f"[GRASPS] wrote {out}: flat={len(db['flat'])} antipodal={len(db['antipodal'])} "
          f"washer={len(db['washer'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
