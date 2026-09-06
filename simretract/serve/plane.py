"""Plane-normal fits on a masked depth image, shared by the data pipeline
and the policy server.

Both fits back-project the tool-mask pixels of a depth image with the
simulated wrist-camera pinhole (58 deg vertical FOV on a 640x480 image:
fx = fy = 240 / tan(29 deg), cx = 320, cy = 240), fit a total-least-squares
plane by SVD, drop the 3-sigma orthogonal outliers, refit, and orient the
unit normal toward the camera (n_z < 0). The normal is the 3-vector that the
``qpos_plane`` state mode appends to the joint state, so the two fits must
build the training-time and the serving-time conditioning the same way.

There are deliberately TWO functions:

``fit_plane_sidecar``
    The training-data fit used by ``simretract.data.plane_sidecar``. It runs
    on clean rendered depth from the ground-truth mask, has no noise filter,
    and returns a planarity confidence (ratio of the two smallest singular
    values). Its output is what every shipped checkpoint was trained on.

``fit_plane_served``
    The serving-time fit used by the policy server on real (or simulated)
    query frames. It adds the depth noise filter that a RealSense D405 needs
    (5x5 median blur, a 50..900 mm depth window and a 5-MAD trim around the
    median depth, all switchable with ``plane_filter``) and the optional
    top-cut that removes the tip end of the tool from the fit, guarded so
    that a degenerate reduced cloud never swings the normal by more than
    ``top_cut_max_deg``.

Neither function may be altered without retraining: the numeric steps below
are the reference implementation the checkpoints depend on.
"""
from __future__ import annotations

import numpy as np

PLANE_FY: float = 240.0 / np.tan(np.radians(58.0) / 2)
"""Focal length (px) of the 640x480 wrist camera with a 58 deg vertical FOV."""
PLANE_CX: float = 320.0
PLANE_CY: float = 240.0

MIN_POINTS: int = 200
"""Minimum masked depth pixels for a fit; fewer gives the degenerate marker."""
MIN_REFIT_POINTS: int = 100
"""Minimum inliers for the second (3-sigma trimmed) fit."""

PlaneFit = tuple[np.ndarray, np.ndarray]
"""(unit normal float32 (3,), centroid float64 (3,)) in the camera frame."""


def fit_plane_sidecar(dep_mm: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit the training-time plane normal of one cached frame.

    Args:
        dep_mm: (H, W) depth in millimetres (uint16 or any integer/float
            dtype); 0 marks an invalid pixel.
        mask: (H, W) ground-truth tool mask; any value > 0 is tool.

    Returns:
        ``(normal, conf)``: ``normal`` is a float32 unit vector oriented
        toward the camera (``normal[2] < 0``) or ``(0, 0, 0)`` when fewer
        than ``MIN_POINTS`` valid pixels exist; ``conf`` is ``s2 / s3`` of the
        (refitted) point cloud's singular values (large = planar, below ~3 =
        degenerate) or 0 for a missing fit.
    """
    m = (mask > 0) & (dep_mm > 0)
    v, u = np.nonzero(m)
    if len(v) < MIN_POINTS:
        return np.zeros(3, np.float32), 0.0
    z = dep_mm[v, u].astype(np.float64) / 1000.0
    P = np.stack([(u - PLANE_CX) / PLANE_FY * z, (v - PLANE_CY) / PLANE_FY * z, z], 1)
    c = P.mean(0)
    _, s, Vt = np.linalg.svd(P - c, full_matrices=False)
    n = Vt[2]
    d = np.abs((P - c) @ n)
    keep = d < 3.0 * max(d.std(), 1e-6)
    if keep.sum() >= MIN_REFIT_POINTS:
        c2 = P[keep].mean(0)
        _, s, Vt = np.linalg.svd(P[keep] - c2, full_matrices=False)
        n = Vt[2]
    if n[2] > 0:
        n = -n
    return n.astype(np.float32), float(np.float32(s[1] / max(s[2], 1e-9)))


def plane_top_cut(mask: np.ndarray, frac: float) -> np.ndarray:
    """Remove the top ``frac`` of the tool's pixel length from a mask.

    Some retractor tips carry a dent or fork that tilts the fitted plane, so
    the serving fit can drop that end before fitting. The tool length is the
    1st..99th percentile span of ALL mask pixels along the mask's principal
    image axis, so a mask split into two islands is still measured as one
    tool and only the upper island loses pixels; "top" is the end higher in
    the image. Returns the input unchanged for ``frac <= 0``, for masks with
    fewer than 300 pixels, or when the cut would leave fewer than 200.
    """
    if frac <= 0:
        return mask
    v, u = np.nonzero(mask > 0)
    if len(v) < 300:
        return mask
    pts = np.stack([u, v], 1).astype(np.float64)
    c = pts.mean(0)
    _, sv, Vt = np.linalg.svd(pts - c, full_matrices=False)
    ax = Vt[0]
    if ax[1] > 0:
        ax = -ax
    t = (pts - c) @ ax
    t0, t1 = np.percentile(t, 1), np.percentile(t, 99)
    keep = t <= t1 - frac * (t1 - t0)
    if keep.sum() < 200:
        return mask
    out = np.zeros_like(mask)
    out[v[keep], u[keep]] = 255
    return out


def fit_plane_trimmed(dep: np.ndarray, mask_in: np.ndarray, plane_filter: bool) -> PlaneFit | None:
    """Trimmed total-least-squares plane on the masked depth (serving path).

    ``dep`` is depth in millimetres, already median-blurred by the caller
    when ``plane_filter`` is on. Pixels outside the 50..900 mm window are
    dropped; with ``plane_filter`` the cloud is additionally trimmed to
    ``|z - median| <= max(5 * MAD, 10 mm)`` when at least ``MIN_POINTS``
    survive. Returns ``(normal, centroid)`` or ``None`` for a degenerate
    frame (fewer than ``MIN_POINTS`` pixels).
    """
    m = (mask_in > 0) & (dep > 50) & (dep < 900)
    v, u = np.nonzero(m)
    if len(v) < MIN_POINTS:
        return None
    z = dep[v, u].astype(np.float64) / 1000.0
    if plane_filter:
        zm = np.median(z)
        mad = np.median(np.abs(z - zm))
        keep0 = np.abs(z - zm) <= max(5.0 * mad, 0.01)
        if keep0.sum() >= MIN_POINTS:
            u, v, z = u[keep0], v[keep0], z[keep0]
    P = np.stack([(u - PLANE_CX) / PLANE_FY * z, (v - PLANE_CY) / PLANE_FY * z, z], 1)
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    n = Vt[2]
    d = np.abs((P - c) @ n)
    keep = d < 3.0 * max(d.std(), 1e-6)
    if keep.sum() >= MIN_REFIT_POINTS:
        c = P[keep].mean(0)
        _, _, Vt = np.linalg.svd(P[keep] - c, full_matrices=False)
        n = Vt[2]
    if n[2] > 0:
        n = -n
    n = (n / np.linalg.norm(n)).astype(np.float32)
    return n, c


def fit_plane_served(
    dep_mm: np.ndarray,
    mask: np.ndarray,
    *,
    plane_filter: bool = True,
    top_cut: float = 0.0,
    top_cut_max_deg: float = 20.0,
) -> PlaneFit | None:
    """Fit the serving-time plane of one query frame.

    Args:
        dep_mm: (H, W) depth in millimetres as delivered by the camera (or
            the simulator); 0 = invalid.
        mask: (H, W) tool mask from the segmenter (or the ground truth in
            simulation); any value > 0 is tool.
        plane_filter: apply the depth noise filter (5x5 median blur before
            the fit plus the 5-MAD trim inside it). Flying pixels and
            background seen through hollow tool parts inflate the cloud's
            depth extent and flip the SVD normal sideways on front-facing
            poses; the filter removes them. It is on for every shipped
            checkpoint's benchmark and deployment.
        top_cut: fraction of the tool length removed from the top end before
            the fit (``plane_top_cut``); 0 disables it.
        top_cut_max_deg: the cut is a small correction, so if the cut fit
            deviates from the uncut fit by more than this angle the reduced
            cloud is considered degenerate and the uncut plane is kept. When
            only the cut fit succeeds it is used as is.

    Returns:
        ``(normal, centroid)`` with a float32 unit normal oriented toward
        the camera, or ``None`` when no fit is possible; the server turns
        ``None`` into the zero vector, the "no plane" marker.
    """
    dep = np.asarray(dep_mm)
    mask_in = np.asarray(mask)
    if plane_filter:
        import cv2

        dep = cv2.medianBlur(dep.astype(np.float32), 5)
    res = fit_plane_trimmed(dep, mask_in, plane_filter)
    tc = float(top_cut)
    if tc > 0:
        res_cut = fit_plane_trimmed(dep, plane_top_cut(mask_in, tc), plane_filter)
        if res is None:
            res = res_cut
        elif res_cut is not None:
            ang = np.degrees(np.arccos(np.clip(abs(float(res[0] @ res_cut[0])), 0.0, 1.0)))
            if ang <= top_cut_max_deg:
                res = res_cut
    return res
