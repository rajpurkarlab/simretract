"""Metric tool crop and foreground plane fit for real deployment.

The crop follows the visible shaft. A projected, depth-scaled fallback keeps
usable input when 3-D calibration fails. Legacy and training fits live in
``serve.plane``; this module runs only for positive metric lengths.
"""
import cv2
import numpy as np


def _camera(intrinsics):
    if intrinsics is None:
        return 433.0, 433.0, 320.0, 240.0
    k = np.asarray(intrinsics, dtype=np.float64)
    if k.shape == (3, 3):
        k = np.array([k[0, 0], k[1, 1], k[0, 2], k[1, 2]])
    if k.shape == (4,) and np.isfinite(k).all() and (k[:2] > 0).all():
        return tuple(k)
    raise ValueError("intrinsics must be a 3x3 matrix or (fx, fy, cx, cy)")


def _points(u, v, z, camera):
    fx, fy, cx, cy = camera
    return np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))


def _image_axis(v, u):
    """Estimate the shaft direction from its lower portion, not blade area."""
    p = np.column_stack((u, v)).astype(np.float64)
    c = np.median(p, axis=0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    axis = vt[0]
    if axis[1] > 0:
        axis = -axis
    for _ in range(2):
        s = (p - c) @ axis
        lower = s <= np.percentile(s, 72)
        # Equal longitudinal bins stop the wide blade or a filled hole
        # from dominating the shaft direction by pixel count.
        edges = np.linspace(s[lower].min(), s[lower].max() + 1e-6, 20)
        b = np.clip(np.digitize(s[lower], edges) - 1, 0, 18)
        centres = np.array([np.median(p[lower][b == i], axis=0)
                            for i in np.unique(b)])
        if len(centres) < 5:
            break
        c = centres.mean(axis=0)
        _, _, vt = np.linalg.svd(centres - c, full_matrices=False)
        axis = vt[0]
        if axis[1] > 0:
            axis = -axis
    return (p - c) @ axis, axis


def _foreground(mask, depth, s_image, lower_only=False, interior_only=True,
                raw_depth=None, boundary_mask=None):
    """Select depth-connected foreground before any 3-D SVD/MAD.

    A global depth MAD cannot distinguish an open metal frame from the
    background visible through it. Compare connected depth to nearby raw
    silhouette evidence, preserving differently inclined rails and thin
    foreground edges. Never fill the segmentation's holes.
    """
    valid = mask & np.isfinite(depth) & (depth > 50) & (depth < 900)
    # Exclude one pixel of the segmentation edge; depth there often mixes
    # metal and background. This affects fit support, not RGB visibility.
    interior = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    lo = cv2.erode(np.where(valid, depth, 10000).astype(np.float32),
                   np.ones((3, 3), np.uint8))
    hi = cv2.dilate(np.where(valid, depth, 0).astype(np.float32),
                    np.ones((3, 3), np.uint8))
    smooth = (hi - lo) <= np.maximum(8.0, depth * 0.04)
    support = valid & smooth & (interior if interior_only else mask)
    if lower_only:
        support &= s_image <= np.percentile(s_image[mask], 72)
    raw = depth if raw_depth is None else raw_depth
    raw_valid = mask & np.isfinite(raw) & (raw > 50) & (raw < 900)
    neighbours = np.zeros(mask.shape, np.uint8)
    padded = np.pad(raw, 1, constant_values=0)
    for dv, du in ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                   (0, 1), (1, -1), (1, 0), (1, 1)):
        # Constant padding prevents opposite image edges becoming neighbours.
        other = padded[1 + dv:1 + dv + mask.shape[0],
                       1 + du:1 + du + mask.shape[1]]
        neighbours += ((other > 50) & (other < 900)
                       & (np.abs(raw - other) <= np.maximum(4.0, 0.02 * raw)))
    evidence = raw_valid & (neighbours >= 2)
    if lower_only:
        evidence &= s_image <= np.percentile(s_image[mask], 72)
    s0, s1 = np.min(s_image[mask]), np.max(s_image[mask])
    bins = np.zeros(mask.shape, np.int32)
    bins[mask] = np.clip(((s_image[mask] - s0) / max(s1 - s0, 1e-6) * 24).astype(int), 0, 23)
    front = np.full(24, np.nan)
    for b in range(24):
        zz = np.sort(raw[evidence & (bins == b)])
        if len(zz) < 5:
            continue
        # Use the first persistent cluster, not a percentile: the two
        # one-pixel metal rails can be <3% of a filled hollow's mask.
        groups = np.split(zz, np.flatnonzero(np.diff(zz) > 6.0) + 1)
        for group in groups:
            if len(group) >= max(5, 0.01 * len(zz)):
                front[b] = np.percentile(group, 90)
                break
    boundary = mask if boundary_mask is None else boundary_mask
    border_distance = cv2.distanceTransform(boundary.astype(np.uint8), cv2.DIST_L2, 3)
    edge = evidence & (border_distance <= 1.5)
    if edge.sum() >= 20:
        # Compare each rail to its OWN raw silhouette depth. A single
        # global foreground range wrongly discards the far rail of a
        # tilted washer; nearest-edge evidence preserves that rail while
        # preventing a thin foreground rim being replaced by its interior
        # background after erosion.
        seed = np.ones(mask.shape, np.uint8)
        seed[edge] = 0
        _, labels_edge = cv2.distanceTransformWithLabels(
            seed, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
        lut = np.full(int(labels_edge.max()) + 1, np.nan)
        lut[labels_edge[edge]] = raw[edge]
        reference = lut[labels_edge]
    else:
        reference = front[bins]
    support &= np.isfinite(reference) & (depth <= reference + 20.0)
    if not interior_only:
        # Thin, connected raw rails may calibrate an axis but never supply
        # the final plane when all their interior depth is unavailable.
        support = evidence & np.isfinite(reference) & (raw <= reference + 20.0)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        support.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    span = float(np.ptp(s_image[mask]))
    min_size = max(24, int(support.sum() * 0.015))
    candidates = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_size:
            continue
        m = labels == i
        ss = s_image[m]
        if np.ptp(ss) < max(8, span * 0.12):
            continue
        # A background component enclosed inside a filled ring must not
        # win by area. Require support near the original outer silhouette,
        # not a new artificial edge introduced by the top crop.
        if interior_only and np.count_nonzero(m & (border_distance <= 3.0)) < 12:
            continue
        candidates.append(i)
    if not candidates:
        return np.zeros_like(mask)
    return np.isin(labels, candidates)


def _balanced_indices(s, max_points=1800):
    """Deterministic sampling with equal density along the shaft."""
    bins = np.clip(((s - s.min()) / max(float(np.ptp(s)), 1e-6) * 24).astype(int), 0, 23)
    per_bin = max(8, max_points // 24)
    chosen = []
    for i in range(24):
        ids = np.flatnonzero(bins == i)
        if len(ids) > per_bin:
            ids = ids[np.linspace(0, len(ids) - 1, per_bin).astype(int)]
        chosen.extend(ids)
    return np.asarray(chosen, dtype=int)


def _keep_lower_components(original, cut, coordinate):
    """Drop distal fragments detached by THIS cut, preserving both rails.

    Pre-existing disconnected segmentation pieces are handled separately.
    For each original component split by the crop, keep every remainder
    reaching its lower portion; this preserves two long hollow-handle rails
    but excludes a return-bent blade island hanging below the cutoff.
    """
    _, original_labels = cv2.connectedComponents(original.astype(np.uint8), connectivity=8)
    n, labels = cv2.connectedComponents(cut.astype(np.uint8), connectivity=8)
    groups = {}
    for i in range(1, n):
        v, u = np.nonzero(labels == i)
        if len(v):
            groups.setdefault(int(original_labels[v[0], u[0]]), []).append(i)
    clean = cut.copy()
    for source, pieces in groups.items():
        if len(pieces) < 2:
            continue
        remaining = cut & (original_labels == source)
        lo, hi = np.percentile(coordinate[remaining], (1, 99))
        lower_end = lo + 0.4 * (hi - lo)
        for i in pieces:
            piece = labels == i
            if not np.any(coordinate[piece] <= lower_end):
                clean[piece] = False
    return clean


def _plane(points, s, threshold=0.0025, reference_span=None, reject_narrow=True):
    """Deterministic RANSAC, then TLS on inliers, with width/noise checks."""
    if len(points) < 80:
        return None, np.zeros(len(points), bool), "too few foreground points"
    idx = _balanced_indices(s)
    p = points[idx]
    if len(p) < 60:
        return None, np.zeros(len(points), bool), "too few foreground samples"
    rng = np.random.default_rng(731)
    triples = rng.integers(0, len(p), (160, 3))
    a, b, c = p[triples[:, 0]], p[triples[:, 1]], p[triples[:, 2]]
    normals = np.cross(b - a, c - a)
    norm = np.linalg.norm(normals, axis=1)
    # Crop calibration requires broad support. A final narrow-view retry
    # may use small triangles, but exactly collinear points still cannot
    # determine a surface normal.
    good = norm > (2e-5 if reject_narrow else 1e-12)
    normals, a = normals[good], a[good]
    if not len(normals):
        return None, np.zeros(len(points), bool), "shaft too narrow for a plane"
    normals /= norm[good, None]
    offsets = np.einsum("ij,ij->i", a, normals)
    residual = np.abs(np.einsum("ij,kj->ik", p, normals) - offsets[None, :])
    votes = residual < threshold
    counts = votes.sum(axis=0)
    best = int(np.argmax(counts))
    if counts[best] < max(60, 0.5 * len(p)):
        return None, np.zeros(len(points), bool), "no foreground plane consensus"
    keep = np.abs(points @ normals[best] - offsets[best]) < threshold
    for _ in range(3):
        if keep.sum() < 80:
            return None, keep, "too few plane inliers"
        centre = points[keep].mean(axis=0)
        _, sv, vt = np.linalg.svd(points[keep] - centre, full_matrices=False)
        n = vt[-1]
        keep = np.abs((points - centre) @ n) <= threshold
    if keep.sum() < 80:
        return None, keep, "too few plane inliers"
    centre = points[keep].mean(axis=0)
    _, sv, vt = np.linalg.svd(points[keep] - centre, full_matrices=False)
    rms = sv / np.sqrt(keep.sum())
    # A narrow strip (even with many pixels) does not determine a reliable
    # plane normal. Nor does an elongated cloud with noise as wide as it is.
    if reject_narrow and (rms[1] < 0.0015 or rms[2] > 0.35 * rms[1]):
        return None, keep, "shaft too narrow/noisy for a plane"
    span = float(np.ptp(s)) if reference_span is None else reference_span
    if rms[0] > 0.15 or np.ptp(s[keep]) < 0.55 * max(span, 1e-6):
        return None, keep, "plane lacks longitudinal support"
    n = vt[-1]
    if n[2] > 0:
        n = -n
    t0 = vt[0]
    t1 = np.cross(n, t0)
    plane = {"n": n.astype(np.float32), "c": centre.astype(np.float32),
             "t0": t0.astype(np.float32), "t1": t1.astype(np.float32)}
    return plane, keep, "ok"


def _metric_plane_inputs(mask, depth_mm, length_m=0.01, intrinsics=None):
    """Return the metric-mode visibility, excluded tip, and retained plane.

    The cut is contiguous in the projected shaft direction. Depth controls
    the shaft's metric calibration, not a separate cut coordinate for each
    noisy/hollow pixel. A weak plane is returned as None, never replaced by
    an uncut/bent-tip plane. Inputs are not modified.
    """
    original = np.asarray(mask)
    depth = np.asarray(depth_mm, dtype=np.float32)
    if original.ndim != 2 or depth.shape != original.shape:
        raise ValueError("mask and depth must be matching HxW arrays")
    length_m = float(length_m)
    if not np.isfinite(length_m) or length_m < 0:
        raise ValueError("metric cut length must be finite and nonnegative")
    mask_bool = original > 0
    empty = np.zeros(original.shape, np.uint8)
    result = {"mask": empty.copy(), "removed_mask": empty.copy(),
              "support_mask": empty.copy(), "plane": None, "status": "insufficient mask"}
    if length_m == 0:
        result.update(mask=original.copy(), status="disabled")
        return result
    v, u = np.nonzero(mask_bool)
    if len(v) < 100:
        return result
    camera = _camera(intrinsics)
    # 3x3 median suppresses individual flying pixels without a 5x5 erosion
    # of narrow metal rails. Invalid raw depths stay invalid for fitting.
    filtered = cv2.medianBlur(np.where(np.isfinite(depth), depth, 0), 3)
    filtered[(~np.isfinite(depth)) | (depth <= 50) | (depth >= 900)] = 0
    s, _ = _image_axis(v, u)
    s_image = np.full(mask_bool.shape, np.nan)
    s_image[v, u] = s
    lower = _foreground(mask_bool, filtered, s_image, lower_only=True,
                        raw_depth=depth)
    if lower.sum() < 80 or np.ptp(s_image[lower]) < 0.45 * np.ptp(s):
        # A thin edge can still calibrate a longitudinal crop, although it
        # cannot constrain a plane. This fallback is used for the axis only;
        # the final fit continues to require interior foreground support.
        lower = _foreground(mask_bool, depth, s_image, lower_only=True,
                            interior_only=False, raw_depth=depth)
        axis_depth = depth
    else:
        axis_depth = filtered
    vl, ul = np.nonzero(lower)
    if len(vl) < 80:
        result["status"] = "insufficient connected shaft depth"
        return result
    p = _points(ul, vl, axis_depth[vl, ul] / 1000.0, camera)
    sl = s_image[vl, ul]
    if np.ptp(sl) < 0.45 * np.ptp(s):
        result["status"] = "foreground does not span the shaft"
        return result
    # Centreline samples give rings and wide washers equal weight along
    # length. Two distant rails must not dominate PCA by transverse area.
    bs = np.clip(((sl - sl.min()) / max(float(np.ptp(sl)), 1e-6) * 20).astype(int), 0, 19)
    centres = np.array([np.median(p[bs == b], axis=0) for b in np.unique(bs)])
    centre = centres.mean(axis=0)
    _, sv, vt = np.linalg.svd(centres - centre, full_matrices=False)
    axis = vt[0]
    axial = (p - centre) @ axis
    corr = np.corrcoef((centres - centre) @ axis,
                       [np.median(sl[bs == b]) for b in np.unique(bs)])[0, 1]
    if corr < 0:
        axis = -axis
    if (not np.isfinite(corr) or abs(corr) < 0.75
            or sv[0] < 1.8 * sv[1] or np.ptp(axial) > 0.4):
        result["status"] = "unreliable shaft axis"
        return result
    # Closest point on a calibrated 3-D shaft line for each image ray.
    # This gives one geometric tip boundary even across holes/invalid z.
    rays = _points(u, v, np.ones(len(u)), camera)
    rr = np.einsum("ij,ij->i", rays, rays)
    ra = rays @ axis
    rc = rays @ centre
    denom = rr - ra * ra
    if np.any(denom < 0.015):
        result["status"] = "shaft points into camera; metric length ambiguous"
        return result
    preliminary_plane, _, _ = _plane(p, sl, reference_span=0.72 * np.ptp(s))
    t = (ra * rc - float(centre @ axis) * rr) / denom
    if preliminary_plane is not None:
        pn, pc = preliminary_plane["n"], preliminary_plane["c"]
        # Recover metric coordinates across BOTH rails of a tilted wide
        # face; nearest-to-line projection alone shears its crop boundary.
        ray_dot_n = rays @ pn
        z_plane = np.divide(float(pc @ pn), ray_dot_n,
                            out=np.full(len(rays), np.nan), where=np.abs(ray_dot_n) > 0.03)
        if np.all(np.isfinite(z_plane) & (z_plane > 0.05) & (z_plane < 1.5)):
            axis = axis - float(axis @ pn) * pn
            axis /= np.linalg.norm(axis)
            t = (rays * z_plane[:, None] - centre) @ axis
    t0, tip = np.percentile(t, (1, 99))
    span = tip - t0
    if span < length_m + 0.015 or span > 0.4:
        result["status"] = "too little visible shaft for requested cut"
        return result
    retained = t <= tip - length_m
    cut = np.zeros_like(mask_bool)
    cut[v[retained], u[retained]] = True
    coordinate = np.full(mask_bool.shape, np.nan)
    coordinate[v, u] = t
    clean = _keep_lower_components(mask_bool, cut, coordinate)
    detached = int(np.count_nonzero(cut & ~clean))
    cut = clean
    result["removed_mask"] = ((mask_bool & ~cut) * 255).astype(np.uint8)
    result["mask"] = (cut * 255).astype(np.uint8)
    # Foreground selection is repeated after cutting, so excluded blade
    # points cannot support the final fit or silently reappear via fallback.
    support = _foreground(cut, filtered, s_image, raw_depth=depth,
                          boundary_mask=mask_bool)
    vs, us = np.nonzero(support)
    ps = _points(us, vs, filtered[vs, us] / 1000.0, camera)
    plane, keep, status = _plane(ps, s_image[vs, us])
    narrow_retry = status in ("shaft too narrow for a plane",
                              "shaft too narrow/noisy for a plane")
    if narrow_retry:
        # Restore a depth-derived normal for side/narrow views. Keep the
        # strict preliminary fit above so this does not move the top cut.
        plane, keep, status = _plane(ps, s_image[vs, us], reject_narrow=False)
    if plane is not None:
        # A wide washer can determine a plane while its narrow stem loses
        # interior pixels. Validate that plane against un-eroded connected
        # depth across the remainder, so neither a short bent patch nor a
        # background patch can stand in for the full visible shaft.
        check = _foreground(cut, depth, s_image, interior_only=False,
                            raw_depth=depth, boundary_mask=mask_bool)
        vc, uc = np.nonzero(check)
        pc = _points(uc, vc, depth[vc, uc] / 1000.0, camera)
        agrees = np.abs((pc - plane["c"]) @ plane["n"]) <= 0.0035
        if (agrees.sum() < 80 or agrees.mean() < 0.55
                or np.ptp(s_image[vc[agrees], uc[agrees]]) < 0.55 * np.ptp(s_image[cut])):
            plane, status = None, "plane lacks support across retained shaft"
    result.update(plane=plane, status="cut applied; " + status,
                  axis=axis.astype(np.float32), axis_c=centre.astype(np.float32),
                  visible_length_m=float(span), cut_length_m=length_m)
    if detached:
        result["status"] += "; detached tip fragment removed"
    result["support_mask"][vs[keep], us[keep]] = 255
    if plane is not None and not narrow_retry:
        # Reject valid depth behind the retained shaft surface: e.g. tissue
        # visible through a hole accidentally filled by segmentation. Keep
        # unknown-depth RGB and closer foreground; neither fits the plane.
        n, c = plane["n"], plane["c"]
        rn = rays @ n
        expected = np.divide(float(c @ n), rn, out=np.full(len(rn), np.nan),
                             where=np.abs(rn) > 0.05)
        observed = filtered[v, u] / 1000.0
        behind = (observed > 0.05) & np.isfinite(expected) & (expected > 0.05) \
                 & (observed > expected + 0.008)
        result["mask"][v[behind], u[behind]] = 0
    # A recovered narrow normal is an orientation cue only. It must not
    # newly remove RGB/depth pixels with an uncertain ray-plane depth.
    return result


def _projected_metric_fallback(mask, depth_mm, length_m, intrinsics, reason):
    """Keep visible SAM pixels when 3-D crop calibration fails.

    Approximate the requested length in the image shaft direction using
    current connected foreground depth. This projected cut may exclude more
    physical length on a tilted shaft; it is not a recovered 3-D calibration.
    The excluded tip never supplies the retained plane fit.
    """
    original = np.asarray(mask) > 0
    depth = np.asarray(depth_mm, dtype=np.float32)
    v, u = np.nonzero(original)
    if len(v) < 100:
        return None
    camera = _camera(intrinsics)
    filtered = cv2.medianBlur(np.where(np.isfinite(depth), depth, 0), 3)
    filtered[(~np.isfinite(depth)) | (depth <= 50) | (depth >= 900)] = 0
    s, axis = _image_axis(v, u)
    coordinate = np.full(original.shape, np.nan)
    coordinate[v, u] = s
    lower = _foreground(original, filtered, coordinate, lower_only=True,
                        raw_depth=depth)
    calibration_depth = filtered
    if lower.sum() < 80:
        lower = _foreground(original, depth, coordinate, lower_only=True,
                            interior_only=False, raw_depth=depth)
        calibration_depth = depth
    if lower.sum() < 80:
        return None
    z = float(np.median(calibration_depth[lower])) / 1000.0
    metres_per_pixel = z * np.hypot(axis[0] / camera[0], axis[1] / camera[1])
    if not np.isfinite(metres_per_pixel) or metres_per_pixel <= 0:
        return None
    low, tip = np.percentile(s, (1, 99))
    if (tip - low) * metres_per_pixel < length_m + 0.015:
        return None
    cut_pixels = length_m / metres_per_pixel
    cut = original & (coordinate <= tip - cut_pixels)
    cut = _keep_lower_components(original, cut, coordinate)
    if cut.sum() < 100:
        return None
    # The same foreground and retained-shaft checks as the calibrated path
    # prevent hollow/background depths or the excluded blade from fitting.
    support = _foreground(cut, filtered, coordinate, raw_depth=depth,
                          boundary_mask=original)
    vs, us = np.nonzero(support)
    points = _points(us, vs, filtered[vs, us] / 1000.0, camera)
    plane, keep, status = _plane(points, coordinate[vs, us])
    narrow = status in ("shaft too narrow for a plane",
                        "shaft too narrow/noisy for a plane")
    if narrow:
        plane, keep, status = _plane(points, coordinate[vs, us], reject_narrow=False)
    if plane is not None:
        check = _foreground(cut, depth, coordinate, interior_only=False,
                            raw_depth=depth, boundary_mask=original)
        vc, uc = np.nonzero(check)
        pc = _points(uc, vc, depth[vc, uc] / 1000.0, camera)
        agrees = np.abs((pc - plane["c"]) @ plane["n"]) <= 0.0035
        if (agrees.sum() < 80 or agrees.mean() < 0.55
                or np.ptp(coordinate[vc[agrees], uc[agrees]]) < 0.55 * np.ptp(coordinate[cut])):
            plane, status = None, "plane lacks support across retained shaft"
    retained_mask = (cut * 255).astype(np.uint8)
    if plane is not None and not narrow:
        rays = _points(u, v, np.ones(len(u)), camera)
        rn = rays @ plane["n"]
        expected = np.divide(float(plane["c"] @ plane["n"]), rn,
                             out=np.full(len(rn), np.nan), where=np.abs(rn) > 0.05)
        observed = filtered[v, u] / 1000.0
        behind = ((observed > 0.05) & np.isfinite(expected) & (expected > 0.05)
                  & (observed > expected + 0.008))
        retained_mask[v[behind], u[behind]] = 0
    support_mask = np.zeros(original.shape, np.uint8)
    support_mask[vs[keep], us[keep]] = 255
    return {"mask": retained_mask,
            "removed_mask": ((original & ~cut) * 255).astype(np.uint8),
            "support_mask": support_mask, "plane": plane,
            "status": "projected cut fallback (" + reason + "); " + status,
            "cut_mode": "projected_depth_fallback", "cut_length_m": length_m,
            "cut_pixels": float(cut_pixels), "calibration_depth_m": z}


def metric_plane_inputs(mask, depth_mm, length_m=0.01, intrinsics=None):
    """Fit the retained shaft, using a projected cut if calibrated cropping is empty.

    Disabled cuts and tracker dropouts pass through. No past mask or plane
    is reused.
    """
    result = _metric_plane_inputs(mask, depth_mm, length_m, intrinsics)
    if float(length_m) > 0 and not np.any(result["mask"]):
        fallback = _projected_metric_fallback(
            mask, depth_mm, float(length_m), intrinsics, result["status"])
        if fallback is not None:
            return fallback
    return result
