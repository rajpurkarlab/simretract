"""Segmentation-artifact and depth-sensor augmentation for masked-input
training (the v4.2 recipe every shipped checkpoint was trained with).

The policy never sees a raw camera frame: the RGB and depth streams are
masked by a tool segmentation, and at deployment that mask comes from a
learned segmenter (SAM 3) on a RealSense D405. Training on the simulator's
perfect masks and clean depth would leave the policy brittle to both, so the
trainer corrupts them on the fly:

- ``corrupt_mask``: SAM-style mask errors on the ground-truth tool mask,
  sampled per draw at mild/medium/strong severity with a clean-draw share:
  anisotropic over/under-segmentation, smooth boundary wobble (signed
  distance field + low-frequency noise), false-positive blobs bleeding into
  the background, interior holes, inconsistent filling of enclosed hollows,
  and rare half-plane chunk cuts.
- ``random_fill``: what shows through a false-positive region, i.e. an
  out-of-focus cluttered background or a plain surface, drawn from a curated
  palette with jitter so the policy never learns one leak colour.
- ``augment_depth``: D405 stereo artifacts on clean rendered depth
  (calibration warp, distance-dependent speckle and disparity quantisation,
  edge fattening / flying pixels, one-sided occlusion shadows, invalid
  patches).
- ``random_depth_fill``: the depth a false-positive region would carry,
  a smooth field with object-shaped patches at distinct depths.

Every function takes a ``numpy.random.Generator`` and is deterministic for
a given generator state, which the golden-output unit test relies on.
"""
import cv2
import numpy as np


def smooth_noise(h, w, rng, scale):
    """Low-frequency noise field in [0,1] with feature size ~scale px."""
    gh, gw = max(2, int(h / scale)), max(2, int(w / scale))
    g = rng.random((gh, gw)).astype(np.float32)
    return np.clip(cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC), 0, 1)


def _irregular_blob(shape, cy, cx, r, rng):
    """Noise-distorted ellipse -> boolean blob mask."""
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    a = max(2.0, r * rng.uniform(0.5, 1.6))
    b = max(2.0, r * rng.uniform(0.5, 1.6))
    th = rng.uniform(0, np.pi)
    x = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
    y = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
    d = (x / a) ** 2 + (y / b) ** 2
    n = smooth_noise(h, w, rng, rng.uniform(15, 40))
    return (d + (n - 0.5) * rng.uniform(0.4, 1.2)) < 1.0


def _fill_holes(m8):
    """Fill enclosed zero-regions (e.g. the rake's ring interior)."""
    h, w = m8.shape
    ff = np.pad(m8, 1)  # guarantee a background-connected border
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
    return m8 | (ff[1:-1, 1:-1] == 0).astype(np.uint8)


SEVERITIES = ((0.40, 0.4), (0.35, 0.8), (0.10, 1.4))  # (prob, magnitude)
CLEAN_P = 0.15
CHUNK_P = 0.03


def corrupt_mask(mask, rng):
    """Simulated segmenter output given the GT tool mask (uint8 0/1)."""
    m = (mask > 0)
    if m.sum() < 20 or rng.random() < CLEAN_P:  # tiny mask or clean draw
        return m.astype(np.uint8)
    h, w = m.shape
    roll, acc, mag = rng.random(), 0.0, 1.0
    for p, sev in SEVERITIES:
        acc += p
        if roll < acc:
            mag = sev
            break
    m8 = m.astype(np.uint8)
    # 0) the segmenter inconsistently includes enclosed hollows (rake ring):
    #    with ~45% probability fill holes -> interior becomes a leak region
    if rng.random() < 0.45:
        m8 = _fill_holes(m8)
    # 1) anisotropic over/under-segmentation
    kx = 1 + int(rng.integers(1, 4) * mag * 2)
    ky = 1 + int(rng.integers(1, 4) * mag * 2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx, ky))
    m8 = cv2.dilate(m8, k) if rng.random() < 0.65 else cv2.erode(m8, k)
    # 2) smooth boundary wobble via signed distance + low-freq noise
    d_in = cv2.distanceTransform(m8, cv2.DIST_L2, 3)
    d_out = cv2.distanceTransform(1 - m8, cv2.DIST_L2, 3)
    amp = rng.uniform(2.0, 7.0) * mag
    n = (smooth_noise(h, w, rng, rng.uniform(10, 45)) - 0.5) * 2 * amp
    out = (d_in - d_out + n) > 0
    # 3) false-positive blobs bleeding outward from the boundary
    m8 = out.astype(np.uint8)
    edge = np.argwhere(m8 - cv2.erode(m8, np.ones((3, 3), np.uint8)))
    n_blob = int(rng.integers(0, 2 + int(3 * mag)))
    for _ in range(n_blob):
        if not len(edge):
            break
        cy, cx = edge[rng.integers(len(edge))]
        out |= _irregular_blob((h, w), cy, cx,
                               rng.uniform(4, 12) * (0.5 + mag), rng)
    # 4) interior holes (false negatives)
    inside = np.argwhere(out & m)
    n_hole = int(rng.integers(0, 1 + int(2 * mag)))
    for _ in range(n_hole):
        if not len(inside):
            break
        cy, cx = inside[rng.integers(len(inside))]
        out &= ~_irregular_blob((h, w), cy, cx,
                                rng.uniform(3, 10) * (0.5 + mag), rng)
    # 5) rare gross failure: half-plane chunk cut
    if rng.random() < CHUNK_P:
        pts = np.argwhere(m)
        cy, cx = pts[rng.integers(len(pts))]
        th = rng.uniform(0, 2 * np.pi)
        yy, xx = np.ogrid[:h, :w]
        out &= ((xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)) <= 0
    return out.astype(np.uint8)


# Curated background palette (RGB): neutrals, cool blues/teals, greens,
# warm organic tones, metallics. Sampled with jitter, never exact.
_BASE = np.array([
    [245, 247, 248], [240, 237, 232], [221, 224, 226], [194, 199, 202],
    [155, 161, 165], [59, 64, 69], [27, 30, 32],
    [201, 218, 226], [143, 168, 200], [90, 122, 150], [46, 74, 102],
    [47, 126, 133], [78, 140, 138],
    [195, 219, 208], [126, 156, 134], [76, 122, 99], [35, 69, 60],
    [232, 204, 178], [198, 154, 120], [223, 167, 158], [122, 74, 52],
    [142, 59, 55], [78, 31, 30],
    [214, 219, 223], [154, 160, 166], [110, 117, 124], [176, 141, 87],
], np.float32)
# High-saturation accents: small areas only, sparingly.
_ACCENT = np.array([
    [192, 51, 43], [217, 129, 44], [232, 195, 60], [76, 175, 80],
    [107, 76, 138],
], np.float32)


def _rand_color(rng, small=False):
    """Palette color with jitter; accents only for small objects."""
    if small and rng.random() < 0.3:
        c = _ACCENT[rng.integers(len(_ACCENT))]
    else:
        c = _BASE[rng.integers(len(_BASE))]
    col = c * rng.uniform(0.85, 1.15) + rng.uniform(-12, 12, 3)
    return [int(x) for x in np.clip(col, 0, 255)]


def _draw_object(img, rng):
    """One random 'object': ellipse / rotated rect / polygon, shaded."""
    h, w = img.shape[:2]
    cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
    size = int(rng.uniform(20, 180))
    col = _rand_color(rng, small=size < 70)
    kind = int(rng.integers(0, 3))
    if kind == 0:
        ax = (max(6, int(size * rng.uniform(0.4, 1.0))),
              max(6, int(size * rng.uniform(0.4, 1.0))))
        cv2.ellipse(img, (cx, cy), ax, float(rng.uniform(0, 180)),
                    0, 360, col, -1)
    elif kind == 1:
        box = cv2.boxPoints(((cx, cy),
                             (size * rng.uniform(0.4, 1.2),
                              size * rng.uniform(0.4, 1.2)),
                             float(rng.uniform(0, 180))))
        cv2.fillPoly(img, [box.astype(np.int32)], col)
    else:
        pts = np.stack([cx + rng.uniform(-size, size, 5),
                        cy + rng.uniform(-size, size, 5)], 1)
        cv2.fillPoly(img, [pts.astype(np.int32)], col)


def random_fill(h, w, rng):
    """Leak-region fill imitating out-of-focus background clutter:
    a canvas of random overlapping colored objects, defocus-blurred,
    plus mild sensor noise. 15% of draws: plain surface (wall/table)."""
    if rng.random() < 0.15:  # plain surface
        img = np.full((h, w, 3), _rand_color(rng), np.float32)
        img += (smooth_noise(h, w, rng, rng.uniform(40, 120))[..., None]
                - 0.5) * rng.uniform(10, 60)
    else:  # cluttered objects
        img8 = np.full((h, w, 3), _rand_color(rng), np.uint8)
        for _ in range(int(rng.integers(4, 15))):
            _draw_object(img8, rng)
        img = img8.astype(np.float32)
        # uneven illumination across the scene
        img *= (0.55 + 0.9 * smooth_noise(h, w, rng,
                                          rng.uniform(60, 200)))[..., None]
    k = 1 + 2 * int(rng.integers(0, 5))  # defocus blur, odd kernel 1..9
    if k > 1:
        img = cv2.GaussianBlur(img, (k, k), 0)
    img += rng.normal(0, rng.uniform(2, 8), (h, w, 3))  # sensor noise
    return np.clip(img, 0, 255).astype(np.uint8)


def augment_depth(dep, rng):
    """RealSense-D405-style stereo artifacts on clean rendered depth
    (float32 HxW, normalized 0..1 == 0..1.5 m, 0 = invalid):
    low-freq calibration warp, distance-dependent speckle, disparity
    quantization (step grows ~z^2), edge fattening / flying pixels,
    one-sided occlusion-shadow dropout along the stereo baseline, and
    blobby invalid patches (specular / low-texture). Invalid stays 0."""
    h, w = dep.shape
    valid = dep > 1e-6
    d = dep.astype(np.float32).copy()
    z = np.clip(d, 0.05, 1.0)
    # 1) low-frequency calibration/warp bias: amplitude U(0.001, 0.01)
    #    normalized = up to 15 mm peak-to-peak, the error budget of a
    #    factory-calibrated D405 (larger biases are not realistic)
    d += (smooth_noise(h, w, rng, rng.uniform(40, 150)) - 0.5) \
        * rng.uniform(0.001, 0.01)
    # 2) speckle noise, std growing ~quadratically with distance
    d += rng.standard_normal((h, w)).astype(np.float32) \
        * rng.uniform(0.002, 0.012) * (z / 0.3) ** 2
    # 3) disparity quantization
    step = rng.uniform(0.0008, 0.0035) * (z / 0.3) ** 2
    d = np.round(d / np.maximum(step, 1e-5)) * step
    # 4) edge artifacts where stereo matching fails
    gx = cv2.Sobel(dep, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(dep, cv2.CV_32F, 0, 1, ksize=3)
    edge = (np.hypot(gx, gy) > rng.uniform(0.02, 0.06)).astype(np.uint8)
    band = cv2.dilate(edge, np.ones((3, 3), np.uint8),
                      iterations=int(rng.integers(1, 3))) > 0
    dmin = cv2.erode(d, np.ones((5, 5), np.uint8))
    dmax = cv2.dilate(d, np.ones((5, 5), np.uint8))
    tmix = rng.random((h, w)).astype(np.float32)
    fly = band & (rng.random((h, w)) < rng.uniform(0.2, 0.6))
    d[fly] = (dmin * tmix + dmax * (1 - tmix))[fly]
    drop = band & (rng.random((h, w)) < rng.uniform(0.2, 0.7))
    d[drop] = 0
    # 5) one-sided occlusion shadow along the stereo baseline
    kw = int(rng.integers(2, 8))
    k = np.zeros((1, 2 * kw + 1), np.uint8)
    if rng.random() < 0.5:
        k[0, kw:] = 1
    else:
        k[0, :kw + 1] = 1
    shadow = cv2.dilate(edge, k) & (1 - edge)
    d[(shadow > 0) & (rng.random((h, w)) < rng.uniform(0.4, 0.9))] = 0
    # 6) blobby invalid patches
    for _ in range(int(rng.integers(0, 5))):
        cy, cx = int(rng.integers(0, h)), int(rng.integers(0, w))
        d[_irregular_blob((h, w), cy, cx, rng.uniform(4, 25), rng)] = 0
    d[~valid] = 0
    return np.clip(d, 0, 1)


def random_depth_fill(h, w, rng):
    """Normalized-depth field in [0,1] for depth leaks: smooth base with
    object-shaped patches at distinct depths (background clutter has
    depth discontinuities, not just gradients)."""
    base = rng.uniform(0.05, 1.0)
    amp = rng.uniform(0.05, 0.4)
    field = base + (smooth_noise(h, w, rng, rng.uniform(20, 80)) - 0.5) * 2 * amp
    canvas = (np.clip(field, 0, 1) * 255).astype(np.uint8)
    img3 = np.stack([canvas] * 3, -1)
    for _ in range(int(rng.integers(0, 7))):
        _draw_object(img3, rng)  # random gray levels = distinct depths
    out = img3[..., 0].astype(np.float32) / 255.0
    k = 1 + 2 * int(rng.integers(0, 3))
    if k > 1:
        out = cv2.GaussianBlur(out, (k, k), 0)
    return np.clip(out, 0, 1).astype(np.float32)
