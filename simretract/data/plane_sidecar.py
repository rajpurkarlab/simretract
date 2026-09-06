"""Per-frame plane-normal sidecar for a cache set.

For every frame of ``<cache>/<name>_dep.npy`` the ground-truth tool mask
(``<name>_msk.npy``) selects the depth pixels, and
:func:`simretract.serve.plane.fit_plane_sidecar` returns the camera-frame
unit normal of the tool plane (sign toward the camera) plus a planarity
confidence. The trainer's ``qpos_plane`` state mode reads the normal as the
three state components after the seven joint values, so this file must exist
next to the cache before training.

Outputs:
    ``<cache>/<name>_plane.npy``       (N, 3) float32 unit normals; (0, 0, 0)
                                       for frames with fewer than 200 valid
                                       mask pixels
    ``<cache>/<name>_plane_conf.npy``  (N,) float32 ``s2 / s3`` singular-value
                                       ratio (planarity; below ~3 is degenerate)

The loop is sequential and CPU-bound (roughly an hour per 100k frames).
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np

from simretract.serve.plane import fit_plane_sidecar

log = logging.getLogger(__name__)


def precompute_plane(cache: str | Path, name: str, log_every: int = 5000) -> tuple[int, int]:
    """Write the plane sidecar of set ``name`` in ``cache``.

    Returns ``(n_fit, n_frames)``: how many frames received a valid normal.
    """
    cache = Path(cache)
    dep = np.load(cache / f"{name}_dep.npy", mmap_mode="r")
    msk = np.load(cache / f"{name}_msk.npy", mmap_mode="r")
    n_frames = dep.shape[0]
    out = np.zeros((n_frames, 3), np.float32)
    conf = np.zeros(n_frames, np.float32)
    for i in range(n_frames):
        out[i], conf[i] = fit_plane_sidecar(dep[i], msk[i])
        if i % log_every == 0:
            log.info("[PLANE] %d/%d", i, n_frames)
    np.save(cache / f"{name}_plane.npy", out)
    np.save(cache / f"{name}_plane_conf.npy", conf)
    ok = np.linalg.norm(out, axis=1) > 0.5
    n_fit = int(ok.sum())
    if n_fit:
        log.info(
            "[PLANE] done: %d/%d frames with a fit; conf med %.1f (degenerate <3: %.1f%%)",
            n_fit, n_frames, float(np.median(conf[ok])), 100 * float((conf[ok] < 3).mean()),
        )
    else:
        log.warning("[PLANE] done: no frame of %s received a plane fit", name)
    return n_fit, n_frames


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute the per-frame plane-normal sidecar of a cache set.")
    p.add_argument("--cache", default=os.environ.get("SIMRETRACT_CACHE", os.environ.get("CACHE")),
                   required="SIMRETRACT_CACHE" not in os.environ and "CACHE" not in os.environ,
                   help="cache directory holding <name>_dep.npy / <name>_msk.npy (env SIMRETRACT_CACHE or CACHE)")
    p.add_argument("--name", default=os.environ.get("SIMRETRACT_SET", os.environ.get("NAME", "strict")),
                   help="set name / file prefix (env SIMRETRACT_SET or NAME; default strict)")
    p.add_argument("--log-every", type=int, default=5000, help="progress line every N frames")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    a = build_parser().parse_args(argv)
    precompute_plane(a.cache, a.name, a.log_every)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
