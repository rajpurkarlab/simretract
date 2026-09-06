"""Review-grid video for a raw episode directory (no simulator imports).

2x2 grid at half speed: RGB | turbo-coloured depth on the top row, tool
mask | gripper mask on the bottom row. Depth uses a FIXED range so colours
are stable across frames and episodes. Encoded with the ffmpeg binary that
``imageio-ffmpeg`` ships.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess

import numpy as np
from PIL import Image

DEPTH_RANGE_M = (0.05, 1.5)  # fixed normalisation for temporal stability

_LUT = None


def _turbo_lut() -> np.ndarray:
    from matplotlib import cm
    return (cm.get_cmap("turbo")(np.linspace(0, 1, 256))[:, :3]
            * 255).astype(np.uint8)


def depth_to_rainbow(depth_mm: np.ndarray) -> np.ndarray:
    """uint16 mm depth -> turbo RGB; invalid (0) and beyond-range far = black."""
    global _LUT
    if _LUT is None:
        _LUT = _turbo_lut()
    d = depth_mm.astype(np.float32) / 1000.0
    lo, hi = DEPTH_RANGE_M
    norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    idx = (norm * 255).astype(np.uint8)
    out = _LUT[idx]
    out[depth_mm == 0] = 0
    return out


def make_review_grid(ep_dir: str | os.PathLike, out_mp4: str | os.PathLike, fps: int = 12) -> None:
    """Render the grid frames of ``ep_dir`` and encode them into ``out_mp4``."""
    import imageio_ffmpeg

    ep_dir = str(ep_dir)
    frames = sorted(f for f in os.listdir(ep_dir) if f.startswith("rgb_"))
    grid_dir = os.path.join(ep_dir, "grid")
    os.makedirs(grid_dir, exist_ok=True)
    for i, f in enumerate(frames):
        t = f[4:-4]
        rgb = np.asarray(Image.open(os.path.join(ep_dir, f)))
        dep = np.asarray(Image.open(os.path.join(ep_dir, f"depth_{t}.png")))
        dv = depth_to_rainbow(dep)
        msk = np.asarray(Image.open(os.path.join(ep_dir, f"mask_{t}.png")))
        gm = np.asarray(Image.open(os.path.join(ep_dir, f"gmask_{t}.png")))
        top = np.hstack([rgb, dv])
        bot = np.hstack([np.stack([msk] * 3, -1), np.stack([gm] * 3, -1)])
        Image.fromarray(np.vstack([top, bot])).save(
            os.path.join(grid_dir, f"g_{i:04d}.png"))
    os.makedirs(os.path.dirname(os.path.abspath(str(out_mp4))), exist_ok=True)
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-framerate", str(fps),
         "-i", os.path.join(grid_dir, "g_%04d.png"), "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "21", str(out_mp4)],
        check=True, capture_output=True)
    shutil.rmtree(grid_dir, ignore_errors=True)  # frames are waste post-mp4


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render a 2x2 review mp4 from a raw episode dir.")
    ap.add_argument("ep_dir")
    ap.add_argument("out_mp4")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args(argv)
    make_review_grid(args.ep_dir, args.out_mp4, fps=args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
