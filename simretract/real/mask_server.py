"""SAM 3 streaming-tracker tool-mask server for real deployment.

The tool is detected once from a text prompt and then tracked frame to
frame with SAM 3's video predictor: the temporal memory keeps the mask
stable where per-image inference flickers. The selected identity remains
locked until an explicit episode reset, even while its visible mask is empty. Clients must stream
frames continuously (a few fps), not only at policy queries, so that the
memory stays fresh.

Single-object lock: the first sighting locks the largest instance that does
not lie on the gripper; afterwards only that object is followed (by tracker
id when available, else by mask overlap). Other instances are never unioned
in, and a lost object yields an empty mask rather than a switch to another
object, matching the simulation where an out-of-view tool is an empty mask.

Wire protocol (:mod:`simretract.serve.protocol`); arrays travel as bytes so
that clients on a different numpy major version can unpickle replies::

    {"rgb_bytes": bytes, "shape": (H, W, 3), "gmask_bytes": bytes?}  -> mask bytes (H*W u8 0/255)
    {"op": "ping"}                                                    -> {"ok", "prompt", "mode"}
    {"op": "reset"}                                                   -> {"ok"}  (re-detect next frame)

Runs in the mask environment (ultralytics); ``--ckpt`` is the SAM 3 weight file.
"""
from __future__ import annotations

import argparse
import logging
import os
import socket
import tempfile
import time
from typing import Any

import cv2
import numpy as np

from simretract.serve.protocol import recv_msg, send_msg

log = logging.getLogger("simretract.real.mask_server")

PROTOCOL_VERSION = 2
INITIAL_FRAME_CAPACITY = 256


def env_default(cast, default: Any, *names: str) -> Any:
    """Default for a flag: the first set environment variable among ``names``, else ``default``."""
    for n in names:
        v = os.environ.get(n)
        if v is not None and v != "":
            return cast(v)
    return default


class Tracker:
    """SAM 3 video predictor driven manually frame by frame.

    A two-frame seed video gives ``setup_source`` the correct model sizing;
    the dataset frame counter is then advanced by hand and frames are fed
    directly through preprocess / inference / postprocess.
    """

    def __init__(self, ckpt: str, prompt: str, conf: float):
        self.ckpt, self.prompt, self.conf = ckpt, prompt, conf
        self.pred = None
        self.first = True
        self.lock_id, self.lock_mask, self.lost = None, None, 0

    def _load(self, first_rgb):
        """Load model and configure its source once for this server process."""
        from ultralytics.models.sam import SAM3VideoSemanticPredictor
        p = SAM3VideoSemanticPredictor(overrides=dict(
            model=self.ckpt, mode="predict", save=False, verbose=False,
            conf=self.conf))
        p.setup_model(model=None)
        p.model.mask_threshold = 0.0
        # Ultralytics sets model sizes through setup_source. The loader is
        # never iterated afterward: every frame comes from the socket.
        with tempfile.TemporaryDirectory(prefix="sam3_stream_") as directory:
            seed = os.path.join(directory, "seed.mp4")
            vw = cv2.VideoWriter(seed, cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                 (first_rgb.shape[1], first_rgb.shape[0]))
            if not vw.isOpened():
                raise RuntimeError("Could not create the SAM sizing video")
            try:
                for _ in range(2):
                    vw.write(first_rgb[..., ::-1])
            finally:
                vw.release()
            p.setup_source(seed)
            cap = getattr(p.dataset, "cap", None)
            if cap is not None:
                cap.release()
        self.pred = p
        self.reset()

    def reset(self):
        """Forget an episode, retaining weights, text embeddings and kernels.

        In Ultralytics 8.4.64, all semantic object IDs, masks and temporal
        memories live under inference_state. Image caches on both predictor
        instances must also be cleared so the next real frame is fresh.
        """
        if self.pred is not None:
            p = self.pred
            for predictor in (p, p.tracker):
                predictor.inference_state = {}
                predictor.reset_image()
                predictor.prompts = {}
                predictor.segment_all = False
                predictor.batch = None
                predictor.results = None
                predictor.backbone_out = None
            p.dataset.frames = INITIAL_FRAME_CAPACITY
            p.dataset.frame = 1
            type(p).init_state(p)
        self.first = True
        self.lock_id, self.lock_mask, self.lost = None, None, 0

    def _ensure_frame_capacity(self):
        """Grow bookkeeping with observed frames instead of allocating 8 GB.

        Keep the horizon ahead of the current forward-only frame. Nested
        tracker states use the same horizon for temporal pointer selection.
        """
        p = self.pred
        state = p.inference_state
        frame = p.dataset.frame - 1
        prompts = state["per_frame_geometric_prompt"]
        if frame < len(prompts):
            return
        capacity = max(INITIAL_FRAME_CAPACITY, 2 * len(prompts), frame + 1)
        prompts.extend([None] * (capacity - len(prompts)))
        state["num_frames"] = capacity
        p.dataset.frames = capacity
        for tracker_state in state["tracker_inference_states"]:
            tracker_state["num_frames"] = capacity

    def warmup(self):
        """Exercise detector and one-object temporal tracking before serving."""
        import torch
        rgb = np.zeros((480, 640, 3), np.uint8)
        if self.pred is None:
            self._load(rgb)
        self.mask(rgb)
        p = self.pred
        with torch.inference_mode():
            # A blank detector image usually has no objects. Seed a private
            # synthetic mask to also warm the interactive memory path; it
            # never enters semantic metadata or a real episode's ID lock.
            h, w = p.tracker._bb_feat_sizes[0]
            seed = torch.zeros((1, h, w), device=p.device)
            seed[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 1
            states = p._tracker_add_new_objects(
                frame_idx=0, num_frames=INITIAL_FRAME_CAPACITY,
                new_obj_ids=[0], new_obj_masks=seed, tracker_states_local=[])
            p.tracker.propagate_in_video(states[0], frame_idx=1)
            del states, seed
        if p.device.type == "cuda":
            torch.cuda.synchronize(p.device)
        self.reset()

    def candidates(self, r: Any, h: int, w: int, gmask: np.ndarray | None) -> list[tuple[Any, np.ndarray]]:
        """Instance masks of a result as (tracker id or None, bool mask), gripper instances dropped."""
        cands: list[tuple[Any, np.ndarray]] = []
        if r.masks is None or not len(r.masks):
            return cands
        ids = None
        try:
            if r.boxes is not None and r.boxes.id is not None:
                ids = r.boxes.id.cpu().numpy().astype(int).tolist()
        except Exception:
            ids = None
        for i, m in enumerate(r.masks.data.cpu().numpy()):
            mm = m > 0.5
            if mm.shape != (h, w):
                mm = cv2.resize(mm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            if not mm.any():
                continue
            if gmask is not None:
                # the serrated fingers match "metal rake" too: drop instances
                # that mostly lie on the known gripper silhouette
                if (mm & (gmask > 0)).sum() / mm.sum() > 0.5:
                    continue
            cands.append((ids[i] if ids is not None and i < len(ids) else None, mm))
        return cands

    def select(self, cands: list[tuple[Any, np.ndarray]]) -> np.ndarray | None:
        """Apply the single-object lock to this frame's candidates; returns the picked mask."""
        pick = None
        if self.lock_id is None and self.lock_mask is None:
            if cands:  # first sighting: lock the largest
                pick = max(cands, key=lambda c: c[1].sum())
                self.lock_id, self.lock_mask = pick[0], pick[1]
                log.info("locked object id=%s px=%d", pick[0], int(pick[1].sum()))
        else:
            if self.lock_id is not None:
                same = [c for c in cands if c[0] == self.lock_id]
                pick = same[0] if same else None
            else:  # no ids: overlap continuity
                best, biou = None, 0.0
                for c in cands:
                    inter = (c[1] & self.lock_mask).sum()
                    union = (c[1] | self.lock_mask).sum()
                    iou = inter / max(union, 1)
                    if iou > biou:
                        best, biou = c, iou
                pick = best if biou >= 0.1 else None
            if pick is not None:
                self.lock_mask = pick[1]
                self.lost = 0
            else:
                self.lost += 1
                if self.lost in (1, 25, 100):
                    log.info("locked object not visible (%d frames); empty mask", self.lost)
        return None if pick is None else pick[1]

    def mask(self, rgb: np.ndarray, gmask: np.ndarray | None = None) -> np.ndarray:
        """Tool mask (u8 0/255) for one RGB frame."""
        import torch
        if self.pred is None:
            self._load(rgb)
        p = self.pred
        self._ensure_frame_capacity()
        if self.lock_id is not None:
            # Retain the selected video memory through detector misses.
            # Inference still produces a fresh, possibly empty mask.
            metadata = p.inference_state.get("tracker_metadata", {}).get("metadata", {})
            keep_alive = metadata.get("trk_keep_alive", {})
            if self.lock_id in keep_alive:
                keep_alive[self.lock_id] = p.max_trk_keep_alive
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        p.batch = (["live.jpg"], [bgr], "")
        im = p.preprocess([bgr])
        # keep re-prompting with the text until an object is actually locked;
        # a reset with the tool out of view would otherwise leave the tracker
        # promptless for the rest of the episode
        need_prompt = self.first or (self.lock_id is None and self.lock_mask is None)
        with torch.inference_mode():
            preds = p.inference(im, text=[self.prompt] if need_prompt else None)
        self.first = False
        r = p.postprocess(preds, im, [bgr])[0]
        p.dataset.frame += 1
        h, w = rgb.shape[:2]
        out = np.zeros((h, w), np.uint8)
        picked = self.select(self.candidates(r, h, w, gmask))
        if picked is not None:
            out[picked] = 255
        return out


def serve(trk: Tracker, host: str, port: int) -> None:
    """Accept one client at a time; tracker state survives disconnects."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    log.info("serving on %s:%d", host, port)
    n_f = 0
    while True:
        conn, addr = srv.accept()
        log.info("client %s", addr)
        try:
            while True:
                msg = recv_msg(conn)
                if msg is None:
                    break
                if isinstance(msg, dict) and msg.get("op") == "ping":
                    send_msg(conn, {"ok": True, "prompt": trk.prompt, "conf": trk.conf,
                                    "mode": "video-track", "protocol_version": PROTOCOL_VERSION,
                                    "warm_reset": True, "locked_track_keepalive": True,
                                    "ckpt": os.path.realpath(trk.ckpt)})
                    continue
                if isinstance(msg, dict) and msg.get("op") == "reset":
                    trk.reset()
                    n_f = 0
                    send_msg(conn, {"ok": True})
                    log.info("tracker reset")
                    continue
                t0 = time.time()
                rgb = np.frombuffer(msg["rgb_bytes"], np.uint8).reshape(msg["shape"])
                gm = None
                if "gmask_bytes" in msg:
                    gm = np.frombuffer(msg["gmask_bytes"], np.uint8).reshape(msg["shape"][:2])
                m = trk.mask(rgb, gm)
                send_msg(conn, m.tobytes())
                n_f += 1
                if n_f % 25 == 1:
                    log.info("f%d: %.2fs px=%d", n_f, time.time() - t0, int((m > 0).sum()))
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            log.warning("client connection error: %r; back to accept", e)
        finally:
            conn.close()
            log.info("client disconnected (tracker state kept; send op=reset to re-detect)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m simretract.real.mask_server",
        description="SAM 3 video-tracking tool-mask server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--ckpt", default=env_default(str, None, "SIMRETRACT_SAM3_CKPT", "SAM3_CKPT"),
                    required=env_default(str, None, "SIMRETRACT_SAM3_CKPT", "SAM3_CKPT") is None,
                    help="SAM 3 weight file (env SAM3_CKPT)")
    ap.add_argument("--prompt", default=env_default(str, "metal rake", "SIMRETRACT_MASK_PROMPT", "PROMPT"),
                    help="text prompt used while acquiring the lock")
    ap.add_argument("--conf", type=float, default=env_default(float, 0.2, "SIMRETRACT_MASK_CONF", "CONF"),
                    help="predictor confidence setting (preserves SAM tracking thresholds)")
    ap.add_argument("--host", default=env_default(str, "127.0.0.1", "SIMRETRACT_HOST", "HOST"))
    ap.add_argument("--port", type=int, default=env_default(int, 8766, "SIMRETRACT_MASK_PORT", "PORT"))
    return ap


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    a = build_parser().parse_args(argv)
    if not os.path.isfile(a.ckpt):
        raise SystemExit(f"SAM 3 checkpoint not found: {a.ckpt} (see checkpoints/README.md)")
    trk = Tracker(a.ckpt, a.prompt, a.conf)
    log.info("warmup (prompt=%r) ...", a.prompt)
    t0 = time.time()
    trk.warmup()
    log.info("warmup %.2fs", time.time() - t0)
    serve(trk, a.host, a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
