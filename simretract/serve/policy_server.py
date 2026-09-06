"""Policy server: loads one fine-tuned checkpoint and serves 50-step joint
target chunks over TCP to the simulation evaluator and the real-robot client.

The server performs every observation-side step the trainer's validation
path performs, so a checkpoint sees exactly the inputs it was trained on:

* masked RGB (tool | gripper visible, rest black) letterboxed to 224,
* fixed-scale depth (uint16 mm / 1000, clipped at 1.5 m) as the second
  camera stream (float or uint8 round-trip, per ``norm.json``),
* the camera-frame plane normal fitted from THIS query's depth + mask
  (``simretract.serve.plane.fit_plane_served``) appended to the 7-D joint
  state when the checkpoint expects a 10-D state,
* z-scoring of the state and un-normalization of the action chunk with the
  ``norm.json`` statistics saved by the trainer.

Policy classes are chosen from the checkpoint's ``config.json`` ``type``:
pi0 (default), pi05, pi0_fast, smolvla, act and fastwam.  Shorter native
horizons (fastwam: 48) are padded to the 50-step wire chunk by repeating the
last action; the client executes only the native horizon.

Wire protocol (see ``simretract.serve.protocol``):

* request ``{"rgb": (480,640,3) u8, "depth_mm": (480,640) u16,
  "mask": (480,640) u8 0/255, "gmask": (480,640) u8 0/255,
  "state": (14,) f32 = qpos(7) + grasp_pose(7), optional "want_plane",
  optional "plane_top_cut"}`` or ``{"op": "ping"}``;
* reply: the (50, 7) float32 chunk as nested lists, or ``{"chunk", "plane_n",
  "plane_c"}`` when ``want_plane`` is set, or ``{"error": str}``.

Run in the policy environment:
``python -m simretract.serve.policy_server --ckpt <dir> [--port 8765]``.
``--stub`` serves a hold-pose chunk without loading any weights so the
closed-loop machinery can be exercised without a GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from simretract.serve.protocol import ensure_wire_safe, recv_msg, send_msg

log = logging.getLogger("simretract.serve")

#: Fixed language prompt; part of the model contract.
TASK = "take over the tool from the operator and hold its pose"
#: Wire chunk length (actions per query).
H = 50
#: Letterbox resolution of both image streams.
RES = 224
#: Checkpoint types that manage their own precision (no bf16 cast, no autocast).
_OWN_PRECISION = ("fastwam", "pi0_fast", "act")
#: Checkpoint types loaded plainly (no bf16 default-dtype construction).
_PLAIN_LOAD = ("pi0_fast", "pi05", "act", "smolvla")
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------- config
def _env(name: str, default: str) -> str:
    """Environment fallback for a flag: SIMRETRACT_<NAME> wins over <NAME>."""
    return os.environ.get(f"SIMRETRACT_{name}", os.environ.get(name, default))


def _env_flag(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0") == "1"


@dataclass
class ServeConfig:
    """Everything that shapes what the server computes (the "regime")."""

    ckpt: str = ""
    host: str = "127.0.0.1"
    port: int = 8765
    device: str = "cuda"
    half: bool = True
    lean_load: bool = True
    fastwam_cpu_load: bool = True
    eval_aug: bool = False
    nogrip: bool | None = None
    input_mode: str | None = None
    drop_px: int = 300
    plane_filter: bool = True
    plane_top_cut: float = 0.0
    plane_top_cut_max_deg: float = 20.0
    stub: bool = False


def base_weight_id(key: str) -> str:
    """Hub id of a third-party tokenizer/weight from ``configs/hub.yaml``
    (``base_weights.<key>``); the only place hub ids enter the server."""
    from simretract import config

    try:
        return str(config.hub()["base_weights"][key])
    except KeyError as e:
        raise KeyError(f"configs/hub.yaml: base_weights.{key} is missing") from e


def resolve_checkpoint_dir(path: str | os.PathLike[str]) -> Path:
    """Accept a checkpoint directory in either layout: the files directly
    inside ``path`` or inside ``path/final`` (the trainer's output layout)."""
    p = Path(path)
    if (p / "config.json").is_file():
        return p
    if (p / "final" / "config.json").is_file():
        return p / "final"
    raise FileNotFoundError(
        f"{p}: no config.json in the directory or in its final/ subdirectory")


def checkpoint_type(ckpt_dir: str | os.PathLike[str]) -> str:
    """Policy type recorded in ``config.json`` (``pi0`` when absent)."""
    try:
        with open(Path(ckpt_dir) / "config.json") as f:
            return str(json.load(f).get("type", "pi0"))
    except (OSError, ValueError):
        return "pi0"


@dataclass
class NormSpec:
    """Contents of the trainer's ``norm.json`` that the server honours."""

    state_mean: np.ndarray | None = None
    state_std: np.ndarray | None = None
    action_mean: np.ndarray | None = None
    action_std: np.ndarray | None = None
    float_depth: bool = False
    nogrip: bool = False
    input_mode: str = ""
    drop_aug: dict | None = None

    @property
    def has_stats(self) -> bool:
        return self.state_mean is not None

    @classmethod
    def load(cls, ckpt_dir: Path) -> "NormSpec":
        p = ckpt_dir / "norm.json"
        if not p.is_file():
            return cls()
        with open(p) as f:
            n = json.load(f)
        drop = n.get("drop_aug") or {}
        return cls(
            state_mean=np.asarray(n["state_mean"], np.float32),
            state_std=np.asarray(n["state_std"], np.float32),
            action_mean=np.asarray(n["action_mean"], np.float32),
            action_std=np.asarray(n["action_std"], np.float32),
            float_depth=bool(n.get("float_depth", False)),
            nogrip=bool(n.get("nogrip", False)),
            input_mode=str(n.get("input_mode", "masked_rgbd")),
            # dropout-trained checkpoints couple "mask lost" with "no plane"
            # at serve time; every other checkpoint keeps the plane always on
            drop_aug=drop if float(drop.get("mask_p", 0)) > 0 else None,
        )


# ------------------------------------------------------------ preprocess
def letterbox(img: np.ndarray, res: int = RES) -> np.ndarray:
    """Aspect-preserving resize onto a black ``res`` x ``res`` canvas
    (INTER_AREA), identical to the trainer's letterbox."""
    import cv2

    h, w = img.shape[:2]
    s = res / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    out = np.zeros((res, res, 3), img.dtype)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out[(res - nh) // 2:(res - nh) // 2 + nh,
        (res - nw) // 2:(res - nw) // 2 + nw] = r
    return out


def depth01(depth_mm: np.ndarray) -> np.ndarray:
    """uint16 millimetres -> metres / 1.5, clipped to [0, 1]."""
    dep = np.asarray(depth_mm).astype(np.float32) / 1000.0
    return np.clip(dep / 1.5, 0, 1)


def apply_eval_aug(msg: dict, rng: np.random.Generator, nogrip: bool) -> dict:
    """Training-grade perception corruption applied once per query (the
    trainer's train-branch mask/RGB/depth augmentation).  The corrupted
    arrays feed both the plane fit and the policy, mirroring how real
    segmentation and depth errors propagate."""
    from simretract.data.seg_aug import (augment_depth, corrupt_mask,
                                         random_depth_fill, random_fill)

    rgb = np.ascontiguousarray(msg["rgb"])[..., :3]
    msk = np.asarray(msg["mask"])
    gmk = np.asarray(msg["gmask"])
    if nogrip:
        gmk = np.zeros_like(gmk)
    m_in = corrupt_mask(msk, rng)
    tool_b, grip_b, seg_b = msk > 0, gmk > 0, m_in > 0
    visible = (seg_b & tool_b) | grip_b
    leak = seg_b & ~tool_b & ~grip_b
    out = np.zeros_like(rgb)
    out[visible] = rgb[visible]
    if leak.any():
        out[leak] = random_fill(rgb.shape[0], rgb.shape[1], rng)[leak]
    dep = depth01(msg["depth_mm"])
    dep = augment_depth(dep, rng)
    dep_vis = dep * visible
    if leak.any():
        dep_vis[leak] = random_depth_fill(*dep_vis.shape, rng)[leak]
    out_msg = dict(msg)
    out_msg["_aug_rgb_masked"] = out
    out_msg["_aug_dep01"] = dep_vis
    out_msg["_aug_m_in"] = m_in            # the segmentation-like corrupted mask
    out_msg["_aug_dep_mm"] = (dep * 1500.0).astype(np.uint16)
    return out_msg


@dataclass
class Preprocessor:
    """Builds the policy batch from a query message (the trainer's
    validation path for the checkpoint's ``input_mode``)."""

    device: str
    state_dim: int
    input_mode: str
    float_depth: bool
    nogrip: bool
    lang_tokens: Any = None
    lang_masks: Any = None
    context: dict | None = None

    def images(self, msg: dict) -> tuple[np.ndarray, np.ndarray | None]:
        """Return (wrist image, second stream or None) as float32 HxWx3 in
        [0, 1], letterboxed."""
        if "_aug_rgb_masked" in msg:      # pre-corrupted arrays (eval aug)
            out = msg["_aug_rgb_masked"]
            dep = msg["_aug_dep01"]
        else:
            rgb = np.ascontiguousarray(msg["rgb"])[..., :3]
            msk = np.asarray(msg["mask"])
            gmk = np.asarray(msg["gmask"])
            if self.nogrip:               # tool-only input
                gmk = np.zeros_like(gmk)
            dep = depth01(msg["depth_mm"])
            if self.input_mode == "full_rgbd":   # no-mask ablation: raw streams
                out = rgb
            else:
                visible = (msk > 0) | (gmk > 0)
                out = np.zeros_like(rgb)
                out[visible] = rgb[visible]
                dep = dep * visible
        if self.input_mode == "masked_rgb_nodepth":
            # no-depth ablation: single camera; the plane normal is still
            # fitted from depth + mask into the state
            m3 = None
        elif self.input_mode == "masked_rgb":
            # binary visible mask as the second camera
            gm = np.asarray(msg["gmask"])
            if self.nogrip:
                gm = np.zeros_like(gm)
            cam2 = ((np.asarray(msg["mask"]) > 0) | (gm > 0)).astype(np.uint8)
            m3 = letterbox(np.repeat((cam2 * 255)[:, :, None], 3, 2)) \
                .astype(np.float32) / 255.0
        elif self.float_depth:
            m3 = letterbox(np.repeat(dep.astype(np.float32)[:, :, None], 3, 2))
        else:
            d3 = np.repeat((dep * 255).astype(np.uint8)[:, :, None], 3, 2)
            m3 = letterbox(d3).astype(np.float32) / 255.0
        img = letterbox(out).astype(np.float32) / 255.0
        return img, m3

    def build_batch(self, msg: dict) -> dict:
        import torch

        img, m3 = self.images(msg)
        state = np.asarray(msg["state"],
                           dtype=np.float32).ravel()[:self.state_dim].reshape(1, -1)
        batch = {
            "observation.images.wrist": torch.from_numpy(img).permute(2, 0, 1)
            .unsqueeze(0).to(self.device),
            "observation.state": torch.from_numpy(state).to(self.device),
        }
        if m3 is not None:
            batch["observation.images.mask"] = torch.from_numpy(m3).permute(
                2, 0, 1).unsqueeze(0).to(self.device)
        # fastwam treats the prompt and a precomputed context as mutually
        # exclusive: send the task string only when no context is loaded
        if self.context is None:
            batch["task"] = [TASK]
        if self.lang_tokens is not None:
            batch["observation.language.tokens"] = self.lang_tokens
            batch["observation.language.attention_mask"] = self.lang_masks
        if self.context:
            batch.update(self.context)
        return batch


# ----------------------------------------------------------------- engines
class StubEngine:
    """Serves a hold-pose chunk (the query's own joint state repeated H
    times) so the closed-loop evaluator can run without weights."""

    policy_type = "stub"
    state_dim = 7
    horizon = H

    def __init__(self, cfg: ServeConfig) -> None:
        self.cfg = cfg
        self.last_plane: dict[str, list | None] = {"n": None, "c": None}

    def infer(self, msg: dict) -> np.ndarray:
        q = np.asarray(msg["state"], np.float32).ravel()[:7]
        if q.shape[0] < 7:
            q = np.pad(q, (0, 7 - q.shape[0]))
        return np.repeat(q[None, :], H, 0).astype(np.float32)

    def regime(self) -> dict:
        return _regime(self.cfg, self.policy_type, self.horizon, nogrip=False,
                       input_mode="stub", state_dim=self.state_dim)


class PolicyEngine:
    """Loaded checkpoint + preprocessing + plane conditioning + z-scoring."""

    def __init__(self, cfg: ServeConfig) -> None:
        import torch

        self.cfg = cfg
        self.ckpt = resolve_checkpoint_dir(cfg.ckpt)
        self.device = cfg.device
        self.policy_type = checkpoint_type(self.ckpt)
        self.norm = NormSpec.load(self.ckpt)
        self.nogrip = self.norm.nogrip if cfg.nogrip is None else cfg.nogrip
        self.input_mode = cfg.input_mode or self.norm.input_mode or "masked_rgbd"
        self.drop_aug = self.norm.drop_aug
        self.last_plane: dict[str, list | None] = {"n": None, "c": None}
        self._aug_rng = np.random.default_rng() if cfg.eval_aug else None
        log.info("loading %s (%s) on %s", self.ckpt, self.policy_type, self.device)
        if self.norm.has_stats:
            log.info("norm.json: float_depth=%s nogrip=%s input_mode=%s",
                     self.norm.float_depth, self.nogrip, self.input_mode)
            if self.drop_aug is not None:
                log.info("dropout checkpoint: plane zeroed when mask < %d px "
                         "(train drop_aug %s)", cfg.drop_px, self.drop_aug)
        else:
            log.info("no norm.json: serving raw units")
        if cfg.eval_aug:
            log.info("eval aug on: training-grade corruption per query")

        self.policy = self._load_policy()
        pcfg = self.policy.config
        self.horizon = getattr(pcfg, "chunk_size", None) or getattr(pcfg, "action_horizon", None)
        if self.policy_type == "fastwam":
            # shorter native horizon: padded to H on the wire; the client
            # executes only the native steps
            assert pcfg.n_action_steps == self.horizon and self.horizon <= H, \
                f"{self.policy_type} chunking {self.horizon}/{pcfg.n_action_steps} vs H={H}"
            log.info("%s horizon %d (padded to %d; run the client with exec-h %d)",
                     self.policy_type, self.horizon, H, self.horizon)
        else:
            assert pcfg.n_action_steps == H and pcfg.chunk_size == H, \
                f"checkpoint chunking {pcfg.chunk_size}/{pcfg.n_action_steps} != {H}"
        try:   # state width from the checkpoint (10 = qpos+plane, 7 = qpos)
            self.state_dim = int(pcfg.input_features["observation.state"].shape[0])
        except (AttributeError, KeyError, TypeError):
            self.state_dim = 14
        log.info("state dim %d", self.state_dim)

        lang_tokens = lang_masks = None
        context = None
        tokenizer_name = self._tokenizer_name()
        if tokenizer_name:
            # fixed task language pre-tokenized exactly like the trainer
            # (TASK + newline, padded to tokenizer_max_length)
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(tokenizer_name)
            enc = tok(TASK + "\n", padding="max_length", truncation=True,
                      max_length=pcfg.tokenizer_max_length, return_tensors="pt")
            lang_tokens = enc["input_ids"].to(self.device)
            lang_masks = enc["attention_mask"].to(self.device, dtype=torch.bool)
        elif self.policy_type == "fastwam" and (self.ckpt / "context.pt").is_file():
            # precomputed UMT5 context for the fixed prompt (trainer export)
            c = torch.load(self.ckpt / "context.pt", map_location="cpu")
            context = {"context": c["context"].to(self.device, dtype=torch.bfloat16)}
            if "context_mask" in c:
                context["context_mask"] = c["context_mask"].to(self.device)
            log.info("fastwam context.pt loaded: %s for prompt %r",
                     tuple(context["context"].shape), c.get("prompt", "?"))
        self.pre = Preprocessor(self.device, self.state_dim, self.input_mode,
                                self.norm.float_depth, self.nogrip,
                                lang_tokens, lang_masks, context)
        self.prompt_tok = self._prompt_state_tokenizer() \
            if self.policy_type in ("pi0_fast", "pi05") else None

        t0 = time.time()
        out = self.infer(_zero_message(self.state_dim))
        log.info("warmup ok: chunk %s in %.2fs", out.shape, time.time() - t0)

    # -- loading ---------------------------------------------------------
    def _tokenizer_name(self) -> str | None:
        if self.policy_type == "smolvla":
            return base_weight_id("smolvlm_tokenizer")
        if self.policy_type == "pi0":
            return base_weight_id("paligemma_tokenizer")
        return None

    def _load_policy(self):
        import torch

        cfg = self.cfg
        t = self.policy_type
        if t == "smolvla":
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as cls
        elif t == "fastwam":
            from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy as cls
        elif t == "pi0_fast":
            from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy as cls
        elif t == "act":
            from lerobot.policies.act.modeling_act import ACTPolicy as cls
        elif t == "pi05":
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy as cls
        elif t == "pi0":
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy as cls
        else:
            raise ValueError(f"unsupported checkpoint type {t!r} "
                             "(expected pi0, pi05, pi0_fast, smolvla, act or fastwam)")

        ckpt = str(self.ckpt)
        if t == "fastwam":
            from lerobot.configs.policies import PreTrainedConfig

            pc = PreTrainedConfig.from_pretrained(ckpt)
            pc.load_text_encoder = not (self.ckpt / "context.pt").is_file()
            # stage the state dict on the CPU so the GPU never holds the
            # model plus a full state-dict copy at once
            pc.device = "cpu" if cfg.fastwam_cpu_load else self.device
            policy = cls.from_pretrained(ckpt, config=pc)
            pc.device = self.device
            log.info("fastwam load (%s; text encoder %s)",
                     "cpu -> " + self.device if cfg.fastwam_cpu_load else self.device,
                     "loaded" if pc.load_text_encoder else "skipped: context.pt")
        elif t in _PLAIN_LOAD:
            policy = cls.from_pretrained(ckpt)
            log.info("%s plain load", t)
        elif cfg.lean_load:
            # construct with bf16 as the default dtype so the CPU-side
            # instantiation stays small; the end state equals a plain load
            # followed by the bf16 cast below
            assert cfg.half, "--lean-load needs --half"
            torch.set_default_dtype(torch.bfloat16)
            policy = cls.from_pretrained(ckpt)
            torch.set_default_dtype(torch.float32)
            log.info("lean bf16 load")
        else:
            policy = cls.from_pretrained(ckpt)
        if cfg.half and t not in _OWN_PRECISION:
            # cast on the CPU before the device move so the GPU never holds
            # the fp32 copy
            policy.to(torch.bfloat16)
            log.info("weights cast to bf16")
        policy.to(self.device).eval()
        if t == "fastwam" and cfg.fastwam_cpu_load:
            # the fastwam model caches its construction device and moves every
            # input to it; repoint it after the cuda move
            policy.model.device = torch.device(self.device)
        torch.cuda.empty_cache()
        return policy

    def _prompt_state_tokenizer(self) -> Callable[[dict], dict]:
        """pi0_fast / pi05 embed the z-scored state in the prompt: padded to
        max_state_dim, discretized into 256 bins over [-1, 1], tokenized
        with the port-specific suffix."""
        from transformers import AutoTokenizer

        pcfg = self.policy.config
        name = getattr(pcfg, "text_tokenizer_name", None) or base_weight_id("paligemma_tokenizer")
        tok = AutoTokenizer.from_pretrained(name, padding_side="right")
        bins = np.linspace(-1, 1, 257)[:-1]
        clean = TASK.strip().replace("_", " ").replace("\n", " ")
        suffix = ";\n" if self.policy_type == "pi0_fast" else ";\nAction: "
        device = self.device

        def prompt_tok(batch: dict) -> dict:
            import torch

            st = batch["observation.state"].detach().float().cpu().numpy()
            st = np.pad(st, ((0, 0), (0, pcfg.max_state_dim - st.shape[1])))
            dig = np.digitize(st, bins=bins) - 1
            prompts = [f"Task: {clean}, State: {' '.join(map(str, d))}{suffix}"
                       for d in dig]
            enc = tok(prompts, padding="max_length", truncation=True,
                      max_length=pcfg.tokenizer_max_length, return_tensors="pt")
            batch["observation.language.tokens"] = enc["input_ids"].to(device)
            batch["observation.language.attention_mask"] = \
                enc["attention_mask"].to(device, dtype=torch.bool)
            return batch

        log.info("prompt-state builder ready (state pad %d, prompt max_len %d)",
                 pcfg.max_state_dim, pcfg.tokenizer_max_length)
        return prompt_tok

    # -- inference -------------------------------------------------------
    def plane_from_msg(self, msg: dict) -> np.ndarray:
        """Camera-frame plane normal of the tool cloud in this query (zeros
        when no plane can be fitted); also records the plane for the
        ``want_plane`` reply."""
        from simretract.serve.plane import fit_plane_served

        if "_aug_m_in" in msg:            # eval aug: fit from corrupted inputs
            dep = np.asarray(msg["_aug_dep_mm"])
            mask_in = np.asarray(msg["_aug_m_in"])
        else:
            dep = np.asarray(msg["depth_mm"])
            mask_in = np.asarray(msg["mask"])
        top_cut = float(msg.get("plane_top_cut", self.cfg.plane_top_cut))
        res = fit_plane_served(dep, mask_in, plane_filter=self.cfg.plane_filter,
                               top_cut=top_cut,
                               top_cut_max_deg=self.cfg.plane_top_cut_max_deg)
        if res is None:
            self.last_plane["n"] = None
            self.last_plane["c"] = None
            return np.zeros(3, np.float32)
        n, c = res
        self.last_plane["n"] = np.asarray(n).tolist()
        self.last_plane["c"] = np.asarray(c).tolist()
        return np.asarray(n, np.float32)

    def infer(self, msg: dict) -> np.ndarray:
        """Query -> (H, 7) float32 raw joint targets."""
        import torch

        if self._aug_rng is not None:
            msg = apply_eval_aug(msg, self._aug_rng, self.nogrip)
        if self.state_dim == 10:
            # qpos(7) + plane normal from THIS query's depth + mask
            msg = dict(msg)
            q = np.asarray(msg["state"], np.float32).ravel()[:7]
            pl = self.plane_from_msg(msg)
            if self.drop_aug is not None and \
                    int((np.asarray(msg["mask"]) > 0).sum()) < self.cfg.drop_px:
                pl = np.zeros(3, np.float32)   # lost mask -> blind frame, no plane
            msg["state"] = np.concatenate([q, pl])
        norm = self.norm
        if norm.has_stats:
            msg = dict(msg)
            st = np.asarray(msg["state"], np.float32).ravel()[:len(norm.state_mean)]
            msg["state"] = (st - norm.state_mean) / norm.state_std
        batch = self.pre.build_batch(msg)
        if self.prompt_tok is not None:
            batch = self.prompt_tok(batch)
        if self.policy_type == "act":   # ImageNet normalization as in the trainer
            mean = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
            std = torch.tensor(_IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
            for k in list(batch):
                if k.startswith("observation.images."):
                    batch[k] = (batch[k] - mean) / std
        if self.policy_type in _OWN_PRECISION:
            with torch.inference_mode():
                chunk = self.policy.predict_action_chunk(batch)
        else:
            with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
                chunk = self.policy.predict_action_chunk(batch)
        out = chunk[0, :H].float().cpu().numpy().astype(np.float32)
        if out.shape[0] < H:   # shorter native horizon -> pad to the wire chunk
            out = np.concatenate([out, np.repeat(out[-1:], H - out.shape[0], 0)])
        if norm.has_stats:
            out = out * norm.action_std + norm.action_mean
        return out

    def regime(self) -> dict:
        return _regime(self.cfg, self.policy_type, int(self.horizon),
                       nogrip=self.nogrip, input_mode=self.input_mode,
                       state_dim=self.state_dim, ckpt=str(self.ckpt))


def _zero_message(state_dim: int) -> dict:
    return {"rgb": np.zeros((480, 640, 3), np.uint8),
            "depth_mm": np.zeros((480, 640), np.uint16),
            "mask": np.zeros((480, 640), np.uint8),
            "gmask": np.zeros((480, 640), np.uint8),
            "state": np.zeros(state_dim, np.float32)}


def _regime(cfg: ServeConfig, policy_type: str, horizon: int, *, nogrip: bool,
            input_mode: str, state_dim: int, ckpt: str | None = None) -> dict:
    """The effective serving regime, returned with every ping so a driver
    can refuse a server that does not match the protocol it expects."""
    half = bool(cfg.half) and policy_type not in _OWN_PRECISION and policy_type != "stub"
    return {
        "ckpt": ckpt if ckpt is not None else cfg.ckpt,
        "type": policy_type,
        "half": half,
        "plane_filter": bool(cfg.plane_filter),
        "drop_px": int(cfg.drop_px),
        "plane_top_cut": float(cfg.plane_top_cut),
        "plane_top_cut_max_deg": float(cfg.plane_top_cut_max_deg),
        "nogrip": bool(nogrip),
        "chunk": H,
        "horizon": int(horizon),
        "exec_pad": H - int(horizon),
        "input_mode": input_mode,
        "state_dim": int(state_dim),
        "eval_aug": bool(cfg.eval_aug),
        "stub": policy_type == "stub",
    }


# ------------------------------------------------------------------- serve
def serve(engine: PolicyEngine | StubEngine, host: str, port: int) -> None:
    """Accept one client at a time and answer queries until interrupted."""
    regime = engine.regime()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    # the driver waits for this exact phrase in the log
    log.info("serving on %s:%d", host, port)
    while True:
        conn, addr = srv.accept()
        log.info("client %s", addr)
        try:
            n_q = 0
            while True:
                try:
                    msg = recv_msg(conn)
                except (ConnectionResetError, BrokenPipeError, OSError) as e:
                    log.warning("client connection lost: %r", e)
                    break
                if msg is None:
                    break
                if isinstance(msg, dict) and msg.get("op") == "ping":
                    reply = {"ok": True, "ckpt": regime["ckpt"],
                             "eval_aug": "1" if regime["eval_aug"] else "0",
                             "nogrip": bool(regime["nogrip"]),
                             "regime": regime}
                    ensure_wire_safe(reply)
                    send_msg(conn, reply)
                    continue
                try:
                    t0 = time.time()
                    chunk = engine.infer(msg)
                    # plain lists on the wire (numpy-version independent)
                    if isinstance(msg, dict) and msg.get("want_plane"):
                        reply = {"chunk": chunk.tolist(),
                                 "plane_n": engine.last_plane["n"],
                                 "plane_c": engine.last_plane["c"]}
                    else:
                        reply = chunk.tolist()
                    ensure_wire_safe(reply)
                    send_msg(conn, reply)
                    n_q += 1
                    log.info("query %d: %.2fs", n_q, time.time() - t0)
                except Exception as e:  # keep serving; the client decides
                    traceback.print_exc()
                    send_msg(conn, {"error": repr(e)})
        finally:
            conn.close()
            log.info("client disconnected")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="simretract.serve.policy_server",
        description="Serve 50-step action chunks from one checkpoint over TCP. "
                    "Each flag falls back to the environment variable named in "
                    "its help (SIMRETRACT_<NAME> or the bare name).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("ckpt_pos", nargs="?", metavar="CKPT", help=argparse.SUPPRESS)
    p.add_argument("--ckpt", default=_env("CKPT", ""),
                   help="checkpoint directory (config.json + model.safetensors + "
                        "norm.json, directly or under final/) [env CKPT]")
    p.add_argument("--host", default=_env("HOST", "127.0.0.1"), help="bind address [env HOST]")
    p.add_argument("--port", type=int, default=int(_env("PORT", "8765")),
                   help="TCP port (8765 real robot, 8767 sim eval) [env PORT]")
    p.add_argument("--device", default=_env("DEVICE", "cuda"), help="torch device [env DEVICE]")
    p.add_argument("--half", dest="half", action="store_true", default=_env_flag("HALF", True),
                   help="cast pi0/pi05/smolvla weights to bf16 (act, pi0_fast and "
                        "fastwam keep their own precision) [env HALF]")
    p.add_argument("--no-half", dest="half", action="store_false", help="serve fp32 weights")
    p.add_argument("--lean-load", dest="lean_load", action="store_true",
                   default=_env_flag("LEAN_LOAD", True),
                   help="construct pi0 with bf16 default dtype to keep the CPU "
                        "load peak small; requires --half [env LEAN_LOAD]")
    p.add_argument("--no-lean-load", dest="lean_load", action="store_false")
    p.add_argument("--fastwam-cpu-load", dest="fastwam_cpu_load", action="store_true",
                   default=_env_flag("FASTWAM_CPU_LOAD", True),
                   help="fastwam only: stage the state dict on the CPU before the "
                        "device move (lower GPU peak) [env FASTWAM_CPU_LOAD]")
    p.add_argument("--no-fastwam-cpu-load", dest="fastwam_cpu_load", action="store_false")
    p.add_argument("--eval-aug", action="store_true", default=_env_flag("EVAL_AUG", False),
                   help="apply training-grade mask/RGB/depth corruption to every "
                        "query (robustness experiment; off for the protocol) [env EVAL_AUG]")
    p.add_argument("--nogrip", dest="nogrip", action="store_true", default=None,
                   help="zero the gripper mask (default: the checkpoint's norm.json)")
    p.add_argument("--input-mode", default=_env("INPUT_MODE", "") or None,
                   choices=["masked_rgbd", "full_rgbd", "masked_rgb", "masked_rgb_nodepth"],
                   help="override the checkpoint's input mode [env INPUT_MODE]")
    p.add_argument("--drop-px", type=int, default=int(_env("DROP_PX", "300")),
                   help="dropout checkpoints: zero the plane normal when the tool "
                        "mask has fewer pixels [env DROP_PX]")
    p.add_argument("--plane-filter", dest="plane_filter", action="store_true",
                   default=_env_flag("PLANE_FILTER", True),
                   help="median blur + MAD depth trim before the plane fit [env PLANE_FILTER]")
    p.add_argument("--no-plane-filter", dest="plane_filter", action="store_false")
    p.add_argument("--plane-top-cut", type=float, default=float(_env("PLANE_TOP_CUT", "0")),
                   help="fraction of the tool's pixel length removed from the top "
                        "end before the plane fit; a client may override per query "
                        "(sim protocol: 0) [env PLANE_TOP_CUT]")
    p.add_argument("--plane-top-cut-max-deg", type=float,
                   default=float(_env("PLANE_TOP_CUT_MAX_DEG", "20")),
                   help="keep the uncut plane when the cut swings the normal by "
                        "more than this [env PLANE_TOP_CUT_MAX_DEG]")
    p.add_argument("--stub", action="store_true",
                   help="serve a hold-pose chunk without loading weights")
    return p


def config_from_args(a: argparse.Namespace) -> ServeConfig:
    ckpt = a.ckpt or a.ckpt_pos or ""
    return ServeConfig(
        ckpt=ckpt, host=a.host, port=a.port, device=a.device, half=a.half,
        lean_load=a.lean_load, fastwam_cpu_load=a.fastwam_cpu_load,
        eval_aug=a.eval_aug, nogrip=a.nogrip, input_mode=a.input_mode,
        drop_px=a.drop_px, plane_filter=a.plane_filter,
        plane_top_cut=a.plane_top_cut, plane_top_cut_max_deg=a.plane_top_cut_max_deg,
        stub=a.stub)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    if cfg.stub:
        engine: PolicyEngine | StubEngine = StubEngine(cfg)
        log.info("stub mode: hold-pose chunks, no weights loaded")
    else:
        if not cfg.ckpt:
            build_parser().error("--ckpt is required unless --stub is given")
        engine = PolicyEngine(cfg)
    serve(engine, cfg.host, cfg.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
