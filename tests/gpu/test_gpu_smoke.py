"""GPU smoke tier: synthetic cache -> ACT smoke training -> policy server.

Runs in the policy environment on a CUDA GPU.  Every step is a real CLI
invocation (``simretract-build-cache``, ``simretract-train``,
``simretract-serve``) so the console entry points, the environment
delegation and the wire protocol are exercised, not just the library code.
Outputs live in the shared smoke work directory (``SIMRETRACT_SMOKE_DIR``,
default ``<repo>/smoke_tmp``); the sim tier reuses the ACT checkpoint
trained here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from simretract.cli import smoke

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def work() -> Path:
    return smoke.work_dir()


@pytest.fixture(scope="module")
def state() -> dict:
    return {}


def _need(state: dict, key: str) -> object:
    if key not in state:
        pytest.fail(f"prerequisite step '{key}' did not run or failed")
    return state[key]


def test_cuda_available() -> None:
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available(), "the gpu tier needs a CUDA device"


def test_synthetic_cache(work: Path, state: dict) -> None:
    log = work / "logs" / "gpu_cache.log"
    log.unlink(missing_ok=True)
    cache = smoke.write_synthetic_cache(work / "synthetic", log)
    meta = json.loads((cache / "meta.json").read_text())
    assert meta["strict"]["prekept"] is True and meta["recovery"]["flags"]["perturb_trim"] is True
    assert (cache / "strict_plane.npy").is_file() and (cache / "recovery_plane.npy").is_file()
    state["cache"] = cache


def test_train_act_smoke(work: Path, state: dict) -> None:
    cache = _need(state, "cache")
    log = work / "logs" / "gpu_train.log"
    log.unlink(missing_ok=True)
    ckpt = smoke.train_act_smoke(cache, work / "act_smoke", log)
    norm = json.loads((ckpt / "final" / "norm.json").read_text())
    regime = norm["train_regime"]
    assert regime["policy"] == "act" and regime["preset"] is None
    assert norm["input_mode"] == "masked_rgbd" and norm["state_mode"] == "qpos_plane"
    assert len(norm["state_mean"]) == 10 and len(norm["action_mean"]) == 7
    # the run root mirrors final/ so both spellings load
    assert json.loads((ckpt / "config.json").read_text())["type"] == "act"
    state["ckpt"] = ckpt


def test_serve_act_ping_and_query(work: Path, state: dict) -> None:
    ckpt = _need(state, "ckpt")
    log = work / "logs" / "gpu_serve_act.log"
    log.unlink(missing_ok=True)
    with smoke.ServedPolicy(log, ckpt=ckpt) as srv:
        pong = srv.ping()
        assert pong["ok"] is True and pong["eval_aug"] == "0"
        r = pong["regime"]
        assert r["type"] == "act" and r["stub"] is False
        # the trainer mirrors final/ at the run root, so either spelling may be served
        assert Path(r["ckpt"]).resolve() in {ckpt.resolve(), (ckpt / "final").resolve()}
        assert r["half"] is False, "ACT is served in its own precision"
        assert r["plane_filter"] is True and r["drop_px"] == 300
        assert r["plane_top_cut"] == 0.0 and r["chunk"] == 50 and r["eval_aug"] is False
        assert r["input_mode"] == "masked_rgbd" and r["state_dim"] == 10
        chunk = srv.query(smoke.synthetic_query())
        assert chunk.shape == (50, 7) and np.isfinite(chunk).all()


def test_serve_stub(work: Path) -> None:
    log = work / "logs" / "gpu_serve_stub.log"
    log.unlink(missing_ok=True)
    with smoke.ServedPolicy(log, stub=True) as srv:
        r = srv.ping()["regime"]
        assert r["stub"] is True and r["type"] == "stub" and r["chunk"] == 50
        msg = smoke.synthetic_query()
        msg["state"] = np.arange(14, dtype=np.float32) * 0.01
        chunk = srv.query(msg)
        assert chunk.shape == (50, 7)
        assert np.allclose(chunk, np.repeat(msg["state"][None, :7], 50, 0)), "hold-pose chunk"


def test_train_pi0_smoke(work: Path, state: dict) -> None:
    """pi0 needs the released base weights and a training-class GPU."""
    from simretract import config

    base = config.paths()["checkpoints"] / "pi0_base"
    if not (base / "config.json").is_file():
        pytest.skip("checkpoints/pi0_base absent (simretract-download base-weights)")
    torch = pytest.importorskip("torch")
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_gb < 100:
        pytest.skip(f"pi0 smoke needs the reference training GPU (>= 100 GB, have {total_gb:.0f} GB)")
    cache = _need(state, "cache")
    log = work / "logs" / "gpu_train_pi0.log"
    log.unlink(missing_ok=True)
    out = smoke.fresh_dir(work / "pi0_smoke")
    rc = smoke.run_logged([__import__("sys").executable, "-m", "simretract.cli.train",
                           "--preset", "baseline", "--smoke", "--cache", cache, "--out", out,
                           "--init", base, "--num-workers", "2"], log, timeout=3600)
    assert rc == 0, smoke.tail(log)
    assert (out / "final" / "model.safetensors").is_file()
