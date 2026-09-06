"""``simretract-smoke``: run one smoke-test tier, plus the helpers the tiers share.

Tiers (``tests/<tier>``, pytest marker of the same name):

* ``unit``      no GPU, no Isaac Sim                              (< 1 min)
* ``gpu``       synthetic cache -> ACT smoke training -> policy server ping,
                regime check and one query; stub server              (~1 min)
* ``sim``       build the tool assets from the purchased file under
                ``assets/source/`` (or the ``tool_source`` override; skipped
                with a message without one), with the fingerprint, USD and
                grasp-candidate checks; record a strict
                episode and a recovery draw; build both caches; evaluate one
                seed with the ACT smoke checkpoint and with the stub server
                (~10 min; up to ~25 min when a Kit worker is retried)
* ``real-dry``  client dry run against the servers (skipped without the SAM 3
                weights, a robot address and a D405)                 (~2 min)
* ``release``   publication gates (licence files, hub ids, asset exclusion,
                release wording, checksum tables); fails until every gate is
                met                                                   (seconds)

Every tier's pytest process runs in the policy environment; simulator work is
delegated to the ``sim`` interpreter by the CLIs themselves.  The helpers
below (synthetic cache, ACT smoke checkpoint, a served policy) are what the
``gpu`` and ``sim`` test modules call, so the two tiers agree on the layout
of the shared work directory (``SIMRETRACT_SMOKE_DIR``, default
``<repo>/smoke_tmp``, git-ignored).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

TIERS: dict[str, tuple[str, str, str, str]] = {
    # tier: (interpreter role, test directory, pytest marker, expected wall time)
    "unit": ("policy", "tests/unit", "unit", "< 1 min"),
    "gpu": ("policy", "tests/gpu", "gpu", "~ 1 min"),
    "sim": ("policy", "tests/sim", "sim", "~ 10 min (up to ~ 25 min with a retried Kit worker)"),
    "real-dry": ("policy", "tests/real_dry", "real_dry", "~ 2 min (skips without hardware)"),
    "release": ("policy", "tests/release", "release", "seconds"),
}

SERVING_MARK = "serving on "


def repo_root() -> Path:
    from simretract import config

    return config.repo_root()


def work_dir() -> Path:
    """Shared work directory of the smoke tiers (created on demand)."""
    raw = os.environ.get("SIMRETRACT_SMOKE_DIR")
    d = Path(raw).expanduser() if raw else repo_root() / "smoke_tmp"
    d.mkdir(parents=True, exist_ok=True)
    return d


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def run_logged(cmd: Sequence[str], log_path: Path, *, env: dict[str, str] | None = None,
               timeout: float | None = None, cwd: Path | None = None) -> int:
    """Run ``cmd`` with stdout+stderr appended to ``log_path``; returns the
    exit code (-1 on timeout, after killing the process group)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as lf:
        lf.write(f"# {time.strftime('%H:%M:%S')} $ {' '.join(map(str, cmd))}\n".encode())
        lf.flush()
        proc = subprocess.Popen([str(c) for c in cmd], stdout=lf, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=env, cwd=cwd,
                                start_new_session=True)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            proc.wait(timeout=30)
            lf.write(f"# killed after {timeout}s\n".encode())
            return -1


def tail(log_path: Path, n: int = 40) -> str:
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def fresh_dir(path: Path) -> Path:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


# ------------------------------------------------------------- synthetic data
def write_synthetic_cache(root: Path, log_path: Path, *, workers: int = 4) -> Path:
    """Write a synthetic raw set under ``root/raw`` and build the ``strict``
    and ``recovery`` caches (production flags, plane sidecar) into
    ``root/cache`` with the cache CLI of the current interpreter."""
    from simretract.data.synthetic import write_synthetic_raw

    fresh_dir(root)
    raw = write_synthetic_raw(root / "raw")
    cache = root / "cache"
    for set_key in ("strict", "recovery"):
        rc = run_logged([sys.executable, "-m", "simretract.cli.build_cache", "--set", set_key,
                         "--raw", raw[set_key], "--out", cache, "--workers", str(workers)],
                        log_path, timeout=900)
        if rc != 0:
            raise RuntimeError(f"build-cache --set {set_key} failed (rc {rc}):\n{tail(log_path)}")
    meta = json.loads((cache / "meta.json").read_text())
    if not all(meta.get(k, {}).get("N", 0) > 0 for k in ("strict", "recovery")):
        raise RuntimeError(f"synthetic cache is empty: {meta}")
    return cache


def train_act_smoke(cache: Path, out: Path, log_path: Path) -> Path:
    """``simretract-train --preset act --smoke`` on ``cache``; returns the
    checkpoint directory (``out``, with ``final/`` inside)."""
    fresh_dir(out)
    rc = run_logged([sys.executable, "-m", "simretract.cli.train", "--preset", "act", "--smoke",
                     "--cache", cache, "--out", out, "--num-workers", "2"],
                    log_path, timeout=1800)
    if rc != 0:
        raise RuntimeError(f"ACT smoke training failed (rc {rc}):\n{tail(log_path)}")
    for name in ("config.json", "model.safetensors", "norm.json"):
        if not (out / "final" / name).is_file():
            raise RuntimeError(f"ACT smoke checkpoint lacks final/{name}:\n{tail(log_path)}")
    return out


def ensure_act_smoke_checkpoint(work: Path, log_path: Path) -> Path:
    """The ACT smoke checkpoint: ``SIMRETRACT_SMOKE_ACT_CKPT`` when set, else
    ``<work>/act_smoke`` (reused when present, otherwise trained on a fresh
    synthetic cache under ``<work>/synthetic``)."""
    override = os.environ.get("SIMRETRACT_SMOKE_ACT_CKPT")
    if override:
        p = Path(override)
        if not ((p / "config.json").is_file() or (p / "final" / "config.json").is_file()):
            raise FileNotFoundError(f"SIMRETRACT_SMOKE_ACT_CKPT={p}: no config.json")
        return p
    ckpt = work / "act_smoke"
    if (ckpt / "final" / "config.json").is_file():
        return ckpt
    cache = write_synthetic_cache(work / "synthetic", log_path)
    return train_act_smoke(cache, ckpt, log_path)


def synthetic_query(rng: np.random.Generator | None = None) -> dict[str, Any]:
    """One observation message in the evaluator's wire format: a tilted
    depth plane inside an elliptical tool mask, an empty gripper mask and a
    14-D state (7 joints + 7-D grasp pose, as the simulator sends)."""
    rng = rng or np.random.default_rng(0)
    H, W = 480, 640
    yy, xx = np.mgrid[:H, :W]
    tool = (((xx - 330) / 120.0) ** 2 + ((yy - 240) / 50.0) ** 2) < 1.0
    rgb = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    rgb[tool] = (200, 60, 90)
    dep = (300 + 0.2 * (xx - 320) + 0.1 * (yy - 240)).astype(np.uint16)
    dep[~tool] = 900
    state = np.zeros(14, np.float32)
    state[6] = 0.044
    return {"rgb": rgb, "depth_mm": dep, "mask": (tool * 255).astype(np.uint8),
            "gmask": np.zeros((H, W), np.uint8), "state": state, "plane_top_cut": 0.0}


# ------------------------------------------------------------- served policy
class ServedPolicy:
    """A policy server subprocess (checkpoint or ``--stub``) for the duration
    of a ``with`` block; ``ping()`` and ``query()`` talk to it."""

    def __init__(self, log_path: Path, *, ckpt: Path | None = None, stub: bool = False,
                 port: int | None = None, boot_timeout: float = 600.0,
                 extra: Sequence[str] = ()) -> None:
        if not stub and ckpt is None:
            raise ValueError("ckpt or stub=True is required")
        self.log_path = log_path
        self.ckpt = ckpt
        self.stub = stub
        self.port = port or find_free_port()
        self.boot_timeout = boot_timeout
        self.extra = list(extra)
        self.proc: subprocess.Popen | None = None

    def command(self) -> list[str]:
        cmd = [sys.executable, "-m", "simretract.cli.serve", "--port", str(self.port)]
        cmd += ["--stub"] if self.stub else ["--ckpt", str(self.ckpt)]
        return cmd + self.extra

    def __enter__(self) -> "ServedPolicy":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        lf = open(self.log_path, "ab")
        lf.write(f"# $ {' '.join(self.command())}\n".encode())
        self.proc = subprocess.Popen(self.command(), stdout=lf, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        lf.close()
        t0 = time.time()
        while time.time() - t0 < self.boot_timeout:
            if SERVING_MARK in self.log_path.read_text(errors="replace"):
                return self
            if self.proc.poll() is not None:
                raise RuntimeError(f"policy server exited during boot (rc {self.proc.returncode}):\n"
                                   f"{tail(self.log_path)}")
            time.sleep(1.0)
        self.__exit__(None, None, None)
        raise RuntimeError(f"policy server did not report '{SERVING_MARK.strip()}' within "
                           f"{self.boot_timeout}s:\n{tail(self.log_path)}")

    def __exit__(self, *exc: object) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=20)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except OSError:
                    pass
        self.proc = None

    def ping(self) -> dict:
        from simretract.serve.protocol import PolicyClient

        with PolicyClient("127.0.0.1", self.port, timeout=120) as cl:
            return cl.ping()

    def query(self, msg: dict) -> np.ndarray:
        from simretract.serve.protocol import PolicyClient

        with PolicyClient("127.0.0.1", self.port, timeout=300) as cl:
            reply = cl.request(msg)
        if isinstance(reply, dict) and "error" in reply:
            raise RuntimeError(f"server error: {reply['error']}")
        return np.asarray(reply, dtype=np.float32)


# ------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="simretract-smoke",
        description="Run one smoke-test tier with pytest under the interpreter it needs "
                    "(same as scripts/smoke_test.sh). Extra arguments after `--` go to pytest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--tier", choices=[*TIERS, "all"], default="unit",
                   help="tier to run; all = unit + gpu + sim")
    p.add_argument("pytest_args", nargs="*", help="extra pytest arguments")
    return p


def run_tier(tier: str, pytest_args: Sequence[str] = ()) -> int:
    from simretract import config

    role, directory, marker, eta = TIERS[tier]
    python = str(config.interpreter(role))
    root = repo_root()
    cmd = [python, "-m", "pytest", str(root / directory), "-m", marker, "-q",
           "--durations=0", "-p", "no:cacheprovider", *pytest_args]
    print(f"== smoke tier '{tier}' ({directory}) under {python} -- expected wall time {eta}",
          flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(root))
    print(f"== smoke tier '{tier}': {'PASS' if rc == 0 else f'FAIL (rc {rc})'} "
          f"in {time.time() - t0:.0f} s", flush=True)
    return rc


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    tiers = ["unit", "gpu", "sim"] if a.tier == "all" else [a.tier]
    for t in tiers:
        rc = run_tier(t, [x for x in a.pytest_args if x != "--"])
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
