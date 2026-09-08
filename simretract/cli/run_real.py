"""simretract-run-real: one-command real-robot stack.

Boots (or reuses) the policy server under the policy interpreter and the
SAM 3 mask server under the mask interpreter, ping-verifies both including
the serving regime, and then runs the client in the chosen mode:

* ``servers-only`` (default): boot + verify, print the ready line;
* ``dry``: client with perception + inference only, the arm is read-only;
* ``go``: live run (the client still asks for ENTER before moving).

Client regimes come from ``configs/real/default.yaml`` presets (``default``,
``parity``, ``conservative``); flags after the mode override preset values
and are passed through to :mod:`simretract.real.client`. Per-backbone
execution horizons (Fast-WAM 48 of 48 or 10 of 32) are derived from the
checkpoint's ``config.json`` / ``norm.json``. The launcher only ever kills
processes bound to its own two ports, found by PID.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from simretract.real import (FOLLOWER_IP_PLACEHOLDER, default_runs_root,
                             follower_ip_is_set, load_real_config,
                             preset_client_args, resolve_preset)
from simretract.serve.protocol import recv_msg, send_msg

log = logging.getLogger("simretract.run_real")

POLICY_MODULE = "simretract.serve.policy_server"
MASK_MODULE = "simretract.real.mask_server"
CLIENT_MODULE = "simretract.real.client"


# ------------------------------------------------------------------ helpers
def ping(host: str, port: int, timeout: float = 10.0) -> dict[str, Any] | None:
    """Ping a protocol server; None when nothing answers."""
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.settimeout(timeout)
            send_msg(s, {"op": "ping"})
            out = recv_msg(s)
    except (OSError, EOFError, ValueError):
        return None
    return out if isinstance(out, dict) else None


def port_pid(port: int) -> int | None:
    """PID of the process listening on ``port`` (via ``ss``, falling back to ``lsof``)."""
    try:
        out = subprocess.run(["ss", "-ltnp", f"sport = :{port}"], capture_output=True, text=True,
                             timeout=10).stdout
        for tok in out.replace(",", " ").split():
            if tok.startswith("pid="):
                return int(tok[4:])
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    try:
        out = subprocess.run(["lsof", "-t", "-iTCP:%d" % port, "-sTCP:LISTEN"], capture_output=True,
                             text=True, timeout=10).stdout.split()
        return int(out[0]) if out else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def kill_port(port: int, wait_s: float = 15.0) -> None:
    """Terminate whatever listens on ``port`` (SIGTERM, then SIGKILL after ``wait_s``)."""
    pid = port_pid(port)
    if pid is None:
        return
    log.info("stopping pid %d on port %d", pid, port)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            return
        t0 = time.time()
        while time.time() - t0 < wait_s:
            if port_pid(port) is None:
                return
            time.sleep(0.5)


def resolve_ckpt_dir(path: str) -> str:
    """Absolute checkpoint directory, accepting the trainer's ``<run>/final`` layout too.

    The policy server resolves the same way and reports the resolved
    directory in its ping, so the reuse check compares like with like.
    """
    p = Path(path)
    for cand in (p, p / "final"):
        if (cand / "config.json").is_file() and (cand / "norm.json").is_file():
            return os.path.realpath(cand)
    raise SystemExit(f"{path}: no config.json + norm.json in the directory or its final/ subdirectory")


def read_ckpt_meta(ckpt: str) -> dict[str, Any]:
    """Policy type and layout from a checkpoint's config.json + norm.json."""
    cdir = Path(ckpt)
    cj, nj = cdir / "config.json", cdir / "norm.json"
    if not (cj.is_file() and nj.is_file()):
        raise SystemExit(f"{ckpt} lacks config.json / norm.json")
    with open(cj) as f:
        c = json.load(f)
    with open(nj) as f:
        n = json.load(f)
    return {"type": c.get("type", "pi0"),
            "input_mode": n.get("input_mode", "masked_rgbd"),
            "state_mode": n.get("state_mode", "?"),
            "no_aug": bool(n.get("no_aug", False)),
            "chunk_h": int(n.get("chunk_h", c.get("chunk_size", c.get("action_horizon", 50))))}


def exec_h_for(meta: dict[str, Any], fastwam_exec_h: int | None = None) -> int | None:
    """Per-backbone execution horizon override (None = keep the preset).

    Fast-WAM checkpoints execute the whole 48-step chunk, or 10 of a 32-step
    chunk; every other shipped backbone runs the preset value.
    """
    if meta.get("type") != "fastwam":
        return None
    if fastwam_exec_h is not None:
        return int(fastwam_exec_h)
    chunk = int(meta.get("chunk_h", 48))
    return {48: 48, 32: 10}.get(chunk, chunk)


def describe_ckpt(meta: dict[str, Any]) -> str:
    plane = {"qpos": "off (7-D state)", "qpos_plane": "on (plane normal from depth + mask)"}
    inp = {"masked_rgbd": "masked RGB + masked depth",
           "full_rgbd": "full RGB + full depth (plane still from the mask)",
           "masked_rgb_nodepth": "masked RGB only (plane still from depth + mask)"}
    return (f"policy {meta['type']} | chunk {meta['chunk_h']} | images "
            f"{inp.get(meta['input_mode'], meta['input_mode'])} | plane "
            f"{plane.get(meta['state_mode'], meta['state_mode'])} | train aug "
            f"{'off' if meta['no_aug'] else 'on'}")


def policy_regime_ok(pp: dict[str, Any] | None, ckpt: str, *, metric_required: bool = False) -> tuple[bool, str]:
    """Whether a live policy server may be reused for ``ckpt``.

    It must serve this exact checkpoint without evaluation-time corruption,
    with the plane filter on and no server-side top cut (the cut travels per
    query from the client).
    """
    if not pp or not pp.get("ok", True):
        return False, "no server"
    if os.path.realpath(str(pp.get("ckpt", ""))) != os.path.realpath(ckpt):
        return False, f"serves {pp.get('ckpt')}"
    if str(pp.get("eval_aug", "0")) in ("1", "True", "true"):
        return False, "started with EVAL_AUG"
    regime = pp.get("regime") if isinstance(pp.get("regime"), dict) else pp
    if metric_required and not pp.get("metric_plane_addon"):
        return False, "metric crop support missing"
    if float(regime.get("plane_top_cut_m", 0) or 0) != 0:
        return False, "server-side metric crop must be zero"
    if "plane_filter" in regime and not regime["plane_filter"]:
        return False, "plane filter off"
    if "plane_top_cut" in regime and float(regime["plane_top_cut"] or 0.0) != 0.0:
        return False, f"server-side plane_top_cut {regime['plane_top_cut']}"
    return True, "ok"


def mask_regime_ok(reply: dict | None, ckpt: str, prompt: str, conf: float) -> bool:
    """Reuse only the requested warmed tracker with the selected-ID fix."""
    return bool(reply and reply.get("ok") and reply.get("warm_reset")
                and reply.get("locked_track_keepalive")
                and reply.get("protocol_version") == 2
                and os.path.realpath(str(reply.get("ckpt", ""))) == os.path.realpath(ckpt)
                and reply.get("prompt") == prompt and float(reply.get("conf", -1)) == conf)


def resolve_cut_options(preset: dict, fraction: float | None, metric: float | None,
                        extra: list[str]) -> dict:
    """Explicit legacy cuts disable the metric default, before or after mode."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--plane-top-cut", type=float)
    parser.add_argument("--plane-top-cut-m", type=float, nargs="?", const=0.01)
    tail, _ = parser.parse_known_args([a for a in extra if a != "--"])
    if tail.plane_top_cut is not None:
        fraction = tail.plane_top_cut
    if tail.plane_top_cut_m is not None:
        metric = tail.plane_top_cut_m
    result = dict(preset)
    if fraction is not None:
        result["plane_top_cut"] = fraction
        if metric is None:
            result["plane_top_cut_m"] = 0.0
    if metric is not None:
        result["plane_top_cut_m"] = metric
    if not math.isfinite(result["plane_top_cut_m"]) or result["plane_top_cut_m"] < 0:
        raise SystemExit("--plane-top-cut-m must be finite and nonnegative")
    return result


def boot(cmd: list[str], env: dict[str, str], log_path: Path) -> subprocess.Popen:
    """Start a detached server process writing to ``log_path``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("$ %s  (log %s)", " ".join(cmd), log_path)
    lf = open(log_path, "ab")
    return subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)


def wait_for_ping(host: str, port: int, proc: subprocess.Popen | None, log_path: Path | None,
                  timeout: float, name: str) -> dict[str, Any]:
    """Poll the port until it answers a ping; fail with the log tail if the process dies."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        pp = ping(host, port, timeout=5.0)
        if pp is not None:
            return pp
        if proc is not None and proc.poll() is not None:
            break
        time.sleep(5)
    tail = ""
    if log_path is not None and log_path.is_file():
        with open(log_path, "rb") as f:
            tail = f.read()[-2000:].decode("utf-8", "replace")
    raise SystemExit(f"{name} did not come up on {host}:{port} within {timeout:.0f}s\n{tail}")


def resolve_follower_ip(cli_ip: str | None, cfg: dict[str, Any]) -> str:
    """Robot address from the flag or the config; refuses the CHANGE_ME placeholder."""
    ip = cli_ip if cli_ip else (cfg.get("robot") or {}).get("follower_ip")
    if not follower_ip_is_set(ip):
        raise SystemExit(f"robot.follower_ip is {FOLLOWER_IP_PLACEHOLDER!r}: set it in "
                         "configs/real/default.yaml or pass --follower-ip before running")
    return str(ip)


# ------------------------------------------------------------------ CLI
class _RunParser(argparse.ArgumentParser):
    def parse_known_args(self, args=None, namespace=None):
        # A bare optional float must not consume the following positional mode.
        args = list(sys.argv[1:] if args is None else args)
        args = ["--plane-top-cut-m=0.01"
                if value == "--plane-top-cut-m" and i + 1 < len(args)
                and args[i + 1] in ("servers-only", "dry", "go") else value
                for i, value in enumerate(args)]
        return super().parse_known_args(args, namespace)


def build_parser() -> argparse.ArgumentParser:
    ap = _RunParser(
        prog="simretract-run-real",
        description="Boot/reuse the policy + SAM 3 servers and run the real-arm client. "
                    "Launcher options precede the mode; every argument after the mode "
                    "(servers-only | dry | go) is passed to the client unchanged.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("mode", nargs="?", choices=["servers-only", "dry", "go"], default="servers-only",
                    help="servers-only: boot + verify; dry: client without motion; go: live run")
    ap.add_argument("--ckpt", required=True, help="checkpoint directory (config.json + norm.json + weights)")
    ap.add_argument("--preset", default="default", help="client regime from configs/real/default.yaml")
    ap.add_argument("--config", default=None, help="real-run YAML (default configs/real/default.yaml)")
    ap.add_argument("--follower-ip", dest="follower_ip", default=None,
                    help="follower controller IP (default robot.follower_ip from the config)")
    ap.add_argument("--plane-top-cut", dest="plane_top_cut", type=float, default=None,
                    help="legacy plane-only pixel fraction; disables the metric default unless specified")
    ap.add_argument("--plane-top-cut-m", type=float, nargs="?", const=0.01, default=None,
                    help="crop tool tip from plane and model inputs in metres; 0 selects uncut legacy mode")
    ap.add_argument("--exec-h", dest="exec_h", type=int, default=None,
                    help="override the execution horizon (also used for --exec-min)")
    ap.add_argument("--fastwam-exec-h", dest="fastwam_exec_h", type=int, default=None,
                    help="Fast-WAM only: actions executed per query (default 48 of 48, 10 of 32)")
    ap.add_argument("--keep-server", dest="keep_server", action=argparse.BooleanOptionalAction, default=True,
                    help="reuse a live policy server that already serves this checkpoint in the right regime")
    ap.add_argument("--sam3-ckpt", dest="sam3_ckpt", default=None, help="SAM 3 weights (default mask.sam3_ckpt)")
    ap.add_argument("--prompt", default=None, help="SAM 3 text prompt (default mask.prompt)")
    ap.add_argument("--conf", type=float, default=None, help="SAM 3 confidence gate (default mask.conf)")
    ap.add_argument("--policy-port", dest="policy_port", type=int, default=None)
    ap.add_argument("--mask-port", dest="mask_port", type=int, default=None)
    ap.add_argument("--boot-timeout", dest="boot_timeout", type=float, default=None,
                    help="seconds to wait for each server (default services.boot_timeout_s)")
    ap.add_argument("client_args", nargs=argparse.REMAINDER,
                    help="extra flags passed to the client after the preset (e.g. --no-view --yes)")
    return ap


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    a = build_parser().parse_args(argv)
    cfg = load_real_config(a.config)
    follower_ip = resolve_follower_ip(a.follower_ip, cfg)
    ckpt = resolve_ckpt_dir(a.ckpt)
    meta = read_ckpt_meta(ckpt)
    log.info("checkpoint %s: %s", ckpt, describe_ckpt(meta))
    preset = resolve_cut_options(resolve_preset(cfg, a.preset), a.plane_top_cut,
                                 a.plane_top_cut_m, a.client_args)
    eh = a.exec_h if a.exec_h is not None else exec_h_for(meta, a.fastwam_exec_h)
    if eh is not None:
        preset["exec_h"] = preset["exec_min"] = int(eh)
        log.info("execution horizon %d/%d for this backbone", eh, eh)

    svc = cfg.get("services") or {}
    host = str(svc.get("host", "127.0.0.1"))
    policy_port = int(a.policy_port or svc.get("policy_port", 8765))
    mask_port = int(a.mask_port or svc.get("mask_port", 8766))
    boot_timeout = float(a.boot_timeout or svc.get("boot_timeout_s", 450))
    mcfg = cfg.get("mask") or {}
    sam3_ckpt = a.sam3_ckpt or mcfg.get("sam3_ckpt")
    prompt = a.prompt if a.prompt is not None else str(mcfg.get("prompt", "metal rake"))
    conf = float(a.conf if a.conf is not None else mcfg.get("conf", 0.2))
    if not sam3_ckpt or not os.path.isfile(sam3_ckpt):
        raise SystemExit(f"SAM 3 checkpoint not found: {sam3_ckpt} "
                         "(set mask.sam3_ckpt or --sam3-ckpt; see checkpoints/README.md)")

    from simretract import config
    policy_py = str(config.interpreter("fastwam" if meta["type"] == "fastwam" else "policy"))
    mask_py = str(config.interpreter("mask"))
    logs = default_runs_root() / "_servers"

    # ---- policy server -------------------------------------------------
    pp = ping(host, policy_port)
    ok, why = policy_regime_ok(pp, ckpt, metric_required=preset["plane_top_cut_m"] > 0)
    if a.keep_server and ok:
        log.info("reusing the live policy server on %d (%s)", policy_port, ckpt)
    else:
        if pp is not None:
            log.info("restarting the policy server on %d (%s)", policy_port, why)
        kill_port(policy_port)
        env = dict(os.environ, HALF="1", CKPT=ckpt, PORT=str(policy_port))
        for key, value in {"EVAL_AUG": "0", "PLANE_FILTER": "1", "PLANE_TOP_CUT": "0",
                           "PLANE_TOP_CUT_M": "0"}.items():
            env[key] = env["SIMRETRACT_" + key] = value
        proc = boot([policy_py, "-m", POLICY_MODULE, "--ckpt", ckpt, "--port", str(policy_port), "--half"],
                    env, logs / f"policy_{policy_port}.log")
        pp = wait_for_ping(host, policy_port, proc, logs / f"policy_{policy_port}.log", boot_timeout,
                           "policy server")
        ok, why = policy_regime_ok(pp, ckpt, metric_required=preset["plane_top_cut_m"] > 0)
        if not ok:
            raise SystemExit(f"policy server regime mismatch after boot: {why} ({pp})")
    log.info("policy ping: %s", pp)

    # ---- mask server ---------------------------------------------------
    mp = ping(host, mask_port)
    if mask_regime_ok(mp, str(sam3_ckpt), prompt, conf):
        log.info("reusing the live mask server on %d (prompt %r)", mask_port, prompt)
    else:
        if mp is not None:
            log.info("restarting the mask server on %d (prompt %r != %r)", mask_port, mp.get("prompt"), prompt)
        kill_port(mask_port)
        proc = boot([mask_py, "-m", MASK_MODULE, "--ckpt", str(sam3_ckpt), "--prompt", prompt,
                     "--conf", str(conf), "--host", host, "--port", str(mask_port)],
                    dict(os.environ), logs / f"mask_{mask_port}.log")
        mp = wait_for_ping(host, mask_port, proc, logs / f"mask_{mask_port}.log", boot_timeout, "mask server")
    if not mask_regime_ok(mp, str(sam3_ckpt), prompt, conf):
        raise SystemExit("mask server is missing the requested checkpoint or tracking fixes")
    log.info("mask ping: %s", mp)
    log.info("servers verified")

    # ---- client --------------------------------------------------------
    client = [policy_py, "-m", CLIENT_MODULE, *preset_client_args(preset),
              "--follower-ip", follower_ip, "--host", host, "--port", str(policy_port),
              "--mask-port", str(mask_port)]
    extra = [x for x in a.client_args if x != "--"]
    if a.mode == "servers-only":
        print("ready. next: simretract-run-real --ckpt ... dry   (or: go)")
        return 0
    if a.mode == "go":
        client.append("--go")
    client += extra
    log.info("client mode %s / preset %s: %s", a.mode, a.preset, " ".join(client[3:]))
    return subprocess.call(client)


if __name__ == "__main__":
    sys.exit(main())
