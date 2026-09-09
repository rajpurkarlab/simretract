"""Simulation-benchmark driver: policy server + one Kit process per seed,
first-attempt scoring with optional retries.

The benchmark protocol (``configs/eval/sim.yaml``) is: bf16 server with the
plane filter on, no plane top-cut, dropout gate at 300 px; the evaluator
executes the full chunk (exec-h 50; 48 for fastwam) with EMA smoothing
0.45, skips the scripted grasp search, times out after 450 control ticks
and runs the release + load battery.  Each seed runs in a fresh Kit process
with a wall-clock cap; a seed gets up to ``--launches`` launches per pass,
and optional later passes re-run non-completed seeds. Earlier results are
kept as ``result_<seed>.try<k>.json`` without overwriting previous attempts.
Scoring uses the first attempt unless ``--score-attempt latest`` is selected.
Completed seeds are never re-run,
so an interrupted run can simply be restarted.

``--seeds`` is the set to RUN, ``--score-seeds`` the set the ROW line and
table are computed over (they differ when extending an existing run with
new seeds).  Every Kit worker is spawned in its own session and killed as a
process group on timeout; a worker whose log already shows the DONE line
but that lingers in Kit teardown is reaped.  The server is booted before
the first seed, checked for liveness before every seed, and rebooted when
it died; its ping must report the expected regime or the run is refused.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from simretract.eval import score as scoring
from simretract.serve.policy_server import checkpoint_type, resolve_checkpoint_dir
from simretract.serve.protocol import PolicyClient

log = logging.getLogger("simretract.eval.driver")

#: Protocol defaults; ``configs/eval/sim.yaml`` overrides them (see load_protocol).
PROTOCOL: dict[str, Any] = {
    "exec_h": {"pi0": 50, "smolvla": 50, "act": 50, "pi05": 50, "pi0_fast": 50, "fastwam": 48},
    "smooth": 0.45,
    "skip_search": True,
    "max_steps": 450,
    "half": True,
    "fastwam_cpu_load": True,
    "plane_filter": True,
    "drop_px": 300,
    "tries": 10,
    "mu": 0.4,
    "hand_style": "test",
    "plane_top_cut": 0.0,
    "passes": 1,
    "score_attempt": "first",
    "launches": 2,
    "timeout_s": 900,
    "server_boot_s": 600,
    "port": 8767,
    "host": "127.0.0.1",
    "tool": "retractor_rake",
    "seeds": "test_core50.json",
    "score_seeds": "test_core50.json",
    "no_battery": False,
    "keep_frames": False,
}
#: sim.yaml keys (top level / server: / driver:) -> PROTOCOL keys.
_YAML_KEYS = {
    "": ("seeds", "score_seeds", "passes", "score_attempt", "tool", "smooth", "skip_search", "max_steps",
         "tries", "mu", "hand_style", "plane_top_cut", "no_battery", "keep_frames"),
    "server": {"port": "port", "host": "host", "half": "half", "plane_filter": "plane_filter",
               "drop_px": "drop_px", "fastwam_cpu_load": "fastwam_cpu_load"},
    "driver": {"seed_timeout_s": "timeout_s", "launches_per_seed": "launches",
               "server_boot_timeout_s": "server_boot_s"},
}
#: Types served in their own precision (no bf16 cast).
_OWN_PRECISION = ("act", "fastwam", "pi0_fast")
#: Evaluator outcomes that are deterministic for a seed (no result file is
#: written; re-running would only reproduce them).
TERMINAL_OUTCOMES = frozenset({"scene_reject", "no_grasp"})
_SERVING_RE = re.compile(r"serving on ")
_DONE_MARK = "[M5] DONE"
_OUTCOME_RE = re.compile(r"^\[M5\] (\{.*\})\s*$")


def load_protocol(data: dict[str, Any] | None = None) -> dict[str, Any]:
    """PROTOCOL updated from ``configs/eval/sim.yaml`` (``data`` overrides the
    file, for tests).  Missing keys keep the built-in protocol values."""
    proto = json.loads(json.dumps(PROTOCOL))
    if data is None:
        try:
            from simretract import config

            data = config.eval_protocol()
        except Exception as e:  # no checkout configuration: built-in defaults
            log.warning("using built-in protocol defaults (%s)", e)
            return proto
    for k in _YAML_KEYS[""]:
        if k in data:
            proto[k] = data[k]
    for section, keys in (("server", _YAML_KEYS["server"]), ("driver", _YAML_KEYS["driver"])):
        sub = data.get(section) or {}
        for yk, pk in keys.items():
            if yk in sub:
                proto[pk] = sub[yk]
    exec_h = data.get("exec_h")
    if isinstance(exec_h, dict):
        for k, v in exec_h.items():
            if isinstance(v, int) and not isinstance(v, bool):
                proto["exec_h"][str(k)] = v
    return proto


@dataclass
class DriverConfig:
    ckpt: Path
    out: Path
    seeds_json: Path
    score_json: Path
    name: str
    passes: int = 1
    score_attempt: str = "first"
    launches: int = 2
    timeout_s: int = 900
    server_boot_s: int = 600
    host: str = "127.0.0.1"
    port: int = 8767
    exec_h: int = 50
    smooth: float = 0.45
    skip_search: bool = True
    max_steps: int = 450
    tries: int = 10
    mu: float = 0.4
    hand_style: str = "test"
    plane_top_cut: float = 0.0
    plane_filter: bool = True
    drop_px: int = 300
    half: bool = True
    fastwam_cpu_load: bool = True
    tool: str = "retractor_rake"
    keep_frames: bool = False
    review_mp4: bool = True
    battery: bool = True
    stub: bool = False
    policy_python: str = ""
    sim_python: str = ""
    policy_type: str = "pi0"


# ------------------------------------------------------------- pass logic
def result_path(out: Path, seed: int) -> Path:
    return out / f"result_{seed}.json"


def _outcome(path: Path) -> str | None:
    try:
        with open(path) as f:
            return str(json.load(f).get("outcome"))
    except (OSError, ValueError):
        return None


def select_seeds(out: Path, run_seeds: list[int], pass_no: int) -> list[int]:
    """Seeds to run in ``pass_no``: on pass 1 every run seed without a
    result (resume); on later passes every run seed whose result is not
    ``completed`` (archived under the next unused ``.try<k>.json``) plus seeds
    still without a result. Completed seeds are never selected; existing
    attempt archives are preserved across resumed invocations."""
    sel: list[int] = []
    for s in run_seeds:
        p = result_path(out, s)
        if not p.is_file():
            sel.append(s)
            continue
        if pass_no > 1 and _outcome(p) != "completed":
            indices = [int(m.group(1)) for q in out.glob(f"result_{s}.try*.json")
                       if (m := re.fullmatch(rf"result_{s}\.try(\d+)\.json", q.name))]
            archive_no = max(indices, default=0) + 1
            p.rename(out / f"result_{s}.try{archive_no}.json")
            sel.append(s)
    return sel


def terminal_outcome(attempts: dict[str, list[dict]], seed: int) -> str | None:
    """The ledger's terminal outcome for ``seed`` (see TERMINAL_OUTCOMES), if any."""
    for a in attempts.get(str(seed), []):
        if a.get("outcome") in TERMINAL_OUTCOMES:
            return str(a["outcome"])
    return None


# ------------------------------------------------------------- processes
def listening_pids(port: int) -> list[int]:
    """PIDs listening on ``port`` (iproute2 ``ss``; empty when unavailable)."""
    try:
        txt = subprocess.run(["ss", "-ltnp", f"sport = :{port}"], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return sorted({int(m) for m in re.findall(r"pid=(\d+)", txt)})


def free_port(port: int) -> None:
    """Terminate whatever listens on ``port`` (by PID), escalating to SIGKILL."""
    pids = listening_pids(port)
    for pid in pids:
        log.info("port %d busy: terminating pid %d", port, pid)
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue
    if pids:
        for _ in range(10):
            time.sleep(0.5)
            if not listening_pids(port):
                return
        for pid in listening_pids(port):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(1.0)


def _count_serving(log_path: Path) -> int:
    try:
        with open(log_path, errors="replace") as f:
            return sum(1 for line in f if _SERVING_RE.search(line))
    except OSError:
        return 0


def _killpg(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


class Server:
    """The policy server subprocess for one run."""

    def __init__(self, cfg: DriverConfig, log_path: Path) -> None:
        self.cfg = cfg
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None

    def command(self) -> list[str]:
        c = self.cfg
        cmd = [c.policy_python, "-m", "simretract.serve.policy_server",
               "--host", c.host, "--port", str(c.port),
               "--drop-px", str(c.drop_px), "--plane-top-cut", str(c.plane_top_cut),
               "--plane-filter" if c.plane_filter else "--no-plane-filter",
               "--half" if c.half else "--no-half",
               "--fastwam-cpu-load" if c.fastwam_cpu_load else "--no-fastwam-cpu-load"]
        if c.stub:
            cmd.append("--stub")
        else:
            cmd += ["--ckpt", str(c.ckpt)]
        return cmd

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def boot(self) -> None:
        """Free the port, start the server, wait for a NEW 'serving on'
        line, then verify the regime with a ping."""
        free_port(self.cfg.port)
        n0 = _count_serving(self.log_path)
        cmd = self.command()
        log.info("booting server: %s", " ".join(cmd))
        with open(self.log_path, "ab") as lf:
            self.proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < self.cfg.server_boot_s:
            if _count_serving(self.log_path) > n0:
                break
            if not self.alive():
                raise RuntimeError(f"policy server exited during boot (see {self.log_path})")
            time.sleep(2.0)
        else:
            self.stop()
            raise RuntimeError(f"policy server did not report 'serving on' within "
                               f"{self.cfg.server_boot_s}s (see {self.log_path})")
        with PolicyClient(self.cfg.host, self.cfg.port, timeout=60) as cl:
            pong = cl.ping()
        problems = check_regime(pong, self.cfg)
        if problems:
            self.stop()
            raise RuntimeError("server regime mismatch: " + "; ".join(problems))
        log.info("server ready: %s", pong.get("regime", pong))

    def ensure(self) -> None:
        if self.proc is not None and not self.alive():
            log.warning("server died: rebooting")
        if not self.alive():
            self.boot()

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=20)
            except (OSError, subprocess.TimeoutExpired):
                _killpg(self.proc)
        self.proc = None


def check_regime(pong: dict, cfg: DriverConfig) -> list[str]:
    """Differences between the server's reported regime and the protocol."""
    r = pong.get("regime")
    if not isinstance(r, dict):
        return ["ping reply carries no regime (old server?)"]
    want: dict[str, Any] = {
        "plane_filter": bool(cfg.plane_filter),
        "drop_px": int(cfg.drop_px),
        "plane_top_cut": float(cfg.plane_top_cut),
        "eval_aug": False,
        "chunk": 50,
    }
    if cfg.stub:
        want["stub"] = True
    else:
        want["type"] = cfg.policy_type
        want["half"] = bool(cfg.half) and cfg.policy_type not in _OWN_PRECISION
    problems = [f"{k}: server {r.get(k)!r} != expected {v!r}"
                for k, v in want.items() if r.get(k) != v]
    if not cfg.stub:
        served = r.get("ckpt")
        try:
            same = served is not None and Path(served).resolve() == Path(cfg.ckpt).resolve()
        except OSError:
            same = False
        if not same:
            problems.append(f"ckpt: server {served!r} != expected {str(cfg.ckpt)!r}")
        if int(r.get("horizon", 50)) != cfg.exec_h and cfg.policy_type == "fastwam":
            problems.append(f"exec-h {cfg.exec_h} != fastwam horizon {r.get('horizon')}")
    return problems


def kit_command(cfg: DriverConfig, seed: int) -> list[str]:
    c = cfg
    cmd = [c.sim_python, "-m", "simretract.eval.sim_eval",
           "--seeds", str(seed), "--tag", c.name, "--out", str(c.out),
           "--tool", c.tool, "--host", c.host, "--port", str(c.port),
           "--exec-h", str(c.exec_h), "--smooth", str(c.smooth),
           "--max-steps", str(c.max_steps), "--tries", str(c.tries), "--mu", str(c.mu),
           "--hand-style", c.hand_style, "--plane-top-cut", str(c.plane_top_cut),
           "--skip-search" if c.skip_search else "--search"]
    if c.keep_frames:
        cmd.append("--keep-frames")
    if not c.review_mp4:
        cmd.append("--no-review-mp4")
    if not c.battery:
        cmd.append("--no-battery")
    return cmd


def parse_outcomes(log_path: Path) -> list[dict]:
    """``[M5] {json}`` rows written by the evaluator."""
    rows = []
    try:
        with open(log_path, errors="replace") as f:
            for line in f:
                m = _OUTCOME_RE.match(line)
                if m:
                    try:
                        rows.append(json.loads(m.group(1)))
                    except ValueError:
                        continue
    except OSError:
        pass
    return rows


def run_worker(cfg: DriverConfig, seed: int, log_path: Path) -> dict:
    """One Kit launch for ``seed``; returns a ledger entry."""
    cmd = kit_command(cfg, seed)
    env = dict(os.environ)  # OMNI_KIT_ACCEPT_EULA was verified (or set by --accept-eula) in run()
    t0 = time.time()
    with open(log_path, "wb") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=env, start_new_session=True)
    status = "exited"
    done_seen_at: float | None = None
    while True:
        rc = proc.poll()
        if rc is not None:
            break
        elapsed = time.time() - t0
        if elapsed > cfg.timeout_s:
            status = "timeout"
            _killpg(proc)
            break
        if done_seen_at is None:
            try:
                with open(log_path, "rb") as f:
                    f.seek(max(0, os.path.getsize(log_path) - 4096))
                    if _DONE_MARK.encode() in f.read():
                        done_seen_at = time.time()
            except OSError:
                pass
        elif time.time() - done_seen_at > 60:
            status = "reaped_after_done"   # finished but stuck in Kit teardown
            _killpg(proc)
            break
        time.sleep(2.0)
    rows = [r for r in parse_outcomes(log_path) if r.get("seed") == seed]
    outcome = rows[-1].get("outcome") if rows else None
    entry = {"seed": seed, "rc": proc.returncode, "status": status,
             "outcome": outcome, "seconds": round(time.time() - t0, 1),
             "log": str(log_path)}
    log.info("seed %d: %s rc=%s outcome=%s (%.0fs)", seed, status, proc.returncode,
             outcome, entry["seconds"])
    return entry


# --------------------------------------------------------------- the run
def _write_ledger(path: Path, ledger: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(ledger, f, indent=1)
    tmp.replace(path)


def run(cfg: DriverConfig) -> scoring.Funnel:
    """Execute all passes, score, write ``results.json`` + ``table.md``."""
    from simretract import config

    config.require_kit_eula()
    cfg.out.mkdir(parents=True, exist_ok=True)
    logs = cfg.out / "logs"
    logs.mkdir(exist_ok=True)
    run_seeds, _ = scoring.load_seeds(cfg.seeds_json)
    score_seeds, label = scoring.load_seeds(cfg.score_json)
    ledger_path = cfg.out / "ledger.json"
    ledger: dict[str, Any] = {"config": {k: str(v) if isinstance(v, Path) else v
                                         for k, v in asdict(cfg).items()},
                              "attempts": {}}
    if ledger_path.is_file():
        try:
            with open(ledger_path) as f:
                ledger["attempts"] = json.load(f).get("attempts", {})
        except (OSError, ValueError):
            pass
    server = Server(cfg, logs / "server.log")
    try:
        for pass_no in range(1, cfg.passes + 1):
            todo = select_seeds(cfg.out, run_seeds, pass_no)
            if not todo:
                log.info("pass %d: nothing to run", pass_no)
                continue
            log.info("pass %d: %d seed(s) on port %d: %s", pass_no, len(todo), cfg.port,
                     " ".join(map(str, todo)))
            server.ensure()
            for seed in todo:
                term = terminal_outcome(ledger["attempts"], seed)
                if term:
                    log.info("seed %d: %s on an earlier attempt; not re-run", seed, term)
                    continue
                server.ensure()
                for launch in range(1, cfg.launches + 1):
                    if result_path(cfg.out, seed).is_file():
                        break
                    entry = run_worker(cfg, seed, logs / f"worker_{seed}_p{pass_no}_l{launch}.log")
                    entry.update({"pass": pass_no, "launch": launch})
                    ledger["attempts"].setdefault(str(seed), []).append(entry)
                    _write_ledger(ledger_path, ledger)
                    if entry["outcome"] in TERMINAL_OUTCOMES:
                        break   # deterministic for this seed: no second launch
                    if not server.alive():
                        break   # the next seed reboots it; this launch's result decides the retry
            log.info("pass %d done", pass_no)
    finally:
        server.stop()

    f = scoring.funnel(scoring.load_results(cfg.out, attempt=cfg.score_attempt), score_seeds)
    line = scoring.row_line(cfg.name, label, f, cfg.exec_h, cfg.passes, attempt=cfg.score_attempt)
    print(line, flush=True)
    with open(cfg.out / "results.json", "w") as fh:
        json.dump({"name": cfg.name, "label": label, "exec_h": cfg.exec_h,
                   "passes": cfg.passes, "score_attempt": cfg.score_attempt,
                   "statistics": {"population": "secured", "sd_ddof": 0}, "protocol": {
                       "smooth": cfg.smooth, "skip_search": cfg.skip_search,
                       "max_steps": cfg.max_steps, "half": cfg.half,
                       "plane_filter": cfg.plane_filter, "drop_px": cfg.drop_px,
                       "plane_top_cut": cfg.plane_top_cut, "hand_style": cfg.hand_style,
                       "tries": cfg.tries, "mu": cfg.mu, "tool": cfg.tool},
                   "row": line, "funnel": f.to_dict()}, fh, indent=1)
    with open(cfg.out / "table.md", "w") as fh:
        fh.write(scoring.table_md(cfg.name, label, f, cfg.exec_h, attempt=cfg.score_attempt))
        fh.write("\n" + line + "\n")
    return f


# -------------------------------------------------------------------- CLI
def build_parser(proto: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    proto = proto or load_protocol()
    p = argparse.ArgumentParser(
        prog="simretract-eval-sim",
        description="Run the simulation benchmark for one checkpoint: boots the "
                    "policy server, evaluates every seed in a fresh Isaac Sim "
                    "process (with optional retries) and scores the "
                    "funnel. Defaults are the benchmark protocol.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", default="", help="checkpoint directory (flat or with final/)")
    p.add_argument("--stub", action="store_true",
                   help="use the hold-pose stub server instead of a checkpoint (smoke test)")
    p.add_argument("--name", default="", help="run name (default: checkpoint directory name)")
    p.add_argument("--out", default="", help="output directory (default: <paths.eval>/<name>)")
    p.add_argument("--seeds", default="",
                   help=f"selection json of seeds to RUN (default: data/selections/{proto['seeds']})")
    p.add_argument("--score-seeds", default="",
                   help="selection json the ROW/table are scored over (default: --seeds)")
    p.add_argument("--passes", type=int, default=int(proto["passes"]),
                   help="passes to run; later passes retry non-completed seeds")
    p.add_argument("--score-attempt", choices=("first", "latest"), default=proto["score_attempt"],
                   help="result selection for reporting (first is the paper protocol)")
    p.add_argument("--launches", type=int, default=int(proto["launches"]),
                   help="Kit launches per seed per pass (crash protection)")
    p.add_argument("--timeout", type=int, default=int(proto["timeout_s"]),
                   help="wall-clock seconds per Kit launch")
    p.add_argument("--server-boot-timeout", type=int, default=int(proto["server_boot_s"]),
                   help="seconds to wait for the server's 'serving on' line")
    p.add_argument("--host", default=str(proto["host"]))
    p.add_argument("--port", type=int, default=int(proto["port"]), help="policy server port")
    p.add_argument("--exec-h", default="auto",
                   help="actions executed per chunk; auto = 50 (48 for fastwam)")
    p.add_argument("--smooth", type=float, default=float(proto["smooth"]), help="EMA on targets")
    p.add_argument("--max-steps", type=int, default=int(proto["max_steps"]),
                   help="control ticks before timeout")
    p.add_argument("--search", dest="skip_search", action="store_false",
                   default=bool(proto["skip_search"]),
                   help="run the scripted grasp search (protocol skips it)")
    p.add_argument("--tries", type=int, default=int(proto["tries"]))
    p.add_argument("--mu", type=float, default=float(proto["mu"]))
    p.add_argument("--hand-style", default=str(proto["hand_style"]), choices=["test", "train"],
                   help="operator-hand occluder distribution (protocol: test)")
    p.add_argument("--plane-top-cut", type=float, default=float(proto["plane_top_cut"]),
                   help="plane top-cut sent per query (protocol: 0)")
    p.add_argument("--no-plane-filter", dest="plane_filter", action="store_false",
                   default=bool(proto["plane_filter"]))
    p.add_argument("--drop-px", type=int, default=int(proto["drop_px"]))
    p.add_argument("--no-half", dest="half", action="store_false", default=bool(proto["half"]),
                   help="serve fp32 weights (protocol: bf16)")
    p.add_argument("--tool", default=str(proto["tool"]))
    p.add_argument("--keep-frames", action="store_true", default=bool(proto["keep_frames"]),
                   help="keep per-frame PNGs")
    p.add_argument("--no-review-mp4", dest="review_mp4", action="store_false", default=True)
    p.add_argument("--no-battery", dest="battery", action="store_false",
                   default=not bool(proto["no_battery"]),
                   help="gravity-only tier (protocol runs the battery)")
    p.add_argument("--policy-python", default="", help="override the policy interpreter")
    p.add_argument("--sim-python", default="", help="override the Isaac Sim interpreter")
    p.add_argument("--accept-eula", action="store_true",
                   help="set OMNI_KIT_ACCEPT_EULA=YES for the workers")
    return p


def config_from_args(a: argparse.Namespace, proto: dict[str, Any] | None = None) -> DriverConfig:
    from simretract import config

    proto = proto or load_protocol()
    if a.accept_eula:
        os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    if a.stub:
        ckpt = Path(a.ckpt) if a.ckpt else Path("stub")
        ptype = "stub"
        name = a.name or "stub"
    else:
        if not a.ckpt:
            build_parser(proto).error("--ckpt is required unless --stub is given")
        ckpt = resolve_checkpoint_dir(a.ckpt)
        ptype = checkpoint_type(ckpt)
        name = a.name or (ckpt.parent.name if ckpt.name == "final" else ckpt.name)
    seeds_json = Path(a.seeds) if a.seeds else config.selection_path(str(proto["seeds"]))
    if a.score_seeds:
        score_json = Path(a.score_seeds)
    elif a.seeds:
        score_json = seeds_json
    else:
        score_json = config.selection_path(str(proto["score_seeds"]))
    out = Path(a.out) if a.out else Path(config.paths()["eval"]) / name
    if a.exec_h == "auto":
        exec_h = int(proto["exec_h"].get(ptype, 50))
    else:
        exec_h = int(a.exec_h)
    role = "fastwam" if ptype == "fastwam" else "policy"
    policy_python = a.policy_python or (sys.executable if a.stub else str(config.interpreter(role)))
    sim_python = a.sim_python or str(config.interpreter("sim"))
    return DriverConfig(
        ckpt=ckpt, out=out, seeds_json=seeds_json, score_json=score_json, name=name,
        passes=a.passes, score_attempt=a.score_attempt, launches=a.launches, timeout_s=a.timeout,
        server_boot_s=a.server_boot_timeout, host=a.host, port=a.port, exec_h=exec_h,
        smooth=a.smooth, skip_search=a.skip_search, max_steps=a.max_steps, tries=a.tries,
        mu=a.mu, hand_style=a.hand_style, plane_top_cut=a.plane_top_cut,
        plane_filter=a.plane_filter, drop_px=a.drop_px, half=a.half,
        fastwam_cpu_load=bool(proto["fastwam_cpu_load"]), tool=a.tool,
        keep_frames=a.keep_frames, review_mp4=a.review_mp4, battery=a.battery,
        stub=a.stub, policy_python=policy_python, sim_python=sim_python, policy_type=ptype)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    proto = load_protocol()
    args = build_parser(proto).parse_args(argv)
    cfg = config_from_args(args, proto)
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
