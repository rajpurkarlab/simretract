"""Campaign driver: run recorder chunks in parallel Isaac Sim processes.

Kit is expensive to boot and occasionally wedges, so the campaign splits
the work into chunks of a few seeds (or virtual indices), runs each chunk
in a fresh simulator process (``<sim python> -m simretract.sim.recorder``
or ``recovery_recorder``), and keeps ``parallel`` workers busy. Workers are
spawned in their own session so a timeout can kill the whole process
group; a worker whose log already shows the ``[M3] DONE`` line but whose
process lingers in Kit teardown is reaped. Per-item outcomes are parsed
from the ``[M3] {json}`` lines into a JSON ledger; an item is retried until
it has a done marker (``meta.npz`` / ``.filtered`` / ``.wedged``) or its
launch count reaches ``tries``. Re-running the same command resumes.

Quota mode (strict sets) targets ``per_bin`` kept episodes per grasp-height
bin: fresh seeds are pre-screened without the simulator (a seed whose hand
occluder blocks the target bin entirely is skipped, and a soft cap keeps the
tilt distribution balanced), then recorded with the bin as a hard height
constraint until every bin is full.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from simretract.sim import assemble
from simretract.sim.recorder import DONE_LINE, OUTCOME_TAG, RecorderConfig, episode_done
from simretract.sim.recovery_recorder import episode_name, virtual_to_episode
from simretract.sim.sampling import (bin_fully_blocked, hand_blocked_interval,
                                     sample_occluder_spec, sample_presented_pose_arrays)
from simretract.sim.tool_usd import tip_local

log = logging.getLogger(__name__)

LEDGER_NAME = "campaign_ledger.json"
TILT_EDGES = (0, 20, 40, 60, 90)
TILT_CAP = 0.35          # soft cap on the share of any tilt bin
RELBAL = 12              # slack (episodes) allowed above the cap while balancing
HAND_GATE_M = 0.010      # height picker hand keep-out used by the pre-screen


def sim_python() -> str:
    """Interpreter of the simulator environment (``simretract.config``)."""
    from simretract import config

    return str(config.interpreter("sim"))


@dataclasses.dataclass
class CampaignConfig:
    """Process-lifecycle knobs of a campaign."""
    out: str
    parallel: int = 3
    stagger_s: float = 15.0
    chunk: int = 5
    timeout_s: float = 4800.0
    tries: int = 2
    poll_s: float = 10.0
    done_grace_s: float = 30.0
    python: str | None = None      # simulator interpreter (default: config)
    env: dict[str, str] = dataclasses.field(default_factory=dict)

    @classmethod
    def add_args(cls, ap, defaults: dict[str, Any] | None = None) -> None:
        """Add the campaign flags. ``defaults`` (the ``campaign:`` block of a
        generation recipe: parallel, stagger_s, chunk, timeout_s, tries)
        overrides the dataclass defaults."""
        d = {"parallel": cls.parallel, "stagger_s": cls.stagger_s, "chunk": cls.chunk,
             "timeout_s": cls.timeout_s, "tries": cls.tries, "poll_s": cls.poll_s}
        d.update({k: v for k, v in (defaults or {}).items() if k in d})
        g = ap.add_argument_group("campaign (parallel simulator workers)")
        g.add_argument("--parallel", type=int, default=int(d["parallel"]),
                       help="simultaneous simulator processes")
        g.add_argument("--stagger", type=float, default=float(d["stagger_s"]),
                       help="seconds between worker boots")
        g.add_argument("--chunk", type=int, default=int(d["chunk"]), help="items per simulator process")
        g.add_argument("--timeout", type=float, default=float(d["timeout_s"]),
                       help="seconds before a chunk is killed")
        g.add_argument("--tries", type=int, default=int(d["tries"]),
                       help="launches per item before giving up")
        g.add_argument("--poll", type=float, default=float(d["poll_s"]),
                       help="seconds between worker polls")
        g.add_argument("--sim-python", default=None,
                       help="simulator interpreter (default: the sim_python recorded in "
                            "configs/envs.local.yaml by envs/setup.sh)")

    @classmethod
    def from_args(cls, args, out: str, env: dict[str, str] | None = None) -> "CampaignConfig":
        return cls(out=out, parallel=args.parallel, stagger_s=args.stagger, chunk=args.chunk,
                   timeout_s=args.timeout, tries=args.tries, poll_s=args.poll,
                   python=args.sim_python, env=dict(env or {}))


# ----------------------------------------------------------------- outcome parsing
_OUTCOME_RE = re.compile(re.escape(OUTCOME_TAG) + r" (\{.*\})\s*$")


def parse_outcomes(text: str) -> list[dict[str, Any]]:
    """All ``[M3] {json}`` rows in a worker log, in order."""
    rows = []
    for line in text.splitlines():
        m = _OUTCOME_RE.match(line.strip())
        if not m:
            continue
        try:
            rows.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            continue
    return rows


def log_is_done(path: str | os.PathLike) -> bool:
    """True when the worker log contains the DONE line (as a whole line)."""
    try:
        with open(path, errors="replace") as f:
            return any(line.strip() == DONE_LINE for line in f)
    except OSError:
        return False


def chunked(items: list, n: int) -> list[list]:
    return [items[i:i + n] for i in range(0, len(items), max(1, n))]


# ----------------------------------------------------------------- ledger
class Ledger:
    """Per-item launch counts and last outcomes, persisted as JSON."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.data: dict[str, Any] = {"items": {}, "chunks": 0, "meta": {}}
        if self.path.is_file():
            with open(self.path) as f:
                self.data.update(json.load(f))

    def item(self, key: str) -> dict[str, Any]:
        return self.data["items"].setdefault(str(key), {"attempts": 0, "outcome": None})

    def attempts(self, key: str) -> int:
        return int(self.item(key)["attempts"])

    def launched(self, keys: Iterable[str]) -> None:
        self.data["chunks"] += 1
        for k in keys:
            self.item(k)["attempts"] += 1

    def record(self, key: str, outcome: str) -> None:
        self.item(key)["outcome"] = outcome

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=1)
        os.replace(tmp, self.path)


# ----------------------------------------------------------------- workers
@dataclasses.dataclass
class Job:
    keys: list[str]                 # ledger keys of the items in this chunk
    cmd: list[str]
    tag: str                        # log-file stem
    is_done: Callable[[str], bool]  # key -> done marker present
    key_of_row: Callable[[dict], str | None]


@dataclasses.dataclass
class Worker:
    job: Job
    proc: subprocess.Popen
    log_path: Path
    started: float
    done_seen: float | None = None


class WorkerPool:
    """Spawns simulator processes and reaps them on exit, timeout or a
    lingering teardown after the DONE line."""

    def __init__(self, cfg: CampaignConfig):
        self.cfg = cfg
        self.python = cfg.python or sim_python()
        self.logs = Path(cfg.out) / ".logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.active: list[Worker] = []
        self._last_spawn = 0.0

    def can_spawn(self) -> bool:
        return (len(self.active) < self.cfg.parallel
                and time.time() - self._last_spawn >= self.cfg.stagger_s)

    def spawn(self, job: Job) -> Worker:
        log_path = self.logs / f"{job.tag}.log"
        env = dict(os.environ)
        env.update(self.cfg.env)
        cmd = [self.python] + resolve_cmd(job)
        with open(log_path, "a") as lf:
            lf.write(f"# campaign job {job.tag}: {len(job.keys)} items "
                     f"({job.keys[0]}..{job.keys[-1]})\n")
        lf = open(log_path, "a")
        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                start_new_session=True)
        lf.close()
        w = Worker(job, proc, log_path, time.time())
        self.active.append(w)
        self._last_spawn = time.time()
        log.info("spawned %s (pid %d, %d items)", job.tag, proc.pid, len(job.keys))
        return w

    @staticmethod
    def kill(w: Worker) -> None:
        try:
            os.killpg(w.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            w.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            log.warning("%s did not exit after SIGKILL", w.job.tag)

    def reap(self) -> list[tuple[Worker, str]]:
        """Finished workers as (worker, reason) with reason in
        {exit, reaped, timeout}."""
        finished, still = [], []
        now = time.time()
        for w in self.active:
            if w.proc.poll() is not None:
                finished.append((w, "exit"))
                continue
            if w.done_seen is None and log_is_done(w.log_path):
                w.done_seen = now
            if w.done_seen is not None and now - w.done_seen >= self.cfg.done_grace_s:
                self.kill(w)
                finished.append((w, "reaped"))
            elif now - w.started > self.cfg.timeout_s:
                self.kill(w)
                finished.append((w, "timeout"))
            else:
                still.append(w)
        self.active = still
        return finished

    def kill_all(self) -> None:
        for w in self.active:
            self.kill(w)
        self.active = []


# ----------------------------------------------------------------- generic runner
def run_jobs(cfg: CampaignConfig, ledger: Ledger, pending: list[Job],
             on_finished: Callable[[Worker, str], None] | None = None) -> dict[str, Any]:
    """Run ``pending`` jobs; items without a done marker are re-queued
    until their attempts reach ``cfg.tries``. Returns a summary."""
    pool = WorkerPool(cfg)
    queue = collections.deque(pending)
    t0 = time.time()
    n_launched = 0
    try:
        while queue or pool.active:
            for w, reason in pool.reap():
                job = w.job
                try:
                    rows = parse_outcomes(w.log_path.read_text(errors="replace"))
                except OSError:
                    rows = []
                for row in rows:
                    key = job.key_of_row(row)
                    if key is not None:
                        ledger.record(key, str(row.get("outcome")))
                left = [k for k in job.keys if not job.is_done(k) and ledger.attempts(k) < cfg.tries]
                gave_up = [k for k in job.keys if not job.is_done(k) and ledger.attempts(k) >= cfg.tries]
                log.info("%s finished (%s, rc=%s): %d outcomes, %d to retry, %d given up",
                         job.tag, reason, w.proc.returncode, len(rows), len(left), len(gave_up))
                if left:
                    queue.appendleft(dataclasses.replace(
                        job, keys=left, tag=f"{job.tag}_r{ledger.attempts(left[0])}"))
                if on_finished:
                    on_finished(w, reason)
                ledger.save()
            while queue and pool.can_spawn():
                job = queue.popleft()
                ledger.launched(job.keys)
                pool.spawn(job)
                n_launched += 1
                ledger.save()
            if queue and not pool.can_spawn() and len(pool.active) < cfg.parallel:
                time.sleep(max(0.5, min(cfg.poll_s, cfg.stagger_s)))
                continue
            if pool.active:
                time.sleep(cfg.poll_s)
    except KeyboardInterrupt:
        log.warning("interrupted: killing %d workers", len(pool.active))
        pool.kill_all()
        ledger.save()
        raise
    ledger.save()
    return {"chunks_launched": n_launched, "elapsed_s": round(time.time() - t0)}


def resolve_cmd(job: Job) -> list[str]:
    """The job command with the ``{items}`` placeholder replaced by the keys
    still pending (a re-queued job carries fewer keys than its first launch)."""
    return [c.replace("{items}", ",".join(job.keys)) if "{items}" in c else c for c in job.cmd]


# ----------------------------------------------------------------- strict campaigns
def strict_job(out: str, keys: list[str], recipe_args: list[str], stored_winners: str | None,
               tag: str) -> Job:
    cmd = ["-m", "simretract.sim.recorder", "--out", out, "--seeds", "{items}"] + recipe_args
    if stored_winners:
        cmd += ["--stored-winners", stored_winners]

    def is_done(k: str) -> bool:
        return episode_done(os.path.join(out, f"ep_{k}"))

    def key_of_row(row: dict) -> str | None:
        return None if "seed" not in row else str(int(row["seed"]))
    return Job(keys=[str(k) for k in keys], cmd=cmd, tag=tag, is_done=is_done,
               key_of_row=key_of_row)


def run_strict_list(cfg: CampaignConfig, rcfg: RecorderConfig, seeds: list[int],
                    stored_winners: str | None) -> dict[str, Any]:
    """Record an explicit seed list with the recipe ``rcfg``."""
    ledger = Ledger(Path(cfg.out) / LEDGER_NAME)
    ledger.data["meta"] = {"mode": "strict", "recipe": rcfg.to_dict()}
    todo = [s for s in seeds if not episode_done(os.path.join(cfg.out, f"ep_{s}"))
            and ledger.attempts(str(s)) < cfg.tries]
    log.info("strict list: %d seeds, %d to record", len(seeds), len(todo))
    jobs = [strict_job(cfg.out, ch, rcfg.to_cli_args(), stored_winners, f"strict_{ch[0]}")
            for ch in chunked(todo, cfg.chunk)]
    summary = run_jobs(cfg, ledger, jobs)
    summary["recorded"] = sum(1 for s in seeds
                              if os.path.isfile(os.path.join(cfg.out, f"ep_{s}", "meta.npz")))
    return summary


def bin_counts(out: str, z_lo: float, z_hi: float, bins: int) -> tuple[list[int], list[int]]:
    """Kept episodes per grasp-height bin and per tilt bin from disk."""
    per_bin = [0] * bins
    tilt = [0] * (len(TILT_EDGES) - 1)
    for r in assemble.scan_strict_raw(out):
        b = assemble.bin_index(r["z_m"], z_lo, z_hi, bins)
        if b is None:
            continue
        per_bin[b] += 1
        try:
            t = float(np.load(os.path.join(out, r["name"], "meta.npz"))["tilt_deg"])
        except (OSError, KeyError, ValueError):
            continue
        tilt[tilt_bin(t)] += 1
    return per_bin, tilt


def tilt_bin(t: float) -> int:
    for i in range(len(TILT_EDGES) - 1):
        if t < TILT_EDGES[i + 1]:
            return i
    return len(TILT_EDGES) - 2


def prescreen_seed(seed: int, tip, hand_style: str, z_lo: float, z_hi: float,
                   tilt_counts: list[int], target_total: int) -> tuple[bool, str]:
    """Cheap, simulator-free feasibility check of ``seed`` for the bin
    ``[z_lo, z_hi]``. Returns (accept, reason)."""
    rng = np.random.default_rng(seed)
    _pos, theta, _phi, _psi = sample_presented_pose_arrays(rng)
    spec = sample_occluder_spec(seed, tip, hand_style)
    if bin_fully_blocked(hand_blocked_interval(spec, HAND_GATE_M), z_lo, z_hi):
        return False, "hand_blocks_bin"
    tb = tilt_bin(float(np.degrees(theta)))
    total_now = max(sum(tilt_counts), 1)
    if (tilt_counts[tb] >= TILT_CAP * target_total
            or tilt_counts[tb] > TILT_CAP * total_now + RELBAL):
        return False, "tilt_cap"
    return True, "ok"


def run_strict_quota(cfg: CampaignConfig, rcfg: RecorderConfig, seed_start: int, quota: int,
                     bins: int, z_lo: float, z_hi: float, max_seeds: int = 0) -> dict[str, Any]:
    """Fill every grasp-height bin with ``quota`` kept episodes using fresh,
    pre-screened seeds and bin-targeted recording (hard height constraint).
    Stops when all bins are full or ``max_seeds`` fresh seeds were used."""
    ledger = Ledger(Path(cfg.out) / LEDGER_NAME)
    ledger.data["meta"] = {"mode": "strict_quota", "recipe": rcfg.to_dict(), "quota": quota,
                           "bins": bins, "z_range": [z_lo, z_hi]}
    next_seed = max(int(ledger.data.get("next_seed", seed_start)), seed_start)
    tip = tip_local(rcfg.tool)
    edges = assemble.bin_edges(z_lo, z_hi, bins)
    target_total = quota * bins
    pool = WorkerPool(cfg)
    skipped = collections.Counter()
    t0 = time.time()
    launched = 0

    def deficits() -> list[int]:
        per_bin, _tilt = bin_counts(cfg.out, z_lo, z_hi, bins)
        return [max(0, quota - c) for c in per_bin]

    def next_chunk(b: int, tilt_counts: list[int]) -> list[int]:
        nonlocal next_seed
        chunk: list[int] = []
        while len(chunk) < cfg.chunk:
            if max_seeds and next_seed - seed_start >= max_seeds:
                break
            s = next_seed
            next_seed += 1
            ok, why = prescreen_seed(s, tip, rcfg.hand_style, edges[b], edges[b + 1],
                                     tilt_counts, target_total)
            if ok:
                chunk.append(s)
            else:
                skipped[why] += 1
        return chunk

    try:
        while True:
            for w, reason in pool.reap():
                rows = parse_outcomes(w.log_path.read_text(errors="replace"))
                for row in rows:
                    if "seed" in row:
                        ledger.record(str(int(row["seed"])), str(row.get("outcome")))
                log.info("%s finished (%s): %d outcomes", w.job.tag, reason, len(rows))
            per_bin, tilt_counts = bin_counts(cfg.out, z_lo, z_hi, bins)
            need = [max(0, quota - c) for c in per_bin]
            if not any(need):
                pool.kill_all()
                break
            exhausted = bool(max_seeds and next_seed - seed_start >= max_seeds)
            if exhausted and not pool.active:
                log.warning("seed budget exhausted with deficits %s", need)
                break
            while pool.can_spawn() and any(need) and not exhausted:
                # serve the emptiest bin first
                b = int(np.argmax(need))
                chunk = next_chunk(b, tilt_counts)
                if not chunk:
                    exhausted = True
                    break
                args = rcfg.to_cli_args() + ["--z-lo", str(edges[b]), "--z-hi", str(edges[b + 1]),
                                             "--z-strict"]
                job = strict_job(cfg.out, chunk, args, None, f"quota_b{b}_{chunk[0]}")
                ledger.launched(job.keys)
                pool.spawn(job)
                launched += 1
                need[b] = max(0, need[b] - len(chunk))
                ledger.data["next_seed"] = next_seed
                ledger.data["bins"] = per_bin
                ledger.data["skipped"] = dict(skipped)
                ledger.save()
                log.info("bins %s / %d each, tilt %s, skipped %s", per_bin, quota, tilt_counts,
                         dict(skipped))
            ledger.data["next_seed"] = next_seed
            ledger.save()
            time.sleep(cfg.poll_s)
    except KeyboardInterrupt:
        pool.kill_all()
        ledger.data["next_seed"] = next_seed
        ledger.save()
        raise
    per_bin, tilt_counts = bin_counts(cfg.out, z_lo, z_hi, bins)
    ledger.data.update({"next_seed": next_seed, "bins": per_bin, "done": not any(
        quota - c > 0 for c in per_bin)})
    ledger.save()
    return {"bins": per_bin, "tilt_bins": tilt_counts, "next_seed": next_seed,
            "chunks_launched": launched, "skipped": dict(skipped),
            "elapsed_s": round(time.time() - t0)}


# ----------------------------------------------------------------- recovery campaigns
def recovery_job(out: str, keys: list[str], recipe_args: list[str], strict_json: str,
                 winners: str, draws: int, seeds_sorted: list[int], tag: str) -> Job:
    cmd = ["-m", "simretract.sim.recovery_recorder", "--out", out, "--strict-json", strict_json,
           "--winners", winners, "--draws", str(draws), "--virt", "{items}"] + recipe_args

    def is_done(k: str) -> bool:
        seed, draw = virtual_to_episode(int(k), seeds_sorted, draws)
        return episode_done(os.path.join(out, episode_name(seed, draw)))

    def key_of_row(row: dict) -> str | None:
        if "virt" in row:
            return str(int(row["virt"]))
        if "seed" in row and "draw" in row:
            try:
                return str(seeds_sorted.index(int(row["seed"])) * draws + int(row["draw"]))
            except ValueError:
                return None
        return None
    return Job(keys=[str(k) for k in keys], cmd=cmd, tag=tag, is_done=is_done,
               key_of_row=key_of_row)


def run_recovery_list(cfg: CampaignConfig, rcfg: RecorderConfig, virts: list[int],
                      strict_json: str, winners: str, draws: int) -> dict[str, Any]:
    """Record an explicit list of virtual indices."""
    with open(strict_json) as f:
        seeds_sorted = sorted(int(s) for s in json.load(f)["seeds"])
    ledger = Ledger(Path(cfg.out) / LEDGER_NAME)
    ledger.data["meta"] = {"mode": "recovery", "recipe": rcfg.to_dict(), "draws": draws}
    probe = recovery_job(cfg.out, [], [], strict_json, winners, draws, seeds_sorted, "probe")
    todo = [v for v in virts if not probe.is_done(str(v)) and ledger.attempts(str(v)) < cfg.tries]
    log.info("recovery list: %d indices, %d to record", len(virts), len(todo))
    jobs = [recovery_job(cfg.out, ch, rcfg.to_cli_args(), strict_json, winners, draws,
                         seeds_sorted, f"rec_{ch[0]}") for ch in chunked(todo, cfg.chunk)]
    summary = run_jobs(cfg, ledger, jobs)
    summary["keeps"] = sum(len(v) for v in assemble.scan_recovery_raw(cfg.out).values())
    return summary


def plan_topup(out: str, seeds_sorted: list[int], draws: int, target_keeps: int,
               yield_rate: float = 0.3) -> tuple[list[int], dict[str, int]]:
    """Virtual indices of fresh draws for seeds short of ``target_keeps``.

    Untried draws (no directory or marker) are over-provisioned by
    ``need / yield_rate`` (min ``need + 1``) because top-up yield on the hard
    seeds is low; surplus keeps are trimmed at assembly.
    """
    keeps: collections.Counter = collections.Counter()
    tried: dict[int, set] = collections.defaultdict(set)
    pat = re.compile(r"^ep_(\d+)_p(\d+)(\.filtered|\.wedged)?$")
    for d in os.listdir(out) if os.path.isdir(out) else []:
        m = pat.match(d)
        if not m:
            continue
        s, p = int(m.group(1)), int(m.group(2))
        tried[s].add(p)
        if not m.group(3) and os.path.isfile(os.path.join(out, d, "meta.npz")):
            keeps[s] += 1
    plan: list[int] = []
    deficit = 0
    for i, s in enumerate(seeds_sorted):
        need = target_keeps - keeps[s]
        if need <= 0:
            continue
        deficit += need
        fresh = [p for p in range(draws) if p not in tried[s]]
        fresh = fresh[:max(need + 1, int(round(need / yield_rate)))]
        plan += [i * draws + p for p in fresh]
    stats = {"deficit_seeds": sum(1 for s in seeds_sorted if keeps[s] < target_keeps),
             "missing_keeps": deficit, "draws_planned": len(plan),
             "surplus_seeds": sum(1 for s in seeds_sorted if keeps[s] > target_keeps)}
    return plan, stats


def run_recovery_topup(cfg: CampaignConfig, rcfg: RecorderConfig, strict_json: str, winners: str,
                       draws: int, target_keeps: int, max_rounds: int = 6) -> dict[str, Any]:
    """Rounds of planned top-up draws until every seed has ``target_keeps``
    keeps or no untried draw is left."""
    with open(strict_json) as f:
        seeds_sorted = sorted(int(s) for s in json.load(f)["seeds"])
    rounds = []
    for r in range(max_rounds):
        plan, stats = plan_topup(cfg.out, seeds_sorted, draws, target_keeps)
        log.info("top-up round %d: %s", r + 1, stats)
        rounds.append(stats)
        if not plan:
            break
        run_recovery_list(cfg, rcfg, plan, strict_json, winners, draws)
    _plan, final = plan_topup(cfg.out, seeds_sorted, draws, target_keeps)
    return {"rounds": rounds, "final": final}
