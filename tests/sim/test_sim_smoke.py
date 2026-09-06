"""Simulation smoke tier: assets -> generation -> cache -> benchmark.

Runs in the policy environment; every step is a CLI invocation that
delegates Kit work to the ``sim`` interpreter recorded by ``envs/setup.sh``
(``simretract-build-assets``, ``simretract-generate``,
``simretract-build-cache``, ``simretract-eval-sim``).  The steps build on
each other in file order; a failed step fails the ones after it with a
message naming the prerequisite.  Outputs live under the shared smoke work
directory (``SIMRETRACT_SMOKE_DIR``, default ``<repo>/smoke_tmp``).

The tool asset is third-party and never part of the repository: the first
step runs the default asset build, ``simretract-build-assets --tool
retractor_rake``, which converts the source file the recipe
(``configs/assets/retractor_rake.yaml``) names under ``assets/source/`` (or
the ``tool_source`` override) into the tool meshes, checks them against the
tracked ``meta.json`` and the recipe fingerprints, builds the USD and
verifies the tracked grasp candidates.  Without a source file the build
step is skipped with a message, and the later steps run only if the assets
were built earlier on this machine (``docs/ASSETS.md``).

Recorder outcomes are not deterministic (PhysX + GPU rendering): the strict
step records candidate seeds two at a time until one is kept, and a
``.filtered`` recovery draw is a valid recorder outcome (the recovery cache
step is then skipped with a message).  A Kit worker occasionally stalls
mid-episode; the recovery lane is killed after 10 minutes and relaunched
once, and the benchmark driver's second launch per seed (with a 7-minute
wall cap here) absorbs a stalled evaluator.  Typical wall time is about 10
minutes, up to about 25 with a retried worker.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from simretract.cli import smoke

pytestmark = pytest.mark.sim

#: Strict candidates, recorded in pairs until one episode is kept.
STRICT_CANDIDATES = (200350, 400851, 200037, 200066)
#: Benchmark seed evaluated with the smoke checkpoints (a core-50 seed:
#: scene feasibility under --skip-search is established).
EVAL_SEED = 200037
#: The reference tool (assets/tools/<TOOL>/meta.json is tracked; its meshes are built locally).
TOOL = "retractor_rake"


@pytest.fixture(scope="module")
def work() -> Path:
    w = smoke.work_dir() / "sim"
    for sub in ("strict", "recovery", "cache", "eval_act", "eval_stub"):
        smoke.fresh_dir(w / sub)
    (w / "logs").mkdir(exist_ok=True)
    return w


@pytest.fixture(scope="module")
def state() -> dict:
    return {}


def _need(state: dict, key: str) -> object:
    if key not in state:
        pytest.fail(f"prerequisite step '{key}' did not run or failed")
    return state[key]


def _ok_env() -> dict[str, str]:
    return dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES")


def test_interpreters_configured() -> None:
    from simretract import config

    assert config.interpreter("sim").is_file()
    assert config.interpreter("policy").is_file()


NO_SOURCE = ("no tool source: drop the asset pack under assets/source/<pack>/ as the recipe "
             "configs/assets/retractor_rake.yaml names it, or set tool_source in configs/envs.local.yaml "
             "(or SIMRETRACT_TOOL_SOURCE); docs/ASSETS.md")


def _tool_source() -> Path | None:
    from simretract.sim import asset_recipe

    return asset_recipe.resolve_source(TOOL)[0]


def _require_tool_assets() -> None:
    """Skip (not fail) when the tool assets were never built on this machine."""
    from simretract.sim.tool_usd import require_tool_assets, require_tool_usd

    try:
        require_tool_assets(TOOL)
        require_tool_usd(TOOL)
    except FileNotFoundError as exc:
        pytest.skip(f"{exc}; the build step is skipped without a tool_source ({NO_SOURCE})")


def test_build_assets_from_source(work: Path) -> None:
    """The default build: the recipe's source file -> tool meshes (checked
    against meta.json and the recipe fingerprints) -> tool USD (checked
    against meta.json) -> grasp candidates compared with the tracked JSON,
    built in place under assets/ (git-ignored). Earlier build outputs are
    removed first so the check lines below can only come from a fresh
    conversion of the source."""
    if _tool_source() is None:
        pytest.skip(NO_SOURCE)
    import shutil

    from simretract.sim.tool_usd import tool_dir, tool_usd_path

    (tool_dir(TOOL) / "visual.obj").unlink(missing_ok=True)
    shutil.rmtree(tool_dir(TOOL) / "collision", ignore_errors=True)
    tool_usd_path(TOOL).unlink(missing_ok=True)
    log = work / "logs" / "build_assets.log"
    log.unlink(missing_ok=True)
    rc = smoke.run_logged([sys.executable, "-m", "simretract.cli.build_assets", "--tool", TOOL,
                           "--accept-eula"],
                          log, env=_ok_env(), timeout=900)
    text = log.read_text(errors="replace")
    assert rc == 0, smoke.tail(log)
    assert any(ln.startswith("[SOURCE]") and "recipe source" in ln for ln in text.splitlines()), smoke.tail(log)
    for tag in ("[MESH] check vs", "[VERIFY] ", "[BUILD] check vs", "[GRASPS] compare vs"):
        lines = [ln for ln in text.splitlines() if ln.startswith(tag) and " vs " in ln]
        assert lines and ": OK" in lines[-1], f"{tag}: {lines or smoke.tail(log)}"
    assert "MISMATCH" not in text, smoke.tail(log)
    from simretract.sim.tool_usd import require_tool_assets, require_tool_usd

    require_tool_assets(TOOL)
    require_tool_usd(TOOL)


def test_verify_and_grasp_candidates_reproduce(work: Path) -> None:
    """The standalone checks: the built visual.obj against the recipe
    fingerprints (no simulator needed) and the tracked grasp-candidate
    annotations re-sampled from the locally built mesh (the released data
    contract)."""
    _require_tool_assets()
    log = work / "logs" / "build_assets.log"
    rc = smoke.run_logged([sys.executable, "-m", "simretract.cli.build_assets", "--tool", TOOL,
                           "--verify"], log, env=_ok_env(), timeout=300)
    text = log.read_text(errors="replace")
    assert rc == 0 and "[VERIFY] MISMATCH" not in text, smoke.tail(log)
    rc = smoke.run_logged([sys.executable, "-m", "simretract.cli.build_assets", "--tool", TOOL,
                           "--grasps", "--compare", "--accept-eula"],
                          log, env=_ok_env(), timeout=900)
    text = log.read_text(errors="replace")
    assert rc == 0, smoke.tail(log)
    assert "[GRASPS] compare vs" in text and "[GRASPS] MISMATCH" not in text, smoke.tail(log)


def test_generate_strict(work: Path, state: dict) -> None:
    _require_tool_assets()
    out = work / "strict"
    log = work / "logs" / "generate_strict.log"
    log.unlink(missing_ok=True)
    kept = None
    for i in range(0, len(STRICT_CANDIDATES), 2):
        pair = STRICT_CANDIDATES[i:i + 2]
        rc = smoke.run_logged([sys.executable, "-m", "simretract.cli.generate", "strict",
                               "--out", out, "--seeds", ",".join(map(str, pair)),
                               "--parallel", "2", "--chunk", "1", "--stagger", "5",
                               "--timeout", "1500", "--tries", "1", "--accept-eula"],
                              log, env=_ok_env(), timeout=3600)
        assert rc == 0, smoke.tail(log)
        for s in pair:
            done = ((out / f"ep_{s}" / "meta.npz").is_file()
                    or (out / f"ep_{s}.filtered").is_file() or (out / f"ep_{s}.wedged").is_file())
            assert done, f"seed {s}: no done marker (meta.npz / .filtered / .wedged)"
        for s in pair:
            if (out / f"ep_{s}" / "meta.npz").is_file():
                kept = s
                break
        if kept is not None:
            break
    assert kept is not None, f"no candidate seed was kept: {smoke.tail(log)}"
    import numpy as np

    meta = np.load(out / f"ep_{kept}" / "meta.npz")
    for key in ("states", "actions", "grasp_pose", "seed", "verdict", "info", "tilt_deg"):
        assert key in meta.files, key
    assert meta["states"].shape[0] >= 50 and meta["states"].shape[1] == 7
    assert (out / f"ep_{kept}" / "mask_0000.png").is_file()
    state["strict_seed"] = kept
    state["strict_dir"] = out
    print(f"\n[smoke] strict episode kept: seed {kept}, {meta['states'].shape[0]} frames")


def test_generate_recovery(work: Path, state: dict) -> None:
    _require_tool_assets()
    seed = _need(state, "strict_seed")
    winners = _need(state, "strict_dir")
    sel = work / "strict_one.json"
    sel.write_text(json.dumps({"name": "smoke_strict", "seeds": [int(seed)],
                               "rule": "the strict episode kept by the sim smoke tier"}))
    out = work / "recovery"
    log = work / "logs" / "generate_recovery.log"
    log.unlink(missing_ok=True)
    rc = smoke.run_logged([sys.executable, "-m", "simretract.cli.generate", "recovery",
                           "--out", out, "--strict-json", sel, "--winners", winners,
                           "--draws", "1", "--range", "0:1", "--parallel", "1", "--chunk", "1",
                           "--timeout", "600", "--tries", "2", "--accept-eula"],
                          log, env=_ok_env(), timeout=1800)
    assert rc == 0, smoke.tail(log)
    ep = out / f"ep_{seed}_p0"
    kept = (ep / "meta.npz").is_file()
    assert kept or (out / f"ep_{seed}_p0.filtered").is_file() or (out / f"ep_{seed}_p0.wedged").is_file(), \
        f"no done marker for ep_{seed}_p0: {smoke.tail(log)}"
    if kept:
        import numpy as np

        info = json.loads(str(np.load(ep / "meta.npz")["info"]))
        assert "perturb_win" in info and info["perturb_win"]["end"] >= info["perturb_win"]["start"]
        assert "perturb" in info and 0.0 < info["perturb"]["frac"] < 1.0
    state["recovery_kept"] = kept
    state["recovery_dir"] = out
    state["recovery_json"] = sel
    print(f"\n[smoke] recovery draw ep_{seed}_p0: {'kept' if kept else 'filtered (valid outcome)'}")


def _cache_args(seed: int) -> list[str]:
    from simretract import config

    pool = json.loads(config.selection_path("test_pool").read_text())["seeds"]
    # a held-out seed is excluded by default; the smoke keeps it on purpose
    return ["--no-exclude-seeds"] if int(seed) in set(pool) else []


def test_build_cache_strict(work: Path, state: dict) -> None:
    seed = _need(state, "strict_seed")
    raw = _need(state, "strict_dir")
    log = work / "logs" / "build_cache.log"
    log.unlink(missing_ok=True)
    rc = smoke.run_logged([sys.executable, "-m", "simretract.cli.build_cache", "--set", "strict",
                           "--raw", raw, "--out", work / "cache", "--workers", "4",
                           *_cache_args(seed)], log, timeout=900)
    assert rc == 0, smoke.tail(log)
    meta = json.loads((work / "cache" / "meta.json").read_text())["strict"]
    # the pair that produced the keep may hold two kept episodes
    assert meta["N"] > 0 and f"ep_{seed}" in meta["source_episodes"]
    assert (work / "cache" / "strict_plane.npy").is_file()
    assert "VERIFY PASS" in log.read_text(errors="replace")


def test_build_cache_recovery(work: Path, state: dict) -> None:
    seed = _need(state, "strict_seed")
    raw = _need(state, "recovery_dir")
    if not _need(state, "recovery_kept"):
        pytest.skip("the recovery draw was filtered by the recorder (valid outcome); nothing to cache")
    log = work / "logs" / "build_cache.log"
    rc = smoke.run_logged([sys.executable, "-m", "simretract.cli.build_cache", "--set", "recovery",
                           "--raw", raw, "--out", work / "cache", "--workers", "4",
                           *_cache_args(seed)], log, timeout=900)
    assert rc == 0, smoke.tail(log)
    meta = json.loads((work / "cache" / "meta.json").read_text())["recovery"]
    assert meta["N"] > 0 and meta["flags"]["perturb_trim"] is True


def _run_eval(work: Path, name: str, extra: list[str], log: Path) -> dict:
    _require_tool_assets()
    sel = work / "eval_seed.json"
    sel.write_text(json.dumps({"name": "smoke_eval", "seeds": [EVAL_SEED],
                               "rule": "one benchmark seed for the sim smoke tier"}))
    out = work / name
    # a Kit worker occasionally stalls mid-episode; the protocol's second launch
    # per seed covers it, and a short wall cap keeps the smoke bounded
    rc = smoke.run_logged([sys.executable, "-m", "simretract.cli.eval_sim", "--seeds", sel,
                           "--passes", "1", "--launches", "2", "--timeout", "420",
                           "--out", out, "--name", name,
                           "--port", str(smoke.find_free_port()), "--accept-eula", *extra],
                          log, env=_ok_env(), timeout=2400)
    assert rc == 0, smoke.tail(log)
    assert (out / "results.json").is_file() and (out / "table.md").is_file()
    assert "ROW " in log.read_text(errors="replace")
    result = out / f"result_{EVAL_SEED}.json"
    if result.is_file():
        row = json.loads(result.read_text())
    else:
        # scene_reject / no_grasp seeds write no result file; the ledger records them
        ledger = json.loads((out / "ledger.json").read_text())
        attempts = ledger["attempts"].get(str(EVAL_SEED), [])
        assert attempts, f"no result file and no ledger entry for seed {EVAL_SEED}: {smoke.tail(log)}"
        row = {"outcome": attempts[-1]["outcome"], "queries": 0}
    assert row["outcome"] in {"completed", "timeout", "no_grasp"}, row
    if row["outcome"] != "no_grasp":
        assert row["queries"] >= 1, row
    print(f"\n[smoke] eval {name}: outcome {row['outcome']}, queries {row.get('queries')}")
    return row


def test_eval_act_checkpoint(work: Path) -> None:
    log = work / "logs" / "eval_act.log"
    log.unlink(missing_ok=True)
    ckpt = smoke.ensure_act_smoke_checkpoint(smoke.work_dir(), work / "logs" / "act_smoke.log")
    _run_eval(work, "eval_act", ["--ckpt", str(ckpt)], log)


def test_eval_stub_server(work: Path) -> None:
    log = work / "logs" / "eval_stub.log"
    log.unlink(missing_ok=True)
    _run_eval(work, "eval_stub", ["--stub"], log)
