# Tests

`scripts/smoke_test.sh [unit|gpu|sim|real-dry|release|all]` runs one tier with pytest
and prints the wall time of the tier and of every test (`--durations=0`);
`simretract-smoke --tier <tier>` is the same entry point from Python. Every
tier's pytest process runs in the policy environment (the only one that needs
pytest); the console commands under test delegate simulator work to the
`sim` interpreter recorded by `envs/setup.sh`. Each test carries the marker
of its tier directory (`tests/conftest.py`), so `pytest tests/unit -m unit`
selects a whole tier.

## Tiers

| tier | directory | environment | wall time | what it proves |
|---|---|---|---|---|
| unit | `tests/unit` | policy env, no GPU, no Isaac | < 1 min | sampling determinism against golden occluder specs; `seg_aug` shapes, ranges and golden digests; both plane fits against reference implementations on synthetic depth; the scorer on fixture results; preset resolution of the six reference rows against their launch flag sets; wire framing round-trip; config loading (including `tool_source`); the cache builder, plane sidecar and alignment gate on a tiny synthetic raw set; assembly and campaign logic; the real-client argument contract; the leak guard over every tracked file; the asset exclusion (no tool geometry or purchased pack would be committed, tracked tool files hold poses and metadata only); the asset recipe (loads, names the drop-in source, agrees with `meta.json`) and the fingerprint verifier on synthetic meshes; the reference checks of the asset build |
| gpu | `tests/gpu` | policy env + CUDA GPU | ~1 min | synthetic raw set -> `simretract-build-cache` (both sets) -> `simretract-train --preset act --smoke` (60 steps) -> `simretract-serve` boot, ping, regime check and one query on the smoke checkpoint; `--stub` server ping and hold-pose query; the pi0 smoke runs only with `checkpoints/pi0_base` present on a training-class GPU (>= 100 GB) |
| sim | `tests/sim` | policy env + Isaac env + GPU; the purchased tool pack under `assets/source/` or a `tool_source` override (`docs/ASSETS.md`) | ~10 min (up to ~25 min when a stalled Isaac worker is retried) | `simretract-build-assets --tool retractor_rake` (the default build: checksum the source, convert it in place with the recipe settings, check the meshes against the tracked `meta.json` and the recipe fingerprints, build the USD and check it, compare the re-sampled grasp candidates; skipped with a message when there is no source file, and the later steps then run only if the assets were built earlier); `simretract-build-assets --verify` and `--grasps --compare` on their own; `simretract-generate strict` on candidate seeds two at a time until one is kept (`.filtered` is a valid outcome); `simretract-generate recovery` for one draw of that seed (a stalled lane is killed after 10 minutes and relaunched once); `simretract-build-cache` on both smoke raw dirs; `simretract-eval-sim` on one benchmark seed with the ACT smoke checkpoint and with the stub server (a result with outcome in {completed, timeout, no_grasp} and at least one policy query) |
| real-dry | `tests/real_dry` | policy env; needs SAM 3 weights, `robot.follower_ip` and a D405, skipped otherwise | ~2 min | `simretract-run-real ... dry` (perception + inference, arm read-only) for a few ticks |
| release | `tests/release` | policy env | seconds | publication gates: `LICENSE` present, `LICENSING.md` / `LICENSE-CHECKPOINTS.md` / `LICENSE-DATA.md` present with the data license chosen, licence fields filled in, the release statement verbatim in the README, no tool geometry among the files that would be committed, hub organisation and checksum tables filled in; fails until the release is ready (not part of `all`) |

pi0 and SmolVLA training smokes need the reference training GPU and are
documented, not run, by the gpu tier on smaller cards.

## Work directory and outputs

The gpu and sim tiers write to `smoke_tmp/` in the repository (git-ignored;
`SIMRETRACT_SMOKE_DIR` relocates it): the synthetic cache and the ACT smoke
checkpoint (`smoke_tmp/act_smoke`, reused by the sim tier when present, or
`SIMRETRACT_SMOKE_ACT_CKPT`), the recorded smoke episodes, caches and
benchmark outputs under `smoke_tmp/sim/`, and one log per step under
`smoke_tmp/logs/` and `smoke_tmp/sim/logs/`. The unit tier uses pytest's
temporary directories only.

## Fixtures

No fixture files are shipped; everything the tests need is synthesized in
code so the repository carries no recordings:

- `simretract.data.synthetic` writes the raw episodes used by the cache
  tests and the gpu tier (geometric masks, a tilted depth plane, a joint
  trajectory with a stall and a linear gripper close, the recorder's
  `meta.npz` keys);
- the plane-fit tests build synthetic depth images with known normals and
  compare against verbatim reference implementations of the sidecar fit and
  the served fit;
- the `seg_aug` tests assert golden digests computed once from the reference
  augmentation implementation with a fixed RNG seed on a synthetic frame;
- the scorer tests write hand-written `result_<seed>.json` files that
  exercise every branch of the funnel;
- the sim tier writes its one-seed selection files itself.

No test contains real-robot recordings or benchmark results.
