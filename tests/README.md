# Tests

Run a tier from the checkout:

```bash
scripts/smoke_test.sh unit
scripts/smoke_test.sh gpu
```

The wrappers use the interpreters configured by `envs/setup.sh`.
`simretract-smoke --tier unit` is the equivalent console command.

| Tier | Requirements | Coverage |
|---|---|---|
| `unit` | Policy environment; no hardware or GPU | Geometry, metric crops, model-input contracts, tracking state, startup, configuration, data pipeline, scoring, wire protocol, and repository contents |
| `gpu` | Policy environment and CUDA | Synthetic cache, small ACT training run, policy serving with legacy and metric queries, and hold-pose stub |
| `sim` | Policy and Isaac environments, CUDA, locally prepared tool asset | Asset verification, demonstration generation, cache construction, and one-seed evaluation |
| `real-dry` | Configured robot, D405, SAM weights, and policy weights | Live perception and inference with the arm read-only |
| `release` | Policy environment | License metadata, release locations, checksums, required notices, and asset exclusion |

`all` runs unit, GPU, and simulation tiers. Unit and GPU checks normally take
under a minute each; simulation takes about 10 minutes and may retry stalled
workers. The pi0 training smoke requires base weights and a GPU with at least
100 GB; it skips otherwise. Release gates intentionally fail while release locations and terms remain
placeholders.

SAM lifetime tests also exercise the installed Ultralytics implementation.
Run these in the mask environment; pytest is not required:

```bash
conda activate simretract-mask
python -m unittest discover -s tests/unit -p test_locked_tracking.py -v
```

The tests cover selected-ID retention, genuine empty outputs, disconnected
visible pieces, episode reset, temporal warm-up, and growing frame capacity.
They use synthetic masks and do not load model weights or access hardware.

Tests generate their own fixtures. Real demonstration media in `assets/media/`
is independent of the test suite. GPU and simulation outputs live in ignored
`smoke_tmp/`; `SIMRETRACT_SMOKE_DIR` relocates them. See
[Hardware](../docs/HARDWARE.md) for deployment setup and
[Assets](../docs/ASSETS.md) for local simulation asset preparation.
