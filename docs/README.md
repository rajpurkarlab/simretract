# Documentation

Start with the [repository quick start](../README.md) for a real run using an
existing checkpoint. All command examples assume the repository root.

| Goal | Read |
|---|---|
| Install environments, rebuild data, and train | [Reproducing SimRetract](REPRODUCING.md) |
| Configure the WidowX-AI, D405, and real-run presets | [Hardware and deployment](HARDWARE.md) |
| Obtain and prepare the third-party simulation asset | [Asset preparation](ASSETS.md) |
| Run and interpret the fixed simulation benchmark | [Evaluation protocol](PROTOCOL.md) |
| Compare model and data variants | [Presets and ablations](ABLATIONS.md) |
| Understand observation and cache formats | [Data format](DATA_FORMAT.md) |
| Understand late-trajectory observation dropout | [Dropout](DROPOUT.md) |
| Find dataset selections and checkpoint requirements | [Data](../data/README.md) and [checkpoints](../checkpoints/README.md) |
| Run unit, GPU, simulation, or deployment checks | [Tests](../tests/README.md) |
| Review distribution terms and external components | [Licensing](../LICENSING.md) and [notices](../THIRD_PARTY_NOTICES.md) |

## Repository map

- `simretract/`: simulation, data processing, training, serving, evaluation, and real deployment.
- `scripts/`: wrappers around the package's command-line entry points.
- `configs/`: environment paths, asset recipes, data definitions, and run presets.
- `envs/`: environment setup and pinned requirements.
- `assets/`: project gripper assets, tool annotations, and demonstration media.
- `data/selections/`: training selections and held-out evaluation seeds.
- `tests/`: validation tiers and their fixtures.

Raw data, caches, checkpoints, local configuration, and third-party retractor
geometry are downloaded or generated locally and are not tracked.
