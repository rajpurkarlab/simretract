# SimRetract

Simulation-trained retractor take-over and holding with a WidowX-AI robot and a
wrist-mounted RealSense D405. The policy learns from scripted demonstrations in
Isaac Sim and uses masked RGB-D, joint state, and an estimated tool-plane normal.

This repository includes simulation, demonstration generation, training,
evaluation, and real-robot deployment.

[Project website](https://example.com) (placeholder)

## Get started

Real deployment uses a Trossen WidowX-AI follower, the supplied flat-pad finger
design, a wrist-mounted D405, and a CUDA workstation. Read the
[hardware setup](docs/HARDWARE.md) and set `robot.follower_ip` in
`configs/real/default.yaml` before connecting the client.

The finger design is adapted from [Source Robotics' SSG-48 gripper](https://github.com/Source-Robotics/SSG-48-adaptive-electric-gripper),
with modifications to fit the Trossen WidowX-AI. [Included meshes and attribution](assets/gripper/README.md).

```bash
git clone --recurse-submodules https://github.com/xucheng-wang/simretract.git
cd simretract
bash envs/setup.sh
conda activate simretract-policy
```

The setup creates separate policy, simulation, and mask environments. Checkpoint
and dataset locations are configured in `configs/hub.yaml`; its release-location
placeholder must be filled before using the download commands. Gated weights,
including SAM 3, require their upstream access terms. See
[environment and weight setup](docs/REPRODUCING.md#environments-and-weights).

With a compatible checkpoint installed at `checkpoints/baseline`:

```bash
scripts/run_real.sh checkpoints/baseline dry  # perception and inference; no motion
scripts/run_real.sh checkpoints/baseline go   # press ENTER to start motion
```

The default real preset removes the top **1 cm** from both plane estimation and
model visibility, then estimates a plane from the remaining foreground. An
unreliable plane is withheld. Legacy no-cut and pixel-fraction modes remain
available; see [real-run options](docs/HARDWARE.md).

## Simulation and training

Simulation requires a separately obtained retractor asset, prepared locally as
described in [ASSETS.md](docs/ASSETS.md). Real deployment does not require it.

```bash
scripts/build_assets.sh --accept-eula
scripts/eval_sim.sh checkpoints/baseline
```

To rebuild a model, follow [Reproducing SimRetract](docs/REPRODUCING.md): download
or generate demonstrations, build the frame cache, train, and evaluate under the
fixed [benchmark protocol](docs/PROTOCOL.md).

## Documentation

| Task | Guide |
|---|---|
| Navigate the project | [Documentation index](docs/README.md) |
| Configure the robot and camera | [Hardware](docs/HARDWARE.md) |
| Prepare the simulation asset | [Assets](docs/ASSETS.md) |
| Rebuild a checkpoint | [Reproduction](docs/REPRODUCING.md) |
| Understand model variants | [Presets and ablations](docs/ABLATIONS.md) |
| Check data and weights | [Datasets](data/README.md) / [Checkpoints](checkpoints/README.md) |
| Run validation | [Test tiers](tests/README.md) |

The Python package is in `simretract/`; runnable wrappers are in `scripts/` and
configuration is in `configs/`. Downloaded data, weights, and generated assets
are kept outside version control.

## Citation and licensing

See [CITATION.cff](CITATION.cff) for citation metadata.

We release our code, policy checkpoints, and robot-action/grasp-point datasets. The third-party retractor asset by HQ3DMOD must be obtained separately under its publisher's license. Asset acquisition and local preparation instructions are provided in this repository.

License choices and release status are recorded in [LICENSING.md](LICENSING.md),
[checkpoint terms](LICENSE-CHECKPOINTS.md), and [dataset terms](LICENSE-DATA.md).
[Third-party notices](THIRD_PARTY_NOTICES.md) cover upstream software, weights,
and assets. No retractor meshes, textures, or source assets are included.
