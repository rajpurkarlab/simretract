# SimRetract

SimRetract is a complete sim-to-real imitation-learning stack for one surgical
hand-over task: a WidowX-AI arm with a wrist RealSense D405 takes a retractor
out of an operator's hand and holds its presented pose. Every demonstration is
generated in Isaac Sim by a scripted expert, the policy sees only a masked RGB-D
view of the tool plus a camera-frame plane normal of the tool blade, and the
trained model runs on the real arm without any real-robot data. This repository
rebuilds the released models from the ground up: assets, simulation scene,
demonstration generation, frame cache, training, the simulation benchmark, and
the real-robot deployment client.

## Demo

<!-- Demo videos: add links or embeds here. -->
- Simulation benchmark rollout: _video placeholder_
- Real-robot hand-over: _video placeholder_

## What is in the box

- **Simulation environment** - Isaac Sim 5.1 scene with the WidowX-AI follower,
  flat-pad fingers, a compliant "operator hand" that presents the tool, a
  randomized occluder hand, per-episode lighting, and D405-matched wrist optics.
- **Demonstration generator** - a scripted expert that searches a grasp,
  records a 25 Hz hand-over (strict set), and a perturb-and-recover variant that
  interrupts the reach and records the recovery (recovery set); a lane driver
  runs many Isaac processes in parallel with timeouts and resume.
- **Trainer** - one entry point with a model switch (pi0, SmolVLA, ACT; optional
  pi0.5, pi0-FAST, Fast-WAM), data switches, ablation switches, and the
  late-trajectory dropout rule.
- **Simulation benchmark** - a two-process closed-loop evaluator (policy server +
  Isaac client) with a fixed seed set, a three-attempt protocol, and a scorer
  that prints the grabbed / secured / held / kept funnel.
- **Real deployment** - the same policy server, a SAM 3 tool-mask server, and a
  client that reproduces the training observation on the real arm.

## Install

```bash
git clone --recurse-submodules <repo-url> simretract && cd simretract
bash envs/setup.sh              # creates simretract-sim, simretract-policy, simretract-mask
conda activate simretract-policy
export OMNI_KIT_ACCEPT_EULA=YES # required by every headless Isaac Sim process
```

`envs/setup.sh` needs conda (on a fresh Miniconda accept the default channel's
terms once: `conda tos accept --override-channels --channel
https://repo.anaconda.com/pkgs/main`). It records the three interpreters in
`configs/envs.local.yaml` (git-ignored; the tracked `configs/envs.yaml` is the
template). Every `simretract-*` and `hf` command in this README runs in the
`simretract-policy` environment and assumes it is active; the `scripts/*.sh`
wrappers pick the right interpreter themselves and work from any shell. Some
assets are gated on the Hugging Face hub; accept their terms in a browser, then:

```bash
hf auth login                                                    # once
hf download google/paligemma-3b-pt-224 --include "tokenizer*"     # pi0 tokenizer (Gemma terms)
simretract-download base-weights                                 # pi0 base weights (see checkpoints/README.md)
hf download facebook/sam3 sam3.pt --local-dir checkpoints/sam3    # SAM 3 (real robot only)
```

`simretract-download` and the `scripts/download_*.sh` wrappers work once `org`
in `configs/hub.yaml` names the organisation that hosts the release; with the
shipped placeholder they stop with a one-line message.

The retractor 3D model is a third-party asset that is not part of this
repository. Every simulation command (generation, benchmark, the asset
build) needs it prepared locally from your own licensed copy: obtain the Fab
listing named in `configs/assets/retractor_rake.yaml`, drop the downloaded
folder under `assets/source/` as it is, and build:

```bash
# assets/source/vretractor/{vretractor.fbx, textures/}  <- the downloaded folder, unchanged (git-ignored)
scripts/build_assets.sh --accept-eula      # checksum, tool meshes, fingerprint check, USD, grasp-candidate check
```

The converter installs nothing itself: `docs/ASSETS.md` lists the two
optional packages it needs and what each check line means. Building from
this Fab file with the shipped recipe reproduces the released tool geometry
within 0.25 um, and the released grasp candidates were generated from it.
Real-robot deployment does not need the asset.

## Quick start

The quick start assumes the same hardware setup as ours: a Trossen
WidowX-AI follower arm with the flat-pad fingers, an Intel RealSense D405
wrist camera, and a workstation with a CUDA GPU for the policy server and
the SAM 3 mask server (`docs/HARDWARE.md`). Without that hardware only the
simulation parts apply.

```bash
scripts/download_checkpoints.sh baseline            # -> checkpoints/baseline/
scripts/eval_sim.sh checkpoints/baseline            # simulation benchmark, core-50 seeds, 3 passes
scripts/run_real.sh checkpoints/baseline dry        # perception + inference on the robot, no motion
scripts/run_real.sh checkpoints/baseline go         # live run (ENTER gate before the arm moves)
```

Before the first real run set `robot.follower_ip` in `configs/real/default.yaml`
and read `docs/HARDWARE.md`. The simulation benchmark needs the locally built
tool assets (Install, `docs/ASSETS.md`).

## Rebuild from scratch

Two paths lead to the released checkpoints:

- **Path A (reproduce)** - download the released raw episodes, build the cache,
  train. Reproduces the checkpoints up to training nondeterminism (the sampler
  and augmentation are unseeded).
- **Path B (regenerate)** - regenerate the demonstrations. Physics search and
  GPU rendering are not bit-reproducible, so you get a different set of
  episodes from the same distribution, then the same cache and training steps.

Path A:

```bash
scripts/download_data.sh all                        # data/raw/strict_train (500 episodes), data/raw/recovery_far1000 (1000)
scripts/build_cache.sh strict --verify              # --verify: fingerprints must match configs/data.yaml
scripts/build_cache.sh recovery --verify
scripts/train.sh baseline --init checkpoints/pi0_base
```

The `no_smooth` and `no_standoff` rows train on their own raw pairs
(`scripts/download_data.sh strict_nosmooth`, `recovery_nosmooth`,
`strict_nostandoff`, `recovery_nostandoff`, then `scripts/build_cache.sh` with
the same set names; `data/README.md`).

Path B (six commands, then evaluate):

```bash
# 1. assets from your licensed copy (assets/source/vretractor/): tool meshes, fingerprint check, USD, grasp-candidate check
simretract-build-assets --tool retractor_rake --accept-eula
# 2. strict demonstrations: fill 4 grasp-height bins with 125 kept episodes each
simretract-generate strict --out data/raw/strict --seed-start 400001 --quota 125 --bins 4 --parallel 3
# 3. assemble the strict-500 selection and the training view of it
simretract-assemble strict --raw data/raw/strict --per-bin 125 \
    --out data/selections/strict_500.json --train-dir data/raw/strict_train
# 4. recovery demonstrations: 4 perturbation draws per strict seed, then the far-1000 selection
simretract-generate recovery --strict-json data/selections/strict_500.json \
    --winners data/raw/strict_train --out data/raw/recovery --draws 4 --range 0:2000 --parallel 3
simretract-assemble recovery --raw data/raw/recovery --strict-json data/selections/strict_500.json \
    --exact 4 --far 1000 --out-train data/selections/recovery_train_2000.json \
    --out-far data/selections/recovery_far1000.json --far-dir data/raw/recovery_far1000
# 5. frame cache + plane-normal sidecar for both sets
simretract-build-cache --set strict --raw data/raw/strict_train --out data/cache
simretract-build-cache --set recovery --raw data/raw/recovery_far1000 --out data/cache
# 6. train the baseline (pi0 + recovery co-training + dropout)
simretract-train --preset baseline --cache data/cache --out runs/baseline --init checkpoints/pi0_base
# evaluate
simretract-eval-sim --ckpt runs/baseline/final --seeds data/selections/test_core50.json --passes 3 --out eval/baseline
```

## Models

- [SimRetract](https://YOUR_HOST/simretract/checkpoints/baseline)
- [w/o dropout](https://YOUR_HOST/simretract/checkpoints/no_dropout)
- [w/o recovery](https://YOUR_HOST/simretract/checkpoints/no_recovery)
- [half data (250 + 500)](https://YOUR_HOST/simretract/checkpoints/half_data)
- [w/o plane normal](https://YOUR_HOST/simretract/checkpoints/no_plane)
- [w/o mask](https://YOUR_HOST/simretract/checkpoints/no_mask)
- [w/o smoothing](https://YOUR_HOST/simretract/checkpoints/no_smooth)
- [w/o standoff](https://YOUR_HOST/simretract/checkpoints/no_standoff)
- [smolvla](https://YOUR_HOST/simretract/checkpoints/smolvla)

## Hardware

- Training: 1x NVIDIA H200 (141 GB). pi0 needs ~107 GB at batch 16 x accum 4.
- Simulation, benchmark and deployment: 1x NVIDIA RTX 5090 (32 GB), 64 GB host RAM.
- Robot: Trossen WidowX-AI follower arm with the flat-pad fingers from `assets/gripper`,
  Intel RealSense D405 on the wrist, SAM 3 for the tool mask.

Details, memory budgets, ports and network placeholders: `docs/HARDWARE.md`.

## Layout

```
configs/        hub ids, paths, interpreters, the tool asset recipe, generation / training / eval / real presets
envs/           requirements per environment and envs/setup.sh
third_party/    trossen_ai_isaac submodule (robot USD + controller)
assets/         tool annotations (meta.json, grasp candidates), gripper fingers, silhouette LUT;
                source/<pack>/ holds your asset folder (git-ignored); the tool meshes
                and USD are built locally and never tracked (docs/ASSETS.md)
data/           selections/ (shipped seed lists), raw/ and cache/ (downloaded or generated)
checkpoints/    downloaded checkpoints and base weights
simretract/     the package: sim/, data/, train/, serve/, eval/, real/, cli/
scripts/        one-line wrappers for the canonical commands
tests/          unit / gpu / sim / real_dry smoke tiers
docs/           asset acquisition and local preparation, data format, benchmark protocol,
                ablations, hardware, dropout rule
```

## Smoke tests

```bash
scripts/smoke_test.sh all        # unit (< 1 min) + gpu (~1 min) + sim (~10 min); or one tier
```

Each tier is a pytest module under `tests/` that drives the real console
commands; `simretract-smoke --tier <tier>` is the same entry point from
Python. The sim tier starts by building the tool assets from the asset
file under `assets/source/` and skips that step with a message when there
is none. Details and outputs: `tests/README.md`.

## Data and checkpoints

All hub repository ids live in `configs/hub.yaml` (`org: YOUR_HF_ORG` until the
release is published); nothing else in the code names a repository. What ships,
what is downloaded, expected counts and cache fingerprints: `data/README.md` and
`checkpoints/README.md`. Benchmark results are not shipped;
`scripts/eval_sim.sh` produces them from any checkpoint under the protocol of
`docs/PROTOCOL.md`.

## Citation

See `CITATION.cff`. The presets in `docs/ABLATIONS.md` are the model rows of
the paper cited there.

## Release contents and licensing

We release our code, policy checkpoints, and robot-action/grasp-point datasets. The third-party retractor asset by HQ3DMOD must be obtained separately under its publisher's license. Asset acquisition and local preparation instructions are provided in this repository.

- `LICENSING.md`: the code license (`LICENSE`), the checkpoint terms
  (`LICENSE-CHECKPOINTS.md`), the dataset terms (`LICENSE-DATA.md`) and the
  third-party asset rule.
- `docs/ASSETS.md`: the Fab listing to obtain and the local preparation that
  turns your copy into the meshes, USD and grasp candidates. No file that
  carries the asset's geometry is tracked; `assets/LICENSES.md` lists every
  asset file as tracked or built locally.
- `THIRD_PARTY_NOTICES.md`: third-party software, weights, tokenizers and
  the asset listing, with their sources and licenses.
- `scripts/smoke_test.sh release` checks the release gates.
