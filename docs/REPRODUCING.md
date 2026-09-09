# Reproducing SimRetract

Commands below run from the repository root. For the shortest path to an existing checkpoint, start with the [main README](../README.md).

## Environments and weights

```bash
git clone --recurse-submodules https://github.com/YOUR_GITHUB_ORG/simretract.git
cd simretract
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

Checkpoint and dataset download links will be released upon publication.
The corresponding `simretract-download` and `scripts/download_*.sh` commands
require those locations in `configs/hub.yaml`; the shipped values are placeholders.

The retractor 3D model is a third-party asset that is not part of this
repository. Every simulation command (generation, benchmark, the asset
build) needs it prepared locally from your own licensed copy: obtain the Fab
listing named in `configs/assets/retractor_rake.yaml`, drop the downloaded
folder under `assets/source/` as it is, and build:

```bash
# assets/source/vretractor/{vretractor.fbx, textures/}  <- the downloaded folder, unchanged (git-ignored)
scripts/build_assets.sh --accept-eula      # checksum, tool meshes, fingerprint check, USD, grasp-candidate check
```

The converter installs nothing itself: [ASSETS.md](ASSETS.md) lists the two
optional packages it needs and what each check line means. Building from
this Fab file with the shipped recipe reproduces the released tool geometry
within 0.25 um, and the released grasp candidates were generated from it.
Real-robot deployment does not need the asset.


## Rebuild from data or regenerate demonstrations

Two paths reproduce the training pipeline:

- **Path A (after publication)** - download the raw episodes, build the cache,
  and train.
- **Path B (regenerate)** - generate demonstrations locally, then build the
  cache and train.

Path A (after publication):

```bash
scripts/download_data.sh all                        # data/raw/strict_train (500 episodes), data/raw/recovery_far1000 (1000)
scripts/build_cache.sh strict --verify              # --verify: fingerprints must match configs/data.yaml
scripts/build_cache.sh recovery --verify
scripts/train.sh baseline --init checkpoints/pi0_base
```

The `no_smooth` and `no_standoff` rows train on their own raw pairs
(`scripts/download_data.sh strict_nosmooth`, `recovery_nosmooth`,
`strict_nostandoff`, `recovery_nostandoff`, then `scripts/build_cache.sh` with
the same set names; [data/README.md](../data/README.md)).

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
simretract-eval-sim --ckpt runs/baseline/final --seeds data/selections/test_core50.json --passes 1 --out eval/baseline
```


## Evaluate and verify

The [benchmark protocol](PROTOCOL.md) specifies the held-out seeds, first-attempt reporting, and outcome definitions. Use the same configuration when comparing checkpoints.

```bash
scripts/eval_sim.sh checkpoints/baseline
scripts/smoke_test.sh unit
scripts/smoke_test.sh all
```

The full smoke suite includes GPU and simulation checks; it requires the corresponding environments and locally prepared tool asset. See [test tiers](../tests/README.md) for prerequisites and outputs. Run `scripts/smoke_test.sh release` separately to check publication metadata and asset exclusions.

## Other configurations

[ABLATIONS.md](ABLATIONS.md) describes the preset definitions. [DATA_FORMAT.md](DATA_FORMAT.md) describes observations, actions, raw episodes, and caches. [DROPOUT.md](DROPOUT.md) documents the observation-dropout rule. Training and hardware budgets are in [HARDWARE.md](HARDWARE.md).
