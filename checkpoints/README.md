# Checkpoints

`checkpoints/` is git-ignored. `simretract-download checkpoints` fills it from
the model repository named in `configs/hub.yaml` (`org: YOUR_HF_ORG` until the
release is published), verifying file sizes and the sha256 values listed there.

## Released rows

| row | preset | directory | policy | size |
|---|---|---|---|---|
| baseline | `baseline` | `checkpoints/baseline` | pi0 | 16.1 GB |
| no recovery (A0) | `no_recovery` | `checkpoints/no_recovery` | pi0 | 16.1 GB |
| no depth (A3) | `no_depth` | `checkpoints/no_depth` | pi0 | 16.1 GB |
| half data (A7) | `half_data` | `checkpoints/half_data` | pi0 | 16.1 GB |
| SmolVLA | `smolvla` | `checkpoints/smolvla` | SmolVLA | 1.2 GB |
| ACT | `act` | `checkpoints/act` | ACT | 0.2 GB |
| pi0 base weights | (init) | `checkpoints/pi0_base` | pi0 | 14 GB |

Every directory holds `config.json`, `model.safetensors` and `norm.json`
(`docs/DATA_FORMAT.md`, section 5). Reference benchmark numbers per row:
`docs/ABLATIONS.md`.

The released checkpoints contain network parameters and normalization
statistics only: no geometry of the third-party tool asset the training
data were rendered with (the asset is obtained and prepared locally,
`docs/ASSETS.md`). Each row carries the terms of the base weights it
descends from, with the upstream notices preserved: `LICENSE-CHECKPOINTS.md`.

```bash
scripts/download_checkpoints.sh baseline                 # one row
scripts/download_checkpoints.sh all --base-weights       # every row + pi0_base (~95 GB)
simretract-download base-weights                         # pi0_base only
```

## Base weights

The released `pi0_base` directory is the copy of the pi0 base weights that
every pi0 row was fine-tuned from. It is not tensor-identical to the current
hub `lerobot/pi0_base`, so training from the hub id gives a different model.
`simretract-train --init` therefore accepts only a local checkpoint directory
(`config.json` + `model.safetensors`), which `simretract-download base-weights`
writes and verifies against the sha256 in `configs/hub.yaml` once that table
is filled in; pass `--allow-unverified-init` to train from a hub id or any
other directory, knowing the result is not the released row.

The other backbones initialise from hub ids recorded in `configs/hub.yaml`:
`lerobot/smolvla_base` with the `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
tokenizer (SmolVLA), torchvision ResNet18 ImageNet weights (ACT, downloaded
by torchvision on first use), and the optional pi0.5 / pi0-FAST / Fast-WAM
bases.

## Gated assets

**PaliGemma tokenizer (Gemma Terms of Use).** pi0 tokenizes the task prompt
with `google/paligemma-3b-pt-224`, a gated repository. Accept the Gemma Terms
of Use on its hub page, then

```bash
hf auth login
hf download google/paligemma-3b-pt-224 --include "tokenizer*"
```

The pi0 checkpoints in this release, including `pi0_base`, contain a PaliGemma
backbone and are derivative works of Gemma: their use is subject to the Gemma
Terms of Use and the Gemma Prohibited Use Policy in addition to the Apache-2.0
notices of the lerobot conversion (`LICENSE-CHECKPOINTS.md`).

**SAM 3 (real robot only).** The mask server needs `sam3.pt` from the gated
`facebook/sam3` repository, released under the SAM 3 License, which does not
permit redistribution of the weights: the mask server never downloads them
itself and takes the path with `--ckpt`. Accept the license on the hub page,
then

```bash
hf download facebook/sam3 sam3.pt --local-dir checkpoints/sam3
```

and pass `--ckpt checkpoints/sam3/sam3.pt` to the mask server (the real-run
launcher reads the path from `configs/real/default.yaml`). The ultralytics
package that runs SAM 3 is AGPL-3.0 and stays in its own environment and
process.
