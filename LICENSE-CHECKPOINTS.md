# Checkpoint terms

The released policy checkpoints (`checkpoints/README.md`) are fine-tuned or
trained from the base weights named in `configs/hub.yaml`. Each row carries
the terms of its lineage; the terms below apply in addition to any terms
attached to the code that loads the weights. No checkpoint contains
geometry of the third-party tool asset: the models see rendered,
tool-masked RGB-D frames and a plane normal during training and store only
network parameters and normalization statistics.

| row | lineage | terms |
|---|---|---|
| `baseline`, `no_dropout`, `no_recovery`, `half_data`, `no_plane`, `no_mask`, `no_smooth`, `no_standoff` (pi0) | fine-tuned from the released `pi0_base` copy | Gemma Terms of Use and Gemma Prohibited Use Policy (the weights embed a PaliGemma backbone, a Gemma derivative); the Apache-2.0 notices of the lerobot pi0 conversion and of openpi are preserved (below) |
| `pi0_base` | a re-hosted copy of an earlier hub revision of `lerobot/pi0_base` (lerobot's conversion of the openpi pi0 base weights) | same as above; redistributed so the fine-tuning initialisation is fixed (`checkpoints/README.md`) |

## Preserved upstream notices

- pi0 base weights: released by Physical Intelligence (openpi) under
  Apache-2.0; converted to the lerobot format by Hugging Face under
  Apache-2.0. The PaliGemma backbone inside them is a Gemma derivative:
  use is subject to the Gemma Terms of Use
  (<https://ai.google.dev/gemma/terms>) and the Gemma Prohibited Use Policy.
  The pi0 tokenizer (`google/paligemma-3b-pt-224`) is gated on the hub and
  is downloaded by the user after accepting those terms; it is not
  redistributed here.
- The lerobot policy implementations that run every row: Apache-2.0, Hugging Face.

The hub model card of the checkpoint repository repeats this table. No
checkpoint of the other backbones (`simretract-train --policy smolvla | act |
pi05 | pi0_fast | fastwam`) is released; a user who trains one inherits the
terms of the respective base weights named in `configs/hub.yaml`: SmolVLA
base weights (`lerobot/smolvla_base`) and the SmolVLM2 tokenizer are
Apache-2.0 (Hugging Face); the ResNet-18 ImageNet weights used by ACT are
BSD-3-Clause (the PyTorch/torchvision authors); the pi0.5, pi0-FAST and
Fast-WAM bases carry the terms of their hub cards.
