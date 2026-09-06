# Checkpoint terms

Released: the eight pi0 rows and the SmolVLA row of `configs/train/presets.yaml`
and the `pi0_base` copy the pi0 rows were fine-tuned from (`checkpoints/README.md`). The authors' own
contribution is released under a license to be chosen by the authors. No
checkpoint contains geometry of the third-party tool asset.

Upstream notices that travel with the weights:

- pi0 rows and `pi0_base`: the weights embed a PaliGemma backbone, a Gemma
  derivative; use is subject to the Gemma Terms of Use
  (<https://ai.google.dev/gemma/terms>) and the Gemma Prohibited Use Policy.
  The pi0 base weights are Apache-2.0 (Physical Intelligence, openpi;
  lerobot conversion by Hugging Face).
- The SmolVLA row and models a user trains with `--policy smolvla`: Apache-2.0
  base weights and tokenizer (Hugging Face, `lerobot/smolvla_base`).
- Models a user trains with `--policy act`: the torchvision ResNet-18
  ImageNet encoder is BSD-3-Clause.
