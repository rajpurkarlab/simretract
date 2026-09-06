# Ablations and reference results

Every preset in `configs/train/presets.yaml` expands to an exact flag set for
`simretract-train`; the presets differ from `baseline` in exactly one thing.
The common recipe is: strict-500 + recovery far-1000 caches, 50/50 set
sampling, masked RGB + masked float depth as two 224 px letterboxed camera
streams, state = 7 joint positions + 3-D plane normal (`qpos_plane`), gripper
mask never an input (`nogrip`), fixed task string, augmentation v4.2, z-score
normalization, late-trajectory dropout (`docs/DROPOUT.md`), AdamW 2.5e-5 with
betas (0.9, 0.95), weight decay 1e-10, 1000-step warm-up then cosine to 0.1x,
gradient clip 10, bf16 autocast, batch 16 x accum 4, 15,000 steps, 5% of
episodes held out as validation (never evaluated).

## Presets

| preset | id | flags | needs | what changes vs. baseline |
|---|---|---|---|---|
| `baseline` | - | `--policy pi0 --data full --ablation none --dropout --init checkpoints/pi0_base` | strict + recovery caches, pi0 base weights, PaliGemma tokenizer | nothing |
| `no_recovery` | A0 | `--data strict_only` | strict cache only | drops recovery co-training (single set, frame-proportional sampling) |
| `no_depth` | A3 | `--ablation no_depth` (input mode `masked_rgb_nodepth`) | as baseline | removes the depth camera stream; single masked-RGB camera, plane normal kept |
| `half_data` | A7 | `--data half` | as baseline | deterministic half selection of both sets (250 strict seeds, far-500), stats still from the full sets |
| `smolvla` | B2 | `--policy smolvla --smolvla-full` (init `lerobot/smolvla_base`) | strict + recovery caches, SmolVLA base + SmolVLM2 tokenizer | backbone swap; VLM and vision tower unfrozen, baseline optimizer regime |
| `act` | - | `--policy act --act-opt baseline` | strict + recovery caches, torchvision ResNet18 weights | backbone swap; ACT trained from scratch under the baseline optimizer regime, ImageNet mean/std on both streams |
| `no_plane` | A1 | `--ablation no_plane` (state `qpos`) | as baseline | removes the plane normal (7-D state) |
| `no_mask` | A2 | `--ablation no_mask` (input mode `full_rgbd`) | as baseline | unmasked RGB and depth to the policy; the mask still feeds the plane fit |
| `no_aug` | A4 | `--ablation no_aug` | as baseline | no mask corruption, no leak fills, no depth artifacts, no plane noise, no dropout ramp |
| `fastwam` | B3 | `--policy fastwam` (chunk 48, video co-training) | Fast-WAM base weights, `envs/fastwam.txt` | world-action-model backbone |
| `pi05`, `pi0_fast` | - | `--policy pi05` / `--policy pi0_fast` | their base weights | backbone swap |

The six rows above the line (`baseline` ... `act`) are the reference rows;
released checkpoints exist for exactly these rows, trained with the dropout
rule on. Every preset trains from the released raw sets or their regenerated
equivalents; no preset needs data that is not released.

## Reference results

Simulation benchmark, `test_core50.json`, three-attempt protocol,
`exec_h 50`, `smooth 0.45`, no plane top cut (`docs/PROTOCOL.md`); the
benchmark is not bit-reproducible (`docs/PROTOCOL.md`, "Nondeterminism"), so a
re-run lands near, not on, these numbers:

| preset | grabbed | secured | held | kept | timeouts | offset mm, mean +/- sd [med] | twist deg, mean +/- sd [med] |
|---|---|---|---|---|---|---|---|
| `baseline` | 50 | 49 | 33 | 23 | 0 | 8.1 +/- 6.0 [6.7] | 4.6 +/- 7.6 [2.1] |
| `no_recovery` (A0) | 47 | 42 | 27 | 17 | 3 | 15.1 +/- 20.6 [7.0] | 5.8 +/- 8.6 [4.4] |
| `no_depth` (A3) | 50 | 47 | 37 | 30 | 0 | 7.9 +/- 6.6 [6.4] | 3.9 +/- 2.8 [3.4] |
| `half_data` (A7) | 49 | 45 | 35 | 16 | 1 | 13.7 +/- 14.3 [9.5] | 7.0 +/- 8.3 [3.1] |
| `smolvla` | 50 | 47 | 40 | 9 | 0 | 20.1 +/- 16.3 [15.9] | 5.8 +/- 9.3 [3.1] |
| `act` | 42 | 27 | 19 | 10 | 8 | 12.2 +/- 9.9 [9.6] | 6.3 +/- 11.1 [3.8] |

## Reproducing a row

```bash
scripts/train.sh no_depth --init checkpoints/pi0_base       # runs/no_depth/final
scripts/eval_sim.sh runs/no_depth/final                     # eval/no_depth/table.md, ROW line
```

The trainer records the resolved preset in `norm.json["train_regime"]`, so a
checkpoint can always be traced back to its flags; the unit tests assert that
each of the six reference presets resolves to its reference flag set.
