# Ablations

Every preset in `configs/train/presets.yaml` expands to an exact flag set for
`simretract-train`; each preset differs from `baseline` in exactly one thing.
The common recipe is: a strict-500 cache plus its recovery far-1000 cache,
50/50 set sampling, masked RGB + masked float depth as two 224 px letterboxed
camera streams, state = 7 joint positions + 3-D plane normal (`qpos_plane`),
gripper mask never an input (`nogrip`), fixed task string, augmentation v4.2,
z-score normalization, late-trajectory dropout (`docs/DROPOUT.md`), AdamW
2.5e-5 with betas (0.9, 0.95), weight decay 1e-10, 1000-step warm-up then
cosine to 0.1x, gradient clip 10, bf16 autocast, batch 16 x accum 4, 15,000
steps, 5% of episodes held out as validation (never evaluated).

Presets carry flags and a description only. Benchmark results are not
recorded in this repository; `docs/PROTOCOL.md` is the procedure that
produces them from a checkpoint.

## Presets

Every preset is pi0 with the dropout rule on, except `no_dropout`. The
`baseline` flags are `--policy pi0 --data full --ablation none --dropout
--sets strict,recovery --set-frac 0.5,0.5 --input-mode masked_rgbd --state
qpos_plane --init checkpoints/pi0_base` plus the schedule above; the table
lists what each preset changes.

| preset | flags | needs | what changes vs. baseline |
|---|---|---|---|
| `baseline` | (the flags above) | strict + recovery caches, pi0 base weights, PaliGemma tokenizer | nothing |
| `no_dropout` | `--no-dropout` = `--drop-mask-p 0 --drop-plane-p 0 --plane-sig-end 3 --plane-glitch-end 0.03 --plane-rand-frac 0` | as baseline | the dropout rule off: no blind frames, plane noise constant (sigma 3 deg, 3% glitches) instead of ramped |
| `no_recovery` | `--data strict_only` = `--sets strict --set-frac none` | strict cache only | drops recovery co-training (single set, frame-proportional sampling) |
| `half_data` | `--data half` | as baseline | deterministic half selection of both sets (250 strict seeds, far-500, RNG 0); normalization statistics still from the full sets |
| `no_plane` | `--ablation no_plane` = `--state qpos` | as baseline | removes the plane normal (7-D state) |
| `no_mask` | `--ablation no_mask` = `--input-mode full_rgbd` | as baseline | unmasked RGB and depth to the policy; the mask still feeds the plane fit |
| `no_smooth` | `--sets strict_nosmooth,recovery_nosmooth` | the no-smoothing caches (`data/README.md`) | the training data: demonstrations recorded without reach smoothing (two straight segments, the grasp candidate's own standoff, no joint-rate limit) |
| `no_standoff` | `--sets strict_nostandoff,recovery_nostandoff` | the no-standoff caches (`data/README.md`) | the training data: demonstrations recorded without the reach standoff (the smoothed reach curves straight to the grasp) |

The two data variants keep every baseline flag, including the dropout rule;
only the cache pair changes. `--data half` works on any pair (`--preset
no_smooth --data half`): the strict selection of the first set supplies the
per-seed grasp heights the selection stratifies on (`--strict-json` overrides
it). Every preset trains from the released raw sets or their regenerated
equivalents; no preset needs data that is not released.

## Other switches

Two further ablation switches exist on the trainer without a preset:
`--ablation no_depth` (`--input-mode masked_rgb_nodepth`: a single masked-RGB
camera, plane normal kept) and `--ablation no_aug` (`--no-aug`: no mask
corruption, no leak fills, no depth artifacts, no plane noise, no dropout
ramp). A conflicting explicit `--input-mode` / `--state` is an error, never a
silent override.

## Other backbones

The trainer's `--policy` switch runs the same data path and inputs through
other policy classes. There is no preset for them; combine the switch with any
preset (`--preset baseline --policy act`) or with the explicit flags, and the
regime is recorded in `norm.json["train_regime"]` like any other run.

| `--policy` | knobs | init (`configs/hub.yaml` `base_weights`) | environment |
|---|---|---|---|
| `smolvla` | `--smolvla-full` / `--no-smolvla-full` (unfreeze the VLM + vision tower under the baseline optimizer regime, or SmolVLA's own preset) | `lerobot/smolvla_base` + the SmolVLM2 tokenizer | policy |
| `act` | `--act-opt baseline\|official` (baseline pi0 regime or ACT's own optimizer preset); trains from scratch, ResNet-18 ImageNet encoder, ImageNet mean/std on both streams | none (`--init ""`) | policy |
| `pi05` | `--pi05-opt baseline\|official`; state digitised into the prompt | `lerobot/pi05_base` | policy |
| `pi0_fast` | FAST action tokenizer, the port's optimizer preset | `lerobot/pi0fast-base` + `lerobot/fast-action-tokenizer` | policy |
| `fastwam` | `--chunk-h 48 --video-t 9 --video-ratio 6` (defaults for this policy), `--wam-*` | `lerobot/fastwam_base` | `simretract-fastwam` (`envs/fastwam.txt`) |

`--chunk-h`, `--video-t` and `--video-ratio` are ordinary trainer flags
(action chunk length; future kept frames per sample and their spacing for the
video co-training of `fastwam`). The policy server and the simulation
benchmark dispatch on the checkpoint type, so a backbone row is served and
evaluated with the same commands as a pi0 row (`exec_h` 48 for `fastwam`).

## Reproducing a row

```bash
scripts/train.sh no_plane --init checkpoints/pi0_base       # runs/no_plane/final
scripts/eval_sim.sh runs/no_plane/final                     # eval/no_plane/table.md, ROW line
```

The trainer records the resolved preset in `norm.json["train_regime"]`, so a
checkpoint can always be traced back to its flags; the unit tests assert that
each of the eight presets resolves to its flag set.
