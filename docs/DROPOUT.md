# Late-trajectory input dropout and plane deterioration

The dropout rule is part of the baseline recipe and of every released
checkpoint. Everything else in the recipe (data, inputs, augmentation v4.2,
normalization, schedule, base weights) is unchanged by it; `--no-dropout`
disables it.

## 1. Where it lives

| piece | module | flags / fields |
|---|---|---|
| training-time rule | `simretract/train/train.py` (dataset item construction and the plane perturbation helper) | `--drop-mask-p`, `--drop-plane-p`, `--drop-start`, `--plane-sig`, `--plane-sig-end`, `--plane-glitch-p`, `--plane-glitch-end`, `--plane-rand-frac` (env fallbacks `DROP_MASK_P`, `DROP_PLANE_P`, `DROP_START`, `PLANE_SIG`, `PLANE_SIG_END`, `PLANE_GLITCH_P`, `PLANE_GLITCH_END`, `PLANE_RAND_FRAC`) |
| recorded in the checkpoint | `norm.json["drop_aug"]` | `mask_p`, `plane_p`, `start`, `plane_sig_end`, `plane_glitch_end`, `plane_rand_frac` |
| serving gate (simulation and real) | `simretract/serve/policy_server.py` | `--drop-px` (env `DROP_PX`, default 300) |
| real-client tool-lost threshold | `simretract/real/client.py` | `--min-tool-px` (default 300) |
| preset switch | `simretract-train --dropout / --no-dropout` | default on |

## 2. The rule at training time (per sampled frame)

Let f in [0, 1] be the frame's position in the kept trajectory (0 = episode
start, 1 = gripper close). The ramp weight is

    w = max(0, (f - drop_start) / (1 - drop_start))        drop_start = 0.5

so w is 0 for the first half of the approach and rises linearly to 1 at the
close. The ramp is applied only when augmentation is on (`--ablation no_aug`
disables it together with everything else).

(a) **Blind drop** ("lost mask"), probability `drop_mask_p * w` with
`drop_mask_p = 0.25`: both image streams (masked RGB, masked depth) are set to
black AND the plane normal in the state vector is zeroed. On the robot a lost
tool mask means no masked image and no plane fit at the same time, so the two
never disappear independently.

(b) **Plane-only drop**: disabled (`drop_plane_p = 0`). The mask is never
present with a zero plane.

(c) **Plane deterioration while the mask is present** (always on, ramped by w):

- base noise (w = 0), matching the measured SAM 3 + D405 plane noise
  (frame-to-frame median about 2 deg, 90th percentile about 6 deg): a Gaussian
  rotation of the normal about a random perpendicular axis with
  sigma = `plane_sig` = 3 deg, plus a `plane_glitch_p` = 3% chance of a large
  error, uniform in 10-45 deg;
- ramp (w -> 1): sigma rises linearly 3 -> `plane_sig_end` = 8 deg; the glitch
  probability rises linearly 0.03 -> `plane_glitch_end` = 0.25; and a glitch
  becomes, with probability `plane_rand_frac` = 0.5 (only when w > 0), a fully
  random direction (sign flipped toward the camera) instead of a 10-45 deg
  rotation.

The Gaussian part alone has median error 0.674 sigma: about 2.0 deg before
the ramp, 3.7 deg at 75% of the trajectory (sigma 5.5) and 5.4 deg at the
close (sigma 8), before glitches.

## 3. The rule at inference (mirrors the training coupling)

For a checkpoint whose `norm.json` carries `drop_aug.mask_p > 0`, the policy
server zeroes the plane normal whenever the tool mask has fewer than `--drop-px`
(300) pixels - the same threshold as the real client's `--min-tool-px`. Nothing
else changes at test time: no noise is injected, and the images are whatever the
camera and mask pipeline produce (black masked streams when the mask is empty).
The simulation benchmark uses the ground-truth render mask, so the gate matters
mainly on the real arm.

## 4. Values

| flag | dropout rows (default) | `--no-dropout` |
|---|---|---|
| `--drop-mask-p` | 0.25 | 0 |
| `--drop-plane-p` | 0 | 0 |
| `--drop-start` | 0.5 | 0.5 |
| `--plane-sig` / `--plane-sig-end` | 3 / 8 deg | 3 / 3 deg |
| `--plane-glitch-p` / `--plane-glitch-end` | 0.03 / 0.25 | 0.03 / 0.03 |
| `--plane-rand-frac` | 0.5 | 0 |
| server `--drop-px` | 300 | 300 |

With `--no-dropout` the plane keeps its base noise (constant sigma 3 deg, 3%
glitches) and the blind drop is off, which is the pre-dropout recipe. The unit
tests check the ramp statistics (median error at w = 0, 0.5 and 1) against
these values.
