# Data format

This document is the contract between the recorders (`simretract.sim.recorder`,
`simretract.sim.recovery_recorder`), the cache builder (`simretract.data.cache`),
the plane sidecar (`simretract.data.plane_sidecar`), the trainer, and the hub
download helpers (`simretract.data.hub`).

## 1. Raw episode directories

A raw set is a directory of episode directories. Strict episodes are named by
seed, recovery episodes by seed and perturbation draw:

```
<set>/
  ep_<seed>/                 strict episode           (e.g. ep_400127)
  ep_<seed>_p<draw>/         recovery episode         (e.g. ep_400127_p2)
  ep_<seed>.filtered         marker: deterministic reject, do not retry
  ep_<seed>_p<draw>.wedged   marker: the recovery draw stalled, do not retry
  recorder_log_<first>.jsonl one line per attempted seed / virtual index of a worker
  review/                    optional 2x2 review videos (every N-th keep with --review-every N)
```

A directory whose `meta.npz` exists is a kept episode. The lane driver treats
`meta.npz`, `.filtered` and `.wedged` as "done" markers when it resumes.

Each episode directory holds per-frame PNGs and one `meta.npz`:

| file | format | content |
|---|---|---|
| `rgb_%04d.png` | 640x480 RGB8 | wrist camera colour |
| `depth_%04d.png` | 640x480 uint16 | depth in millimetres (distance to image plane) |
| `mask_%04d.png` | 640x480 uint8 0/255 | tool mask (semantic class `tool`) |
| `gmask_%04d.png` | 640x480 uint8 0/255 | gripper mask (link 6 + carriages + fingers) |
| `omask_%04d.png` | 640x480 uint8 0/255 | occluder-hand mask (not consumed downstream) |
| `meta.npz` | numpy archive | trajectory and labels, see below |

Frames are captured at 25 Hz of simulation time (500 Hz physics, control
decimation 20). The wrist camera is the D405 pinhole used everywhere in the
pipeline: 640x480, 58 deg vertical field of view, fx = fy = 240 / tan(29 deg)
= 433 px, cx = 320, cy = 240.

`meta.npz` keys:

| key | shape / type | meaning |
|---|---|---|
| `states` | (T, 7) float32 | joint positions: 6 arm joints (rad) + gripper opening (m) |
| `actions` | (T, 7) float32 | commanded joint-position targets, same convention |
| `tool_pos_t` | (T, 3) float32 | tool position per frame (world frame) |
| `tool_quat_t` | (T, 4) float32 | tool orientation per frame, quaternion (w, x, y, z) |
| `grasp_pose` | (7,) float32 | executed grasp: position (3) + quaternion (w, x, y, z) of the grasp frame |
| `seed` | int | scene seed (all scene randomization derives from it) |
| `task` | str | the fixed language prompt `take over the tool from the operator and hold its pose` |
| `fps` | int | 25 |
| `tool` | str | `retractor_rake` |
| `mass_g` | float | tool mass drawn from U(60, 160) g |
| `verdict` | JSON string | `{"battery_skipped": true}` for the production recipe; battery verdicts (`strict_pass`, `left_gripper`, ...) when the load battery runs |
| `info` | JSON string | grasp bookkeeping, see below |
| `lighting` | JSON string | per-episode light randomization (`dome_i`, `warmth`, `key_i`, `key_elev`, `key_azim`, `n_fill`) |
| `occ_z` | float | occluder-hand height along the tool axis (m) |
| `tilt_deg` | float | presented tilt of the tool (deg) |

`info` fields:

| field | meaning |
|---|---|
| `gap_mm`, `reach_err_mm`, `center_off_mm` | grasp geometry at the close (pad gap, reach error, centering) |
| `drop_mm` | tool drop after the operator hand releases (keep rule: < 20 mm) |
| `pose_t_mm`, `pose_r_deg` | presented vs post-release tool pose (keep rule: < 10 mm and < 10 deg) |
| `grasp_w` (3), `R_grasp` (3x3) | grasp position and rotation in the world frame; reused as a "stored winner" by later recordings |
| `perturb_win` | recovery only: `{"start", "end"}` frame indices of the perturbation; training starts at `end + 1` |
| `perturb` | recovery only: `{"frac", "dpos", "drot", "grip"}` - where along the reach the perturbation started, and its offsets |
| `switched_grasp` | recovery only: whether the recovery re-planned to a different grasp |

## 2. Frame cache

`simretract-build-cache --set strict|recovery --raw DIR --out CACHE` converts a
raw set into flat memmaps that the trainer indexes by global frame number. Both
sets share one cache directory; `meta.json` is merged per set.

```
CACHE/
  meta.json                         {"<set>": {"N": frames, "episodes": [[a0, a1], ...], "prekept": true, "flags": {...}}}
  <set>_rgb.npy                     (N, 480, 640, 3) uint8
  <set>_dep.npy                     (N, 480, 640)    uint16 mm
  <set>_msk.npy                     (N, 480, 640)    uint8 tool mask
  <set>_gmk.npy                     (N, 480, 640)    uint8 gripper mask (sparse / all zero when skip_gmk)
  <set>_state.npy                   (N, 7)  float32
  <set>_action.npy                  (N, 7)  float32
  <set>_gpose.npy                   (N, 7)  float32 grasp_pose broadcast per frame
  <set>_verdicts.json               {"<episode index>": {"seed", "strict", "held", ...}}
  <set>_plane.npy                   (N, 3)  float32 unit plane normal, camera frame, pointing toward the camera
  <set>_plane_conf.npy              (N,)    float32 planarity confidence (s2/s3); 0 when < 200 valid points
```

`episodes` lists half-open frame ranges `[a0, a1)` per stored episode, in
directory-sorted order. The production flags (recorded under `flags`) are:

| flag | strict | recovery | effect |
|---|---|---|---|
| `pretrim` | 1 | 1 | store only frames before the gripper-close cutoff (first frame within 1e-4 m of the final opening, plus a 25-frame margin) |
| `keep_only` | 1 | 1 | store only arc-length-kept frames (joint quantum 2.5e-3 rad, gripper quantum 2.5e-4 m); sets `prekept` so the trainer skips resampling |
| `seed_min` | 200000 | 200000 | ignore episodes whose seed is below this floor (legacy env `ERA_MIN`) |
| `skip_gmk` | 1 | 1 | do not write the gripper-mask memmap (no production row consumes it) |
| `perturb_trim` | 0 | 1 | start at `info.perturb_win.end + 1`; skip episodes without a usable window |

Episodes whose seeds appear in `data/selections/test_pool.json` are excluded
mechanically (default on, logged). Episodes whose first probe frames have an
empty tool mask are skipped. After writing, 20 random frames are decoded again
from the PNGs and compared bitwise. Expected frame counts and array
fingerprints for the released sets are in `configs/data.yaml`
(`simretract-build-cache --verify` compares them).

## 3. Selection files (`data/selections/`)

All selection files are plain JSON with a `name`, the membership, numeric
metadata and a neutral `rule` string; `note` and `purpose` are further
free-text descriptions. Seeds are integers; episode ids are directory names.
`simretract-assemble` writes the training selections with exactly the keys
listed here.

| file | keys | content |
|---|---|---|
| `strict_500.json` | `name`, `seeds` (500), `per_bin` (125), `bins_cm` (4 labels), `rule`, `recipe` (the scalar knobs of `configs/generation/strict.yaml` the set was recorded with), `grasp_z_cm` {seed: grasp height in cm}, `trimmed_out` (seeds dropped from over-full bins), `short_bins` {bin: missing count}, `n_recorded_total` | the strict training set: 125 seeds per 2.5 cm grasp-height bin over -5..+5 cm along the tool axis |
| `recovery_train_2000.json` | `name`, `rule`, `draws_per_seed` (4), `train_episodes` (1954 ids), `surplus` (39 ids), `perturb_frac` {episode: `info.perturb.frac`} | exact-4 selection: the lowest four draw indices per strict seed that produced a keep |
| `recovery_far1000.json` | `name`, `rule`, `episodes` (1000 ids), `frac_max`, `n`, `seeds_covered` | the 1000 training episodes with the smallest `info.perturb.frac` (perturbed farthest from the grasp); ties broken by sorted name |
| `test_pool.json` | `name`, `purpose`, `rule`, `count` (115), `seeds`, `episodes` [{seed, strict, x, y, z, tilt_deg}] | the frozen held-out pool; every seed in it is excluded from every cache |
| `test_core40.json` | `name`, `note`, `seeds` (40) | 4x4-stratified core of the pool (presented x and z) |
| `test_extra10.json` | `name`, `note`, `seeds` (10) | 10 further pool seeds that fill the sparsest cells |
| `test_core50.json` | `name`, `note`, `seeds` (50) | core-40 + extra-10: the benchmark seed set |

The half-data preset writes two files beside its run: `<run>_subset.json`
(`{"strict": [episode indices], "recovery": [episode indices]}`, the filter the
trainer applies before its train/val split) and `<run>_selection.json`
(`rule`, `strict_seeds`, `far_seed_coverage`). The rule is deterministic (RNG
seed 0): 250 strict seeds stratified by grasp-height bin, and the far-1000
episodes of those seeds trimmed round-robin from the seeds with the most draws
(or topped up from unselected seeds, rarest bin first) to exactly 500.

## 4. Hub manifest contract (raw sets)

A released raw set is a hub dataset repository holding tar shards of episode
directories plus a manifest:

```
manifest.json          {"n_episodes": N, "shard_n": 100, "shards": {"shard_0000.tar": ["ep_...", ...], ...}}
shard_0000.tar ...     each tar contains whole episode directories at its root
meta.tar               recorder logs and markers (optional)
<selection>.json       the selection file(s) the set was assembled with
```

`simretract-download data --set strict` downloads every shard and extracts all
episodes; `--set recovery` reads `recovery_far1000.json`, downloads only the
shards that contain selected members, and extracts only those directories.
The download verifies the episode count against the manifest and the selection
(500 strict, 1000 recovery) before it reports success.

## 5. Checkpoint directories

A checkpoint directory is what the trainer writes to `runs/<name>/final/`, what
`simretract-download checkpoints` fetches, and what the server, the evaluator
and the real client take as `--ckpt` (a run directory whose `final/` holds the
files is accepted too; the trainer also links the three files at the run
root, so `runs/<name>` and `runs/<name>/final` name the same checkpoint):

```
<ckpt>/
  config.json          policy configuration (type: pi0 | smolvla | act | pi05 | pi0_fast | fastwam)
  model.safetensors    weights
  norm.json            normalization stats and the input contract (below)
  context.pt           Fast-WAM only: precomputed language context
```

`norm.json` fields: `state_mean`/`state_std` (10-D for `qpos_plane`, 7-D for
`qpos`), `action_mean`/`action_std` (7-D), `normalize` (`zscore`), `float_depth`,
`nogrip`, `input_mode` (`masked_rgbd`, `masked_rgb_nodepth`, `full_rgbd`),
`state_mode` (`qpos_plane`, `qpos`), `no_aug`, `policy`, `chunk_h`, `video_t`,
`video_ratio`, `drop_aug` (`mask_p`, `plane_p`, `start`, `plane_sig_end`,
`plane_glitch_end`, `plane_rand_frac`) and `train_regime` (policy, data,
ablation, dropout knobs, `smolvla_full`, `act_opt`). The server reads
everything it needs from this file; clients send raw observations only.
