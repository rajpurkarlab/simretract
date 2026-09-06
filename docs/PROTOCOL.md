# Benchmark protocol

## 1. Simulation benchmark

`simretract-eval-sim` runs a checkpoint closed-loop in Isaac Sim on a fixed
seed set and scores the outcome funnel. The settings below are pinned in
`configs/eval/sim.yaml` and apply to every model row of `docs/ABLATIONS.md`.

### Seed sets

| file | seeds | use |
|---|---|---|
| `data/selections/test_pool.json` | 115 | the frozen held-out pool; never trained on |
| `data/selections/test_core40.json` | 40 | 4x4-stratified core of the pool (presented x and z) |
| `data/selections/test_extra10.json` | 10 | pool seeds that fill the sparsest cells |
| `data/selections/test_core50.json` | 50 | **the benchmark set** = core-40 + extra-10 |

Each seed rebuilds the recorder-identical scene: presented pose drawn from
x U(0.35, 0.50), y U(-0.12, 0.12), z U(0.15, 0.30) m, tilt U(0, 1.4 rad) about
a random horizontal axis, spin U(-pi, pi), tool mass U(60, 160) g, friction
0.4, lighting and occluder hand from seed-derived streams. The scripted grasp
search is skipped (`skip_search 1`): every benchmark seed is known-feasible.

### Hand style

The occluder hand has two styles. `--hand-style test` (the default and the
benchmark protocol) uses fist probability 0.5 and hand height U(-0.12, -0.05) m
along the tool axis; `--hand-style train` uses the generation recipe's values
(fist probability 0.75, height U(-0.12, -0.065)). The benchmark deliberately
keeps the test style so the seed set matches the pool it was frozen from.

### Control loop

| setting | value | meaning |
|---|---|---|
| `exec_h` | 50 (pi0, SmolVLA, ACT, pi0.5, pi0-FAST), 48 (Fast-WAM) | actions executed from each 50-step chunk before the next query; `--exec-h auto` picks by checkpoint type |
| `smooth` | 0.45 | EMA on the 7-D target: `a = (1 - smooth) * a + smooth * a_prev` |
| `max_steps` | 450 | control ticks (25 Hz, 18 s) before the episode is a `timeout` |
| `half` | 1 | bf16 serving for pi0 / pi0.5 / SmolVLA (ACT, Fast-WAM and pi0-FAST manage their own precision) |
| `plane_filter` | 1 | 5x5 median blur + 5-MAD depth trim before the plane fit |
| `plane_top_cut` | **0** | no top cut in simulation (see section 2 for the real-run value) |
| `drop_px` | 300 | dropout checkpoints: the plane normal is zeroed when the tool mask has fewer pixels |
| `mu` | 0.4 | tool contact friction |

Gripper close is detected when the left carriage moves less than 0.8 mm per
frame for 12 consecutive frames with both pads in contact (or, at
`max_steps`, when the gripper is nearly closed with both pads in contact).
After the close the operator hand releases, the tool settles, the deviation
from the presented pose is measured, and the load battery runs on camera.

The evaluator sends `plane_top_cut: 0.0` with every query. The server's ping
returns its regime (checkpoint, half, plane filter, drop threshold, top cut,
nogrip, type, chunk padding) and the driver refuses to run against a server
whose regime does not match the configuration.

### Three-attempt rule

The driver runs one fresh Isaac process per seed under a 900 s wall cap and
keeps the policy server alive across seeds (liveness check and reboot before
every seed, two launches per seed when a launch produces no result file).

- Pass 1 runs every seed in `--seeds`.
- Pass 2 re-runs only the seeds whose result is not `completed` (or missing);
  the previous result is kept as `result_<seed>.try1.json`.
- Pass 3 does the same once more (`result_<seed>.try2.json`).

`--passes 3` is the protocol. Completed seeds are never re-run, and a seed that
completes on any attempt counts as completed; every attempt stays on disk. The
driver is resume-safe: an interrupted run continues from the result files. The
scored set is `--score-seeds` (default: the same file as `--seeds`); the
denominator is always the size of that set, so seeds without a result count as
failures.

### Funnel and statistics

Every seed ends with one `result_<seed>.json` (fields: `seed`, `outcome`,
`closed_at`, `queries`, `ticks`, `drop_mm`, `held`, `strict`, `verdict`
{`grasp_t_mm`, `grasp_r_deg`, `grasp_swing_deg`, `grasp_twist_deg`,
`left_gripper`, `strict_pass`, `max_world_t_mm`, `max_world_r_deg`,
`max_slip_mm`}, scene metadata). Outcomes are `completed`, `timeout`,
`no_grasp`, `scene_reject` and `server_error:<message>`.

`simretract.eval.score` defines the row:

| column | definition |
|---|---|
| grabbed | `outcome == "completed"` |
| secured | grabbed and `drop_mm <= 20` (the tool did not fall when the hand released) |
| held | `held` is true (the tool stayed in the gripper through the load battery) |
| kept | held and `verdict.grasp_t_mm < 10` and `verdict.grasp_r_deg < 10` (presented pose preserved) |
| timeouts | `outcome == "timeout"` |
| offset | `verdict.grasp_t_mm`: mean +/- sd [median] over completed rows that carry the field |
| twist | `verdict.grasp_twist_deg`: mean +/- sd [median] over the same rows |

`grasp_t_mm` is the tip displacement and `grasp_r_deg` the rotation of the
tool between its presented pose and its pose after the release settles;
twist is the rotation component about the tool axis, swing the remainder.

Outputs under `--out`: `result_<seed>.json` (+ `.try{k}.json`),
`ep_<seed>/meta.npz`, `ep_<seed>_review.mp4`, `eval_log_<seed>.jsonl`, the
driver ledger, `results.json` (all rows) and `table.md` (the scored row). The
driver prints one `ROW` line at the end.

### Nondeterminism

Neither generation nor evaluation is bit-reproducible: PhysX contact
resolution, GPU rendering and bf16 inference all vary between runs, and the
policy's own chunks depend on those observations. Two evaluations of the same
checkpoint on the same 50 seeds can differ by several seeds in every funnel
column, and rollout variance on small seed sets is large. Report the full 50
seeds with the three-attempt rule, and compare rows only under identical
settings (in particular the same `exec_h`; rows recorded with `exec_h 20` are
not comparable to `exec_h 50` rows). Training is not bit-reproducible either
(unseeded sampler and augmentation), so a retrained row is expected to land
near, not on, an earlier evaluation of the same preset.

## 2. Real-robot runs

`simretract-run-real --ckpt <dir> --preset <name> go|dry` boots or reuses the
policy server (port 8765) and the SAM 3 mask server (port 8766), verifies both
with a ping, and runs the client. The client mirrors the simulation loop:
every `exec_h` ticks it captures a D405 RGB-D frame cropped to the simulation
pinhole, fetches the tool mask, indexes the gripper-mask LUT by the measured
opening, reads the joint positions, and queries the server; every 25 Hz tick
it executes the next action with the EMA, a per-tick clamp (`max_step` rad per
joint, 5 mm on the gripper) and a blocking move of `0.04 / speed` seconds.

Presets (`configs/real/default.yaml`):

| preset | exec_h / exec_min | settle | speed | max_step | smooth | plane_top_cut | use |
|---|---|---|---|---|---|---|---|
| `default` | 50 / 50 | 0 | 0.8 | 0.06 | 0.6 | **0.10** | deployment default: whole chunk per query, bad-observation hold guard |
| `parity` | 20 / 20 | 0 | 1.0 | 0.06 | 0.45 | 0.0 | mirrors the simulation control loop at exec_h 20 (not the benchmark regime) |
| `conservative` | 10 / 10 | 0.25 | 0.5 | 0.03 | 0.45 | 0.10 | stop-and-go bring-up regime |

Common client settings: `min_tool_px 300` and `hold_max 2.0` (hold the pose
for up to 2 s while the tool mask is too small or has too little valid depth),
`close_cmd 0.020` with the same stall rule as the simulation (stop the model,
freeze the arm, firm-close the gripper, hold until ENTER), `max_ticks 451`,
`home_speed 0.6` rad/s ramps to and from the simulation home pose
(all joints 0, gripper 0.044 m). Fast-WAM checkpoints use `exec_h 48`.

Plane top cut: `--plane-top-cut f` removes the top fraction `f` of the tool's
pixel length (along its principal axis) before the plane fit, guarded so that
a fit that swings more than 20 deg from the uncut fit is discarded in favour
of the uncut plane. The value travels with every query; the server's own
default is 0. The reported real-robot bench runs used 0 (most runs) or 0.15
(early runs). 0.10 is the deployment default shipped in `default` because it
removes the operator's fingers from the fit without the swing guard
triggering on narrow presentations; it is not a benchmark-protocol value.

A real run writes `settings.json`, the per-query masked inputs, `log.npz`,
`review_view.mp4` and `result.json` (`closed`, `closed_tick`, `aborted`,
`ticks`, `queries`, `final_qpos`) under the run directory;
`simretract.real.tally` counts `closed` outcomes per model. Whether a closed
gripper became a good hand-over is judged by the operator, not by the code.
