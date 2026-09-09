# Benchmark protocol

## 1. Simulation benchmark

`simretract-eval-sim` runs a checkpoint closed-loop in Isaac Sim on a fixed
seed set and scores the outcome funnel. The settings below are pinned in
`configs/eval/sim.yaml` and apply to every model row of `docs/ABLATIONS.md`.

### Benchmark seeds

| file | seeds | use |
|---|---|---|
| `data/selections/test_core50.json` | 50 | the fixed benchmark set |

These seeds belong to the frozen 115-seed held-out pool in
`data/selections/test_pool.json`. The entire pool is excluded from training.

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

### First-attempt reporting

The benchmark runs one attempt per seed (`--passes 1`) and reports its first
result. Each attempt uses a fresh Isaac process under a 900 s wall cap. A
process that produces no result can be relaunched once for crash recovery;
the policy server is reused and checked before every seed.

Timeouts and other failed first results remain failures in the benchmark.
The denominator is the full `--score-seeds` selection (the same as `--seeds`
by default), including seeds with missing results. Interrupted runs resume
from their existing result files.

Additional passes are available for diagnostics. They retry non-completed
seeds and archive prior results as `result_<seed>.try<k>.json`. Archives are
preserved across resumed invocations. Reporting still selects the earliest
attempt; use `--score-attempt latest` explicitly for retry results:

```bash
scripts/eval_sim.sh checkpoints/baseline --passes 3 --score-attempt latest --out eval/baseline_retries
```

When rescoring a directory, `simretract.eval.score` selects the earliest
numbered archive, or the current result if no archive exists. A damaged first
result is counted as missing; a later success cannot replace it. `--attempt
latest` selects current results instead. The scorer's optional `--sample-sd`
uses sample SD for diagnostic comparisons.

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
| secured / Grasped | grabbed and `drop_mm < 20` after the hand released; Grasped (%) is `100 * secured / number of seeds` |
| held | `held` is true (the tool stayed in the gripper through the load battery) |
| kept | held and `verdict.grasp_t_mm < 10` and `verdict.grasp_r_deg < 10` (presented pose preserved) |
| timeouts | `outcome == "timeout"` |
| offset | `verdict.grasp_t_mm`: mean +/- population SD [median] over secured rows that carry the field |
| twist | `verdict.grasp_twist_deg`: mean +/- population SD [median] over the same secured rows |

`grasp_t_mm` is the tip displacement and `grasp_r_deg` the rotation of the
tool between its presented pose and its pose after the release settles;
twist is the rotation component about the tool axis, swing the remainder.

Outputs under `--out`: `result_<seed>.json` (+ `.try{k}.json`),
`ep_<seed>/meta.npz`, `ep_<seed>_review.mp4`, `eval_log_<seed>.jsonl`, the
driver ledger, `results.json` (all rows) and `table.md` (the scored row). The
driver prints one `ROW` line at the end. The eight main paper rows and their
50-seed first results are included in [data/evaluation](../data/evaluation/).

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

| preset | exec_h / exec_min | settle | speed | max_step | smooth | metric cut (m) | use |
|---|---|---|---|---|---|---|---|
| `default` | 50 / 50 | 0 | 1.0 | 0.06 | 0.6 | 0.01 | whole chunk per query with bad-observation hold guard |
| `parity` | 20 / 20 | 0 | 1.0 | 0.06 | 0.45 | 0 | uncut input and 20-step execution |
| `conservative` | 10 / 5 | 0.25 | 0.5 | 0.03 | 0.45 | 0.01 | slower motion and shorter execution near the tool |

Common client settings: `min_tool_px 300` and `hold_max 2.0` (hold the pose
for up to 2 s while the tool mask is too small or has too little valid depth),
`close_cmd 0.020` with the same stall rule as the simulation (stop the model,
freeze the arm, firm-close the gripper, hold until ENTER), `max_ticks 451`,
`home_speed 0.6` rad/s ramps to and from the simulation home pose
(all joints 0, gripper 0.044 m). Fast-WAM checkpoints use `exec_h 48`.

The default `--plane-top-cut-m 0.01` removes 1 cm along the tool shaft from
both plane estimation and model RGB-D. All presets use pixel fraction 0.
`--plane-top-cut-m 0` disables the metric crop; `--plane-top-cut f` selects
the optional pixel-fraction crop for plane estimation only. See
[Plane and model crop options](HARDWARE.md#plane-and-model-crop-options).

A real run writes `settings.json`, the per-query masked inputs, `log.npz`,
`review_view.mp4` and `result.json` (`closed`, `closed_tick`, `aborted`,
`ticks`, `queries`, `final_qpos`) under the run directory;
`simretract.real.tally` counts `closed` outcomes per model. Whether a closed
gripper became a good hand-over is judged by the operator, not by the code.
