# Hardware and resource budgets

## Reference machines

| role | reference | notes |
|---|---|---|
| training | 1x NVIDIA H200 (141 GB), 64+ GB host RAM, ~400 GB fast disk for the caches | one run per GPU |
| simulation, benchmark, deployment | 1x NVIDIA RTX 5090 (32 GB), 64 GB host RAM, 1+ TB disk | Isaac Sim 5.1 headless (`OMNI_KIT_ACCEPT_EULA=YES`) |
| robot | Trossen WidowX-AI follower arm + Intel RealSense D405 on the wrist | flat-pad fingers from `assets/gripper` |

Three conda environments are created by `envs/setup.sh` and recorded in
`configs/envs.local.yaml` (the tracked `configs/envs.yaml` is the template): `simretract-sim` (Python 3.11, Isaac Sim 5.1, numpy 1.26),
`simretract-policy` (Python 3.12, torch 2.8 cu128, the pinned lerobot commit,
transformers 5.3, numpy 2.2, pyrealsense2, trossen-arm), `simretract-mask`
(Python 3.11, ultralytics 8.4.64). Processes exchange plain lists and bytes over
localhost TCP so the numpy versions never meet. The optional
`simretract-fastwam` environment (lerobot 0.6.1) serves and trains Fast-WAM only.

System requirements beyond the environments:

- An NVIDIA driver recent enough for Isaac Sim 5.1 with Vulkan support (the
  reference machine runs the 580 series on a Blackwell card). The first Kit
  boot on a machine compiles shaders for several minutes; warm up with
  `simretract-build-assets --usd --compare --accept-eula` (after the local
  asset preparation, `docs/ASSETS.md`) before `scripts/smoke_test.sh sim`,
  whose per-step time caps assume a warm shader cache.
- The policy environment installs the GUI build of OpenCV (the real-robot
  client's `--view` overlay needs it), which loads `libGL`. On a headless
  training host install the system libraries once: `apt install libgl1
  libglib2.0-0`; without them every `import cv2` (cache builder, augmentation,
  trainer) fails.

## Training defaults (H200)

Every preset trains 15,000 steps at batch 16 x accum 4 (effective batch 64).
pi0 needs about 107 GB of GPU memory at that setting; Fast-WAM about 92 GB;
SmolVLA and ACT fit on far smaller cards.

- `--batch 32 --accum 2` does not fit pi0 on 141 GB; on smaller cards use
  `--batch 8 --accum 8` (same effective batch, same schedule).
- `--save-every 5000` writes intermediate snapshots; the default keeps only
  `final/`.
- The data loader reads the memmapped cache with 8 workers; keep the cache on
  local SSD (the strict and recovery caches together take a few hundred GB
  of uncompressed 640x480 frames).
- Run `simretract-train --policy act --smoke` (60 steps, batch 4) before a long
  run to validate the cache and the environment.
- pi0 needs the gated `google/paligemma-3b-pt-224` tokenizer in the Hugging
  Face cache (`HF_HUB_OFFLINE=1` works once it is cached) and the released
  `pi0_base` weights as `--init` (`checkpoints/README.md`).

## Inference and simulation budgets (RTX 5090, 32 GB)

GPU memory per process: the pi0 policy server needs about 15 GB in bf16
(`--half`; host RAM peaks at about 8 GB while loading with the default lean
load, about 30 GB without it). The SmolVLA server needs about 3 GB, the ACT
server about 1 GB, and the Fast-WAM server up to about 23 GB while loading
(`--fastwam-cpu-load` loads on the CPU first and avoids the peak). The SAM 3
mask server needs 5-15 GB of GPU memory and about 15 GB of host RAM (real
robot only). Each Isaac Sim evaluator or recorder lane needs 3-5 GB of GPU
memory and about 4 GB of host RAM.

Rules that follow from the budget:

- One policy server per GPU while an evaluation runs: booting a second pi0
  server next to a running evaluation exceeds the card. The evaluation driver
  keeps one server alive across seeds and frees the port by PID when it exits.
- Recorder lanes: `--parallel 3` when an evaluation or a policy server shares
  the GPU, up to 7 lanes on an otherwise idle card. Lanes that share a
  saturated GPU can hang at the first frame; the lane driver's timeout
  (`--timeout 4800` s per chunk, `--tries 2`) re-records them.
- Disk: raw episodes are about 60 MB each; evaluation outputs are about
  50 MB per seed (review video + frames).
- Every Isaac process is started with `start_new_session=True` and killed as a
  process group on timeout; a force-killed Isaac process can leave shader-cache
  debris that crash-loops later boots, in which case clear the Kit cache
  directory and restart.

## Robot setup

**Fingers.** The flat-pad finger design is adapted from
[Source Robotics' SSG-48 adaptive electric gripper](https://github.com/Source-Robotics/SSG-48-adaptive-electric-gripper),
with modifications to fit the Trossen WidowX-AI. The repository includes
exported visual and collision meshes; editable CAD is not included. See
[gripper files and attribution](../assets/gripper/README.md).

**Arm.** The client talks to the WidowX-AI controller with the raw
`trossen_arm` driver (version 1.10.0): `configure(...)` is communication only,
`get_all_positions()` reads the 7-D state (6 joints in rad + gripper opening
in m), and the arm moves only on explicit `set_all_positions(target,
goal_time, blocking=True)` calls. Do not use the higher-level robot classes for
inference: their connect/disconnect sequences move the arm.

Trossen arms ship with a factory network address. Nothing in this repository
carries an address: set `robot.follower_ip` in `configs/real/default.yaml`
(`CHANGE_ME` until you do; the launcher refuses to start otherwise) to the
address of your controller, and check reachability on the controller's port
before a run. `simretract.real.home` returns the arm to the simulation home
pose (all joints 0, gripper 0.044 m). A normal episode returns the joints home
while retaining the gripper opening; an aborted episode opens it while homing.

**Camera.** The D405 delivers aligned 640x480 colour and z16 depth; depth
units are the device's own scale (0.1 mm on the D405), which the client
queries at runtime and converts to uint16 mm. The frame is centre-cropped
about the measured principal point to the simulation pinhole (58 deg vertical
field of view, fy 433 px) and resized back to 640x480. The client prints the
effective focal length; it should land within a few pixels of 433. Fingers sit
near the D405's minimum range and partially drop out of the depth image, which
is why the gripper mask comes from a lookup table instead of segmentation.

**Gripper mask.** `assets/gmask_lut.npz` holds 45 rendered gripper silhouettes
indexed by opening (0 to 0.044 m); the client picks the nearest opening. It is
sent to the mask server (to suppress finger self-detections) and zeroed before
it reaches a `nogrip` policy. `simretract-build-assets --lut` re-renders it.

**SAM 3 tool mask.** `simretract.real.mask_server` runs the ultralytics SAM 3
video predictor in its own environment (ultralytics is AGPL-3.0; keep it in a
separate process). Weights: `facebook/sam3` (gated, `sam3.pt`, 3.45 GB), pass
the path with `--ckpt`. The text prompt is `metal rake` with detection
confidence 0.2; probe any new prompt on live frames before trusting it. The
predictor is fed continuously by a background thread in the client so its
temporal memory stays fresh, detections that mostly overlap the gripper mask
are dropped, and the selected identity stays locked through detector misses.
Keepalive preserves that identity's memory; it does not reuse an old mask or
switch to another object when the target disappears. Between episodes, a warm
`reset` clears tracking and image state while retaining loaded weights and
text embeddings. The first policy query waits for fresh, stable masks with
usable raw and cropped depth; this startup gate does not require a plane.

**Display.** `--view` opens an OpenCV overlay from the main thread and needs a
display (`DISPLAY`); without one the client warns and continues headless.

## Ports and hosts

| port | process | default in |
|---|---|---|
| 8765 | policy server, real robot | `configs/real/default.yaml`, `simretract-serve --port` |
| 8766 | SAM 3 mask server | `configs/real/default.yaml`, `simretract.real.mask_server --port` |
| 8767 | policy server, simulation benchmark | `configs/eval/sim.yaml`, `simretract-eval-sim --port` |

All servers bind 127.0.0.1. Reserve the configured ports for SimRetract: the
launchers reuse a server whose ping matches the requested checkpoint and
regime, otherwise they may terminate the listener PID on that port. A port
match alone does not establish process ownership. The wire format is an 8-byte
big-endian length prefix followed by a pickle of plain Python lists and bytes
(`simretract.serve.protocol`).

## Bring-up sequence

```bash
scripts/run_real.sh checkpoints/baseline dry --max-ticks 60 --view  # perception and inference; no motion
scripts/run_real.sh checkpoints/baseline --preset conservative go  # first live run, half speed
scripts/run_real.sh checkpoints/baseline go                        # deployment preset
```

The launcher starts or reuses the policy and mask servers itself. Launcher
options (`--preset`, `--plane-top-cut-m`, `--exec-h`, ...)
go before the mode; everything after `dry` / `go` is passed to the client.
The wrapper selects the configured environments; the equivalent installed
entry point is `simretract-run-real --ckpt <ckpt>`. Always start through the
launcher. Without `--preset`, it uses `default`: exec-h/min 50/50, speed 1.0,
max-step 0.06, smooth 0.6, pixel fraction 0, and metric cut 0.01 m. The direct
Python client's defaults match these deployment values. Ctrl-C interrupts
policy execution and returns the arm home while opening the gripper.

## Plane and model crop options

The default removes the upper **1 cm along the visible shaft direction** from
both the plane fit and model RGB-D. The cut follows the shaft toward its upper
image end, rather than a fixed horizontal row. Foreground depth selection
rejects farther background visible through hollow handles, including rings
and slots. Narrow, side-on views use the same depth-based plane fit when the
remaining support permits it; no cylinder or camera-facing normal is assumed.

A failed plane fit supplies a zero normal, while usable cropped RGB-D stays
available. A zero normal does not itself pause segmentation or policy queries.
When 3-D length calibration fails, a depth-scaled projected cut can retain a
usable input; its physical length is approximate on a tilted shaft. Tracking
continues to produce fresh masks independently of these geometry checks.

```bash
scripts/run_real.sh checkpoints/baseline dry                         # default: 1 cm metric crop
scripts/run_real.sh checkpoints/baseline --plane-top-cut-m 0 dry     # no crop
scripts/run_real.sh checkpoints/baseline --plane-top-cut .10 dry     # legacy pixel fraction: plane only
scripts/run_real.sh checkpoints/baseline --plane-top-cut-m .02 dry   # 2 cm metric crop
```

Replace `dry` with `go` for a live run. An explicit `--plane-top-cut` selects
the legacy path unless `--plane-top-cut-m` is also supplied; a positive metric
value takes precedence. A bare `--plane-top-cut-m` means 0.01 m. The `parity`
preset keeps both cuts disabled.
