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

| policy | batch x accum | steps | GPU memory | time per step | wall time | checkpoint |
|---|---|---|---|---|---|---|
| pi0 (all pi0 rows) | 16 x 4 | 15,000 | ~107 GB | 1.6-1.8 s | ~7.5 h | 16.1 GB |
| SmolVLA (full fine-tune) | 16 x 4 | 15,000 | | | ~2.6 h | 1.2 GB |
| ACT (from scratch) | 16 x 4 | 15,000 | | 0.26 s | ~1.1 h | 0.2 GB |
| Fast-WAM (optional) | 16 x 4 | 15,000 | ~92 GB | ~5 s | ~21 h | 12 GB |

- The effective batch is 64. `--batch 32 --accum 2` does not fit pi0 on 141 GB;
  on smaller cards use `--batch 8 --accum 8` (same effective batch, same
  schedule, roughly the same wall time per step of progress).
- `--save-every 5000` writes intermediate snapshots (16 GB each for pi0); the
  default keeps only `final/`.
- The data loader reads the memmapped cache with 8 workers; keep the cache on
  local SSD. Cache sizes: strict ~130 GB, recovery ~200 GB (uncompressed
  640x480 frames); plus ~1.5 GB of plane sidecars, state and action arrays.
- Run `simretract-train --policy act --smoke` (60 steps, batch 4) before a long
  run to validate the cache and the environment.
- pi0 needs the gated `google/paligemma-3b-pt-224` tokenizer in the Hugging
  Face cache (`HF_HUB_OFFLINE=1` works once it is cached) and the released
  `pi0_base` weights as `--init` (`checkpoints/README.md`).

## Inference and simulation budgets (RTX 5090, 32 GB)

| process | GPU | host RAM | notes |
|---|---|---|---|
| policy server, pi0 bf16 (`--half`) | ~15 GB | ~8 GB peak while loading (default lean load), ~4 GB steady | without lean load the fp32 instantiation peaks at ~30 GB host RAM |
| policy server, SmolVLA bf16 | ~3 GB | | |
| policy server, ACT fp32 | ~1 GB | | |
| policy server, Fast-WAM | up to ~23 GB while loading | | `--fastwam-cpu-load` loads on the CPU first and avoids the peak |
| SAM 3 mask server | 5-15 GB | ~15 GB | real robot only |
| Isaac Sim evaluator, one seed | 3-5 GB | ~4 GB | 6-8 min per seed including release, settling and the load battery |
| Isaac Sim recorder lane | 3-5 GB | ~4 GB | 5-8 min per attempted seed |

Rules that follow from the budget:

- One policy server per GPU while an evaluation runs: booting a second pi0
  server next to a running evaluation exceeds the card. The evaluation driver
  keeps one server alive across seeds and frees the port by PID when it exits.
- Recorder lanes: `--parallel 3` when an evaluation or a policy server shares
  the GPU, up to 7 lanes on an otherwise idle card. Lanes that share a
  saturated GPU can hang at the first frame; the lane driver's timeout
  (`--timeout 4800` s per chunk, `--tries 2`) re-records them.
- Disk: raw episodes are ~60 MB each (~27 GB for strict-500, ~110 GB for the
  full recovery set); evaluation outputs are ~50 MB per seed (review video +
  frames).
- Every Isaac process is started with `start_new_session=True` and killed as a
  process group on timeout; a force-killed Isaac process can leave shader-cache
  debris that crash-loops later boots, in which case clear the Kit cache
  directory and restart.

## Robot setup

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
pose (all joints 0, gripper 0.044 m); every episode starts and ends there
because every training episode does.

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
are dropped, and a `reset` op clears the tracker between episodes.

**Display.** `--view` opens an OpenCV overlay from the main thread and needs a
display (`DISPLAY`); without one the client warns and continues headless.

## Ports and hosts

| port | process | default in |
|---|---|---|
| 8765 | policy server, real robot | `configs/real/default.yaml`, `simretract-serve --port` |
| 8766 | SAM 3 mask server | `configs/real/default.yaml`, `simretract.real.mask_server --port` |
| 8767 | policy server, simulation benchmark | `configs/eval/sim.yaml`, `simretract-eval-sim --port` |

All servers bind 127.0.0.1. The launchers reuse a live server only when its
ping reports the requested checkpoint and regime, and they kill only the
processes they own (by PID on their own ports). The wire format is an 8-byte
big-endian length prefix followed by a pickle of plain Python lists and bytes
(`simretract.serve.protocol`).

## Bring-up sequence

```bash
conda activate simretract-policy
simretract-serve --ckpt checkpoints/baseline --port 8765 --half          # 1. policy server
# second shell: conda activate simretract-mask
python -m simretract.real.mask_server --ckpt checkpoints/sam3/sam3.pt   # 2. mask server
simretract-run-real --ckpt checkpoints/baseline dry --max-ticks 60 --view # 3. perception only, no motion
simretract-run-real --ckpt checkpoints/baseline --preset conservative go  # 4. first live run, half speed
simretract-run-real --ckpt checkpoints/baseline go                       # 5. deployment preset
```

`simretract-run-real` performs steps 1 and 2 itself when the servers are not
already up. Launcher options (`--preset`, `--plane-top-cut`, `--exec-h`, ...)
go before the mode; everything after `dry` / `go` is passed to the client.
`scripts/run_real.sh <ckpt> [launcher flags] go|dry [client flags]` is the
same command from any shell; without `--preset` it uses the deployment preset
(`default`, plane top cut 0.10). Always start the client through the
launcher: `python -m simretract.real.client` without flags runs with the
deployment preset's values (exec-h 50/50, speed 0.8, max-step 0.06, smooth 0.6,
top cut 0.10), not with the `conservative` bring-up values. Keep Ctrl-C under
the operator's hand: it stops commanding and returns the arm home while
opening the gripper.
