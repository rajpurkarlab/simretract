# Third-party notices

This repository builds on the following external software and weights. None of
them is vendored except where stated; each is fetched at setup time under its own
licence.

## Software

| Component | Use | Licence | How it is obtained |
|---|---|---|---|
| NVIDIA Isaac Sim 5.1 (`isaacsim` wheels) | simulation, rendering, PhysX | NVIDIA Omniverse licence (EULA accepted by the user via `OMNI_KIT_ACCEPT_EULA=YES`) | `envs/sim.txt` from the NVIDIA package index |
| `trossen_ai_isaac` (Trossen Robotics) | WidowX-AI follower USD and joint controller | The scripts carry BSD-3-Clause headers (copyright Trossen Robotics); the checkout ships no LICENSE file and states no terms for the USD assets, so the asset terms are unverified (`assets/gmask_lut.npz` is rendered from that USD) | git submodule `third_party/trossen_ai_isaac` |
| `trossen-arm` 1.10.0 | real-arm driver | BSD-3-Clause (Trossen Robotics); bundles BSD-licensed components listed in its `LICENSE-3RD-PARTY.txt` | `envs/policy.txt` |
| `lerobot` (Hugging Face) | pi0 / pi0.5 / pi0-FAST / SmolVLA / ACT policy implementations | Apache-2.0 | pinned commit, `envs/LEROBOT_PIN.md` |
| `transformers`, `safetensors`, `huggingface_hub`, `accelerate`, `diffusers` | model code and weight I/O | Apache-2.0 | `envs/policy.txt` |
| `torch`, `torchvision` | training and inference | BSD-3-Clause | PyTorch cu128 index |
| `ultralytics` | SAM 3 predictor for the real-robot mask server | AGPL-3.0 (separate process and environment, `envs/mask.txt`) | PyPI |
| `pyrealsense2` | Intel RealSense D405 capture | Apache-2.0 | PyPI |
| `trimesh`, `scipy`, `numpy`, `opencv-python`, `pillow`, `imageio`, `imageio-ffmpeg`, `av`, `matplotlib`, `pyyaml`, `einops`, `num2words`, `draccus`, `gymnasium` | utilities | MIT / BSD / Apache-2.0 / LGPL (imageio-ffmpeg bundles an LGPL ffmpeg build) | PyPI |
| `coacd` (optional) | convex decomposition in `simretract-build-assets` (the local asset preparation, `docs/ASSETS.md`) | MIT | PyPI, `coacd==1.0.11` (not installed by default, see `envs/sim.txt`) |
| `pyassimp` (optional) | FBX / multi-object scene import in `simretract-build-assets` | BSD-3-Clause (Open Asset Import Library) | PyPI plus the `assimp` shared library (conda-forge or a distribution package; not installed by default) |

## Weights and tokenizers (downloaded, never redistributed here)

| Asset | Hub id (`configs/hub.yaml`) | Licence / gating |
|---|---|---|
| pi0 base weights | `<org>/simretract-checkpoints/pi0_base`: a re-hosted copy of an earlier hub revision of `lerobot/pi0_base` (lerobot's conversion of the openpi pi0 base), kept because the current hub revision carries different tensors | Apache-2.0 (the lerobot conversion) and the Gemma Terms of Use: the weights embed a PaliGemma backbone (`checkpoints/README.md`) |
| PaliGemma tokenizer | `google/paligemma-3b-pt-224` | Gemma Terms of Use (gated: accept on the hub, then `hf auth login`) |
| SmolVLA base | `lerobot/smolvla_base` | Apache-2.0 |
| SmolVLM2 tokenizer | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | Apache-2.0 |
| SAM 3 weights (`sam3.pt`) | `facebook/sam3` | SAM 3 License (gated; redistribution is not permitted, so the user downloads the file and passes it with `--ckpt`) |
| ResNet-18 ImageNet weights | torchvision `ResNet18_Weights.IMAGENET1K_V1` | BSD-3-Clause |
| Fast-WAM base (optional) | `lerobot/fastwam_base` | see the hub card |
| pi0.5 / pi0-FAST bases (optional) | `lerobot/pi05_base`, `lerobot/pi0fast-base`, `lerobot/fast-action-tokenizer` | see the hub cards |

## 3D assets

| Asset | Use | Licence | How it is obtained |
|---|---|---|---|
| "Operating Volkman Retractor Surgical Instrument" by HQ3DMOD (Fab), node `ovrsi_retractor_body` of `vretractor.fbx` | the default source of the tool: converted with the shipped recipe (`configs/assets/retractor_rake.yaml`) it rebuilds the released simulation mesh (same vertices and faces to the exports' single-precision rounding), the collision parts, the USD and the grasp candidates | Fab Standard License (<https://www.fab.com/eula>); the listing (<https://www.fab.com/listings/28243153-9617-4d27-a1c1-779d22e6f61a>) states "Allows usage with AI: Yes"; not redistributed, no reconstructable derivative tracked | the user's own purchase, dropped under `assets/source/vretractor/` as downloaded and converted locally by `simretract-build-assets` (`docs/ASSETS.md`) |
| "Set of 7 surgical instruments" (CGTrader), rake retractor node `so7si_retractor_2_body` | the export the released datasets were recorded with and the released checkpoints were trained on (the same model as the Fab listing; `meta.json` describes this build) | the CGTrader listing's terms (<https://www.cgtrader.com/3d-models/science/medical/set-of-7-surgical-instruments>); not redistributed, no reconstructable derivative tracked | the user's own purchase; recognised by the recipe and converted locally by `simretract-build-assets` (`docs/ASSETS.md`) |
| WidowX-AI follower USD (`trossen_ai_isaac`) | robot in the simulation scene; source of the gripper silhouette LUT | see the software table above (BSD-3-Clause script headers, asset terms unverified) | git submodule, not redistributed |
| flat-pad fingers (`assets/gripper/`) | gripper fingers in simulation and on the robot | the authors' own CAD, released under the repository license | tracked |

Per-file status of everything under `assets/`: `assets/LICENSES.md`.
