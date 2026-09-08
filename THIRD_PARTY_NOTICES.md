# Third-party notices

Upstream components and adapted assets retain the notices and terms below.

## Software

| name | source | license |
|---|---|---|
| NVIDIA Isaac Sim 5.1 (`isaacsim` wheels) | <https://pypi.nvidia.com> (`envs/sim.txt`) | NVIDIA Omniverse license (EULA accepted via `OMNI_KIT_ACCEPT_EULA=YES`) |
| `trossen_ai_isaac` (WidowX-AI follower USD and controller) | <https://github.com/TrossenRobotics/trossen_ai_isaac> (git submodule) | BSD-3-Clause headers in the scripts; the checkout ships no LICENSE file |
| `trossen-arm` 1.10.0 | <https://pypi.org/project/trossen-arm/> | BSD-3-Clause |
| `lerobot` | <https://github.com/huggingface/lerobot> (pinned commit, `envs/LEROBOT_PIN.md`) | Apache-2.0 |
| `transformers`, `safetensors`, `huggingface_hub`, `accelerate`, `diffusers` | PyPI | Apache-2.0 |
| `torch`, `torchvision` | <https://download.pytorch.org/whl/cu128> | BSD-3-Clause |
| `ultralytics` (SAM 3 predictor, mask server) | <https://pypi.org/project/ultralytics/> | AGPL-3.0 (separate process and environment) |
| `clip` (SAM 3 text encoder utilities) | <https://github.com/ultralytics/CLIP> (pinned in `envs/mask.txt`) | AGPL-3.0 (license distributed with this fork) |
| `timm` (SAM 3 vision modules) | PyPI (`envs/mask.txt`) | Apache-2.0 |
| `pyrealsense2` | <https://pypi.org/project/pyrealsense2/> | Apache-2.0 |
| `trimesh`, `scipy`, `numpy`, `opencv-python`, `pillow`, `imageio`, `imageio-ffmpeg`, `av`, `matplotlib`, `pyyaml`, `einops`, `num2words`, `draccus`, `gymnasium` | PyPI | MIT / BSD / Apache-2.0 / LGPL (`imageio-ffmpeg` bundles an LGPL ffmpeg build) |
| `coacd` 1.0.11 (optional, asset build) | <https://pypi.org/project/coacd/> | MIT |
| `pyassimp` + `assimp` (optional, FBX import) | <https://pypi.org/project/pyassimp/>, <https://github.com/assimp/assimp> | BSD-3-Clause |

## Weights and tokenizers

| name | source | license |
|---|---|---|
| pi0 base weights (`pi0_base`, a re-hosted copy of a `lerobot/pi0_base` revision) | `configs/hub.yaml` | Apache-2.0 (lerobot conversion of openpi); Gemma Terms of Use for the PaliGemma backbone |
| PaliGemma tokenizer | <https://huggingface.co/google/paligemma-3b-pt-224> | Gemma Terms of Use (gated) |
| SmolVLA base | <https://huggingface.co/lerobot/smolvla_base> | Apache-2.0 |
| SmolVLM2 tokenizer | <https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct> | Apache-2.0 |
| SAM 3 weights (`sam3.pt`) | <https://huggingface.co/facebook/sam3> | SAM 3 License (gated; not redistributed) |
| ResNet-18 ImageNet weights | torchvision `ResNet18_Weights.IMAGENET1K_V1` | BSD-3-Clause |
| Fast-WAM, pi0.5, pi0-FAST bases (optional) | <https://huggingface.co/lerobot> | see the hub cards |

## 3D assets

| name | source | license |
|---|---|---|
| "Operating Volkman Retractor Surgical Instrument" by HQ3DMOD | <https://www.fab.com/listings/28243153-9617-4d27-a1c1-779d22e6f61a> | Fab Standard License (<https://www.fab.com/eula>); obtained separately, never redistributed (`docs/ASSETS.md`) |
| WidowX-AI follower USD | `trossen_ai_isaac` submodule (above) | see above |
| flat-pad finger mesh exports (`assets/gripper/`), modified to fit the Trossen WidowX-AI | [Source Robotics, SSG-48 adaptive electric gripper](https://github.com/Source-Robotics/SSG-48-adaptive-electric-gripper); [adaptation notice](assets/gripper/README.md) | upstream Apache-2.0; [license copy](assets/gripper/LICENSE) |
