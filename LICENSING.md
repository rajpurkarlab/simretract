# Licensing

| part | terms |
|---|---|
| Code (`simretract/`, `configs/`, `scripts/`, `tests/`, `envs/`, `docs/`, `data/selections/`, the tool annotations `assets/tools/retractor_rake/meta.json` and `assets/gmask_lut.npz`) | `LICENSE` (to be chosen by the authors; the same license goes into `pyproject.toml`) |
| Adapted gripper mesh exports (`assets/gripper/`) | Derived from Source Robotics' SSG-48 design; upstream Apache-2.0 terms and modification notice in `assets/gripper/README.md`. These upstream terms are separate from the project's pending code-license choice. |
| Policy checkpoints | public downloads in `checkpoints/README.md`; license to be chosen by the authors; upstream notices in `LICENSE-CHECKPOINTS.md` |
| Datasets (robot-action and grasp-point data) | download links upon publication; license to be chosen by the authors; `LICENSE-DATA.md` |
| Third-party components (software, weights, tokenizers, the robot package) | listed with their licenses in `THIRD_PARTY_NOTICES.md` |
| Third-party tool asset (the retractor model) | not part of the release: obtained separately from the source in `THIRD_PARTY_NOTICES.md` and prepared locally (`docs/ASSETS.md`); never redistributed |

`scripts/smoke_test.sh release` checks that `LICENSE` exists, the license
fields and terms are filled in, the hub organisation is set, no tool geometry
is tracked and the README carries the release statement.
