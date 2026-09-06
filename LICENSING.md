# Licensing

The release has four parts with separate terms. Nothing here is a license
itself; each row names the file that carries the terms.

| part | what it covers | terms |
|---|---|---|
| Code | `simretract/`, `configs/` (including the tool asset recipe `configs/assets/retractor_rake.yaml`), `scripts/`, `tests/`, `envs/`, `docs/`, `data/selections/`, the gripper CAD under `assets/gripper/` and the tracked tool annotations (`assets/tools/retractor_rake/meta.json`, `assets/gmask_lut.npz`) | `LICENSE`, to be added by the authors before the first public push (see below) |
| Policy checkpoints | the released rows under `checkpoints/` and the released copy of the pi0 base weights | `LICENSE-CHECKPOINTS.md`: per-row terms inherited from the base weights, with the upstream notices preserved |
| Datasets | the released robot-action datasets (`strict_train`, `recovery_far1000`) and the grasp-point data (`assets/grasp_candidates/retractor_rake.json`, the per-episode grasp poses) | `LICENSE-DATA.md` |
| Third-party tool asset | the retractor 3D model the simulation uses | not released; obtained by each user under its publisher's license (the Fab listing by HQ3DMOD), dropped under `assets/source/` and prepared locally (`docs/ASSETS.md`, `assets/LICENSES.md`) |

Third-party software, weights and tokenizers the stack depends on are listed
with their licenses in `THIRD_PARTY_NOTICES.md`; the ones that need a
separate acceptance (Isaac Sim EULA, Gemma Terms of Use, SAM 3 License) are
fetched by the user, never redistributed here.

## Code license: what the choice has to cover

`LICENSE` does not exist yet. Before publishing, the authors add it and fill
in `license` in `pyproject.toml` and `license` in `CITATION.cff`
(`README.md` points at this file). Any OSI license works for the authors' own
code; the constraints are:

- `simretract/real/mask_server.py` imports `ultralytics` (AGPL-3.0). The
  module runs only in the separate `simretract-mask` environment and
  process, and the library is not vendored, but the chosen license must be
  compatible with distributing that module, and offering the mask server as
  a network service may trigger AGPL obligations.
- The gripper CAD (`assets/gripper/`) is the authors' own work and can carry
  the code license or a CC license; say which in `assets/LICENSES.md`.
- The tracked tool annotation (`meta.json`), the recipe (`configs/assets/`:
  checksums, settings, digests and phase bits) and the gripper LUT contain
  no asset geometry (the LUT is rendered from the vendor robot USD and the
  authors' fingers); they carry the code license.
- `third_party/trossen_ai_isaac` is a git submodule and is not
  redistributed; its own terms apply (see `THIRD_PARTY_NOTICES.md`).

## Checkpoint terms

`LICENSE-CHECKPOINTS.md`. In short: every pi0 row (baseline, no recovery,
no depth, half data) and the released pi0 base copy embed a PaliGemma
backbone and are used under the Gemma Terms of Use in addition to the
Apache-2.0 notices of the lerobot conversion they descend from; the SmolVLA
row descends from Apache-2.0 weights; the ACT row is trained from scratch
(with torchvision's BSD-licensed ImageNet ResNet-18 encoder) and carries the
code license. No checkpoint contains asset geometry.

## Dataset terms

`LICENSE-DATA.md`. The released datasets are robot-action and grasp-point
data generated in simulation with a licensed third-party asset. They hold
joint trajectories, grasp poses, wrist-camera renders (tool-masked RGB,
depth images, tool masks) and metadata; no source file, mesh, collision
part, USD, texture or point cloud of the asset. The grasp files were
reviewed (`assets/LICENSES.md`); the rendered depth and mask frames are
range images of the tool and their review is recorded in that file. The
data license is chosen by the authors in that file.

## Third-party asset rule

The retractor model is never bundled: no source file, visual or collision
mesh, USD, texture, scan or point cloud of it enters the repository, the
hub repositories or the released datasets. Users obtain the asset
themselves (the Fab listing; the recorded data was made from another
export of the same model, `docs/ASSETS.md`), drop the purchased folder
under `assets/source/` (git-ignored) and run the local preparation. The
grasp candidates (poses with sparse finger contact points), the tool
metadata and the recipe fingerprints were reviewed as non-reconstructable
derivatives and are released; `assets/LICENSES.md` records that review.

## Publication gates

`scripts/smoke_test.sh release` (`tests/release/test_release_gates.py`)
fails while any of these is open: no `LICENSE` file; `pyproject.toml`
without a `license` entry; `CITATION.cff` with the placeholder license or
authors; `LICENSE-DATA.md` with the data license still to be chosen or the review of
the rendered depth and mask frames still to be confirmed;
`configs/hub.yaml` with the `YOUR_HF_ORG` placeholder or empty sha256
tables. It also re-checks the gates that already hold: this file,
`LICENSE-CHECKPOINTS.md` and `LICENSE-DATA.md` present, the release
statement verbatim in `README.md`, and no tool geometry in what would be
committed. Run it before the first public push.
