# Dataset terms

The released datasets are the robot-action and grasp-point data described in
`data/README.md` and `docs/DATA_FORMAT.md`:

- `strict_train`: 500 expert take-over episodes (joint states and actions at
  25 Hz, the grasp pose, tool-masked RGB and depth renders, masks, per-episode
  metadata);
- `recovery_far1000`: 1000 perturbation-recovery episodes in the same format;
- the selection files under `data/selections/` and the grasp-candidate
  database `assets/grasp_candidates/retractor_rake.json` (444 grasp poses
  in the tool frame: centre, rotation, grasp axis, approach, width,
  standoff, quality and the two finger contact points of each grasp).

They were generated in Isaac Sim with a third-party retractor model that
the authors used under its publisher's license (`docs/ASSETS.md`,
`assets/LICENSES.md`). The datasets carry no source file, mesh, collision
part, USD, texture or point cloud of that asset. The tool appears in two
forms: as grasp poses (the candidate database and the per-episode grasp
pose) and in the wrist-camera renders of every episode (tool-masked RGB,
depth images in millimetres and tool masks at 25 Hz). The grasp-candidate
database and the per-episode grasp poses were reviewed before release
(`assets/LICENSES.md`, "Review of the released tool annotations"): sparse,
non-uniform contact points that do not allow the mesh to be rebuilt; they
can be regenerated locally from a licensed copy of the asset with
`simretract-build-assets --grasps`. The rendered depth and mask frames are
range images of the tool from many viewpoints and have not been reviewed
the same way. Whether they are published is to be confirmed by the authors
before the first public push: either the review finds them acceptable under
the asset terms and they stay in the public shards, or the depth and mask
frames are dropped from the shards.

## Data license

To be chosen by the authors before the release (CC BY 4.0 is the usual
choice for research datasets of this kind; a code-style license such as
Apache-2.0 also works for data). Replace this section with the chosen terms
and add the same text to the hub dataset cards. Until then
`scripts/smoke_test.sh release` reports this file as open.

Whatever license is chosen, it applies to the authors' own contribution
(the recorded trajectories, poses, renders and metadata) and does not grant
any rights to the third-party asset itself.
