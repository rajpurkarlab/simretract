# Asset files: provenance and release status

Every file under `assets/` is either tracked (released with the code),
dropped in by the user (the purchased asset folder) or built locally from
that folder, and the last two are never tracked. `.gitignore`,
`tests/unit/test_asset_exclusion.py` and the release tier enforce the last
column.

| path | what | origin | tracked |
|---|---|---|---|
| `source/README.md` | the drop-in convention for purchased packs | this repository | yes |
| `tools/retractor_rake/meta.json` | reference description of the tool: extents, longest extent, volume, collision set (number of convex parts, summed hull volume), tip/bend points and the settings of the build that produced them (`select`, `scale`, `pre_rotate`, `align_z`, `coacd_threshold`) | written by `simretract-build-assets --from-mesh`; carries no source path | yes |
| `grasp_candidates/retractor_rake.json` | grasp-candidate database: 444 flat-clamp, antipodal and washer grasp poses in the tool frame (seed 0), each with its two finger contact points; the recorders permute it per seed, so the file is part of the data contract | `simretract-build-assets --grasps`; `--grasps --compare` verifies it against a local build | yes |
| `gmask_lut.npz` | gripper silhouette look-up table (45 openings x 480 x 640) rendered from the vendor robot USD and the fingers below; contains no tool geometry | `simretract-build-assets --lut` | yes |
| `gripper/finger_left_visual.obj`, `gripper/finger_right_visual.obj`, `gripper/collision_left/`, `gripper/collision_right/`, `gripper/material.mtl`, `gripper/gripper_flatpad.png` | flat-pad fingers mounted on the vendor gripper carriages | authors' own CAD; released under the repository license (`LICENSING.md`) | yes |
| `source/vretractor/vretractor.fbx`, `source/vretractor/textures/*.png` | the purchased Fab pack, as downloaded | the user's own purchase | never |
| `tools/retractor_rake/visual.obj` | the retractor visual mesh | built locally by `simretract-build-assets` from the pack above (or the `tool_source` override) | never |
| `tools/retractor_rake/collision/coll_NNN.obj` | CoACD convex parts of the mesh above | same | never |
| `usd/retractor_rake.usd` | rigid-body USD referencing the meshes above | `simretract-build-assets --usd` | never |
| `third_party/trossen_ai_isaac/assets/robots/wxai/wxai_follower.usd` | WidowX-AI follower robot with wrist camera | Trossen Robotics, git submodule; not redistributed here. The submodule's scripts carry BSD-3-Clause headers; it ships no LICENSE file and states no terms for the USD assets (unverified) | submodule |

## The retractor asset

- Default source, the purchasable listing: "Operating Volkman Retractor
  Surgical Instrument" by HQ3DMOD on Fab
  (<https://www.fab.com/listings/28243153-9617-4d27-a1c1-779d22e6f61a>),
  Fab Standard License (<https://www.fab.com/eula>); the listing states
  "Allows usage with AI: Yes". File `vretractor.fbx`, 210,860 bytes, sha256
  `df9689bc33e997417d2898406cdb7214cbabde58744513b2ddd0013950061cbb`, node
  `ovrsi_retractor_body`, converted with the recipe
  `configs/assets/retractor_rake.yaml` (scale 0.01474075, a half-turn about
  y, blade axis aligned to +z). It is another export of the same model as
  the one below and rebuilds the released tool mesh: same 3152 vertices and
  6308 faces in the same order, every vertex within 2.9e-8 m (the exports'
  single-precision rounding), verified by the recipe fingerprints
  (`docs/ASSETS.md`, "Asset identity"). Keep the purchase record and the
  terms that applied at purchase.
- The export the released data was recorded with: rake retractor node
  `so7si_retractor_2_body` of the CGTrader pack "Set of 7 surgical
  instruments"
  (<https://www.cgtrader.com/3d-models/science/medical/set-of-7-surgical-instruments>),
  file `uploads_files_5268240_surginstruments.fbx`, 3,518,400 bytes, sha256
  `8f7182fd5867b5da7c023b366ab37c73a37f13030ca119a0ff3011524bed6de3`,
  converted at scale 0.01 (inferred from the resulting 21.6 cm tool length,
  not read from the pack's metadata) with the blade axis aligned to +z. The
  released datasets, checkpoints and benchmark numbers were produced with
  this export and `meta.json` describes its build; that provenance is
  recorded as it is. Used under the terms of that listing; the terms govern
  what may be done with a copy and do not authorise redistribution of the
  model or of reconstructable derivatives, so none is tracked.

## Review of the released tool annotations

`grasp_candidates/retractor_rake.json`, `tools/retractor_rake/meta.json` and
the recipe `configs/assets/retractor_rake.yaml` were reviewed before
release. What each file holds:

- `grasp_candidates/retractor_rake.json`: 444 candidates (192 flat-clamp,
  240 antipodal, 12 washer). Each holds a centre, a rotation matrix, the
  grasp axis, the approach direction, a finger width, a standoff, a quality
  score and the two finger contact points `p1` / `p2` from which the pose
  was derived. The contact points lie on the tool surface: 888 points, 220
  distinct, about 7% of the 3152 vertices of the visual mesh, with no
  faces, normals or connectivity. They are not a surface sampling: the
  samplers pick antipodal pairs on the shaft and handle and keep 4 cm clear
  of the tip, so the points cluster on the shaft and the rake blade is not
  covered. Sparse, non-uniform and unconnected, they do not allow the mesh
  to be rebuilt.
- `tools/retractor_rake/meta.json`: bounding extents, longest extent,
  volume, part count and hull volume of the collision set, the tip and bend
  marker points and the conversion settings.
- `configs/assets/retractor_rake.yaml`: the listings, file names, checksums
  and conversion settings of the two exports, the expected `meta.json`
  values, and the fingerprints of the released mesh: two SHA-256 digests of
  the vertex array (exact and grid-snapped), one of the face array, the
  extents, the volume, and one phase bit per coordinate (the second bit
  below the micrometre of each coordinate, 9456 bits). Digests and phase
  bits carry no shape information; the vertex and face arrays cannot be
  recovered from them.

They are released as task annotations under the terms in `LICENSE-DATA.md`
(grasp data) and `LICENSING.md` (metadata and recipe). Any of them can be
regenerated from a licensed copy of the asset: `simretract-build-assets
--from-mesh` rewrites `meta.json` (with `--accept-meta`), `--grasps`
re-samples the candidates, `python -m simretract.sim.asset_recipe
--print-fingerprint` prints the fingerprint block. If a future review finds
that a released annotation discloses reusable geometry, remove it from the
repository and regenerate it locally instead.
