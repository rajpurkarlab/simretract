# Tool asset: acquisition and local preparation

The retractor the policy hands over is a third-party 3D model. The
repository never contains it in any form that carries its geometry: no
source file, no visual or collision mesh, no USD, no textures, no
reconstruction captures. What the repository tracks for the tool is
annotation and recipe only: `assets/tools/retractor_rake/meta.json`
(extents, volume, tip/bend points, the collision-set summary and the
settings of the build that produced them), `configs/assets/retractor_rake.yaml`
(which file to convert, how, and the fingerprints the result must
reproduce) and `assets/grasp_candidates/retractor_rake.json` (the sampled
grasp poses the recorders draw from). Everything else is built on your
machine from your own licensed copy of the asset with
`simretract-build-assets`, and the build is checked against the tracked
references so you know whether your copy reproduces the released setup.

## Drop-in folder

Place the pack under `assets/source/<pack>/` exactly as downloaded,
without renaming or converting anything:

```
assets/source/vretractor/
  vretractor.fbx
  textures/
    vretractor_metal_Mat_BaseColor.png
    vretractor_metal_Mat_Normal.png
    vretractor_metal_Mat_Roughness.png
    vretractor_metal_Mat_Metallic.png
    vretractor_metal_Mat_AO.png
```

`assets/source/` is git-ignored except for its README, and the unit and
release tiers check that nothing under it would be committed. The
converter takes `vretractor.fbx` from there by default; `tool_source` in the
git-ignored `configs/envs.local.yaml` (or `SIMRETRACT_TOOL_SOURCE`) is an
override for a file kept elsewhere, and `--from-mesh FILE` on the command
line overrides both. The textures are not used by the simulation (the tool
is rendered with a steel shader).

## Acquisition

| item | value |
|---|---|
| asset | Operating Volkman Retractor Surgical Instrument |
| creator | HQ3DMOD |
| listing | <https://www.fab.com/listings/28243153-9617-4d27-a1c1-779d22e6f61a> |
| license | Fab Standard License, <https://www.fab.com/eula>; the listing states "Allows usage with AI: Yes" |
| file | `vretractor.fbx`, 210,860 bytes |
| sha256 | `df9689bc33e997417d2898406cdb7214cbabde58744513b2ddd0013950061cbb` |
| FBX node | `ovrsi_retractor_body` (the file's only mesh node) |
| conversion | scale `0.01474075`, pre-rotation = half-turn about y, then the converter's own steps (recipe `source` block) |

Obtain the asset from the listing under its license and drop the folder in
as shown above. The converter verifies the checksum of the file it converts
and prints `[SOURCE] vretractor.fbx: recipe source 'vretractor' (sha256
verified)`; a later revision of the pack with a different digest is
reported and converted anyway, and the fingerprint check then decides
whether it still reproduces the released tool.

Building this file with the shipped recipe reproduces the released tool
geometry within 0.25 um (the recipe's grid fingerprint): the same 3152
vertices and 6308 faces in the same order, the same extents and volume, and
the same tip and bend points. The released grasp candidates were generated
from it and re-sample identically from the local build (`--grasps
--compare`, 1 um on positions and 1e-4 on rotation entries). The collision
decomposition is the one step that is not deterministic: CoACD
(`coacd==1.0.11`, threshold 0.015) splits the mesh into 20 or 21 convex parts
with a summed hull volume between 7.82 and 8.00 cm3 from run to run, so the
reference checks accept the collision set within 3 parts and 5% of the hull
volume. Physics-driven outputs (regenerated demonstrations, benchmark
rollouts) are not bit-reproducible in any case (`README.md`, "Rebuild from
scratch").

Users may obtain other retractor assets and run the same pipeline ("Using a
different asset"); the shipped grasp candidates and metadata are for this
Fab model.

## Dependencies

The conversion runs in the simulator environment (`envs/sim.txt` pins
`trimesh` and `scipy`). Two optional extras are needed for an FBX source
and for the collision decomposition; they are not installed by
`envs/setup.sh`:

```bash
conda activate simretract-sim                 # or whatever sim_python in configs/envs.local.yaml is
pip install "coacd==1.0.11" "pyassimp==5.2.5"
conda install -c conda-forge assimp=5.4.2     # the assimp shared library that pyassimp loads (pulls libboost and zstd)
```

The verified pairing is `pyassimp==5.2.5` with the conda-forge `assimp`
5.4 package; `pyassimp` reads the FBX through that library, and the
converter adds the environment's `lib/` directory to the loader path when
the library was installed with conda.

- `coacd==1.0.11` is the CoACD build the reference decomposition was made
  with; a different build can split the mesh differently and the
  collision-set check then reports it (see "Reference values").
- `pyassimp` reads FBX and other multi-object scene files through
  `libassimp`. Any assimp shared library found on `LD_LIBRARY_PATH`, in the
  environment's `lib/` or in the system library directories works (a
  distribution package such as `libassimp-dev` is an alternative to the
  conda package). OBJ, STL, PLY and glTF sources need only `trimesh`; STEP
  sources need `cascadio`, `gmsh` or FreeCAD.
- The fingerprint verifier (`simretract.sim.asset_recipe`) needs numpy and
  PyYAML only and runs in any of the environments.

## Build

One command runs the whole build from the recipe:

```bash
scripts/build_assets.sh --accept-eula         # = simretract-build-assets --tool retractor_rake --accept-eula
```

which is, step by step (each step can be run on its own):

```bash
# 1. source file -> assets/tools/retractor_rake/{visual.obj, collision/coll_NNN.obj},
#    checked against meta.json ([MESH]) and the recipe fingerprints ([VERIFY])
simretract-build-assets --tool retractor_rake --from-mesh
# 2. rigid-body USD for Isaac Sim -> assets/usd/retractor_rake.usd, checked against meta.json ([BUILD])
simretract-build-assets --tool retractor_rake --usd --accept-eula
# 3. the tracked grasp candidates must re-sample identically from the built mesh ([GRASPS])
simretract-build-assets --tool retractor_rake --grasps --compare
# fingerprint check of an already built visual.obj (no simulator needed)
simretract-build-assets --tool retractor_rake --verify
# (optional) gripper silhouette LUT for the real-robot client; the tracked file is the reference
simretract-build-assets --lut --accept-eula
```

Every setting has a default from the recipe source when the input file
matches it (`--select`, `--scale`, `--pre-rotate`) or from the recipe
(`--align-z`, `--coacd-threshold`), then from `meta.json`; spelled out:

```bash
simretract-build-assets --from-mesh assets/source/vretractor/vretractor.fbx --tool retractor_rake \
    --select ovrsi_retractor_body --scale 0.01474075 --pre-rotate "-1,0,0, 0,1,0, 0,0,-1" \
    --align-z --coacd-threshold 0.015
```

Expected console lines: `[SOURCE] vretractor.fbx: recipe source 'vretractor'
(sha256 verified)`, `[MESH] check vs .../meta.json: OK (reference kept)`,
`[VERIFY] ... vertex_grid_sha256 OK: every coordinate within 0.25 um of the
released mesh` and `[VERIFY] .../visual.obj vs configs/assets/retractor_rake.yaml: OK`,
`[BUILD] check vs .../meta.json: OK`, `[GRASPS] compare vs
.../retractor_rake.json: OK`. Step 1 takes seconds; step 2 boots Isaac Sim
(a minute or two, longer on the first boot while shaders compile); step 3
takes about a minute.

## What the converter does

`simretract/sim/mesh_convert.py`, in this order:

1. **Load.** For an FBX the selected node is read with assimp
   (triangulated, identical vertices joined) and every transform from the
   scene root down to that node is baked into the vertices, so the
   coordinates are the pack's world frame. Other mesh formats are read with
   trimesh (`--select` picks a geometry of a scene file).
2. **Scale and pre-rotate.** Coordinates are multiplied by `--scale`
   (metres per source unit; the recipe's value) and by the recipe's
   `pre_rotate` matrix in one step. The pre-rotation fixes the input
   orientation the alignment step below sees, which sets the rotation
   about the tool axis. The converter warns and names the scale that would
   land a tool at about 0.2 m when the result is far from that.
3. **Repair.** Merge coincident vertices, drop duplicate and degenerate
   faces, remove unreferenced vertices, fix face winding, fill holes.
4. **Centre.** Translate so the volumetric centroid (the centre of mass of
   the watertight mesh; the bounding-box centre otherwise) is the origin.
   The removed offset, in metres after scaling and before alignment, is
   `center_removed_pre_m` (source-specific; the recipe records the expected
   value per source).
5. **Align (`--align-z`).** The principal axis of the vertices (SVD) is
   rotated onto +z, oriented so the blade end is at +z, and the mesh is
   re-centred. The blade end is the end of the axis whose extreme vertex
   sits farther off the axis (the bent rake). The resulting tool frame:
   shaft along z, rake tip at +z, handle at -z. Only +z is fixed by the
   converter; the rotation about z is whatever the alignment rotation
   leaves, so it follows from the input orientation, which is why the
   recipe's pre-rotation is part of the recipe. For the reference tool x comes out as the
   blade-width direction (about 3.1 cm of extent) and y as the blade
   thickness (about 1 cm). The scene's `tip_down` option flips local +z to
   world-down when the tool is presented.
6. **Tip and bend.** `suggested_tip` is the extreme vertex of the blade end;
   `suggested_bend` is the vertex mean plus 0.7 of the tip's projection
   along the axis. The USD carries both as marker xforms; the recorders and
   the grasp sampler read `suggested_tip` from `meta.json` (the sampler's
   4 cm tip keep-out).
7. **Export.** `visual.obj` (trimesh OBJ export of the mesh above, eight
   decimals).
8. **Decompose.** CoACD reads `visual.obj` back and writes one convex part
   per file into `collision/coll_NNN.obj` (`coacd_threshold` 0.015: lower
   means more parts). Decomposing the exported file, not the in-memory
   mesh, makes the parts depend only on that file and the CoACD build.
9. **Check.** The fresh extents, longest extent, volume, part count and
   hull volume are compared with the tracked `meta.json` (0.1 mm, 0.1 mm,
   1%, 3 parts, 5%). On a match the tracked file is left untouched; on a
   mismatch the command exits non-zero and leaves it untouched unless
   `--accept-meta` records the local build as the new reference.
   `meta.json` never records the source path.
10. **Verify.** With a recipe, the exported `visual.obj` is fingerprinted
    and compared with the recipe block (vertex count, face count, exact
    face hash, exact or grid vertex hash, extents to 0.1 um, volume to
    1e-6, watertightness); a mismatch exits non-zero. `--no-verify` skips
    this for a different asset.

The USD builder (`simretract/sim/tool_usd.py`) then authors a rigid body
with steel density (7900 kg/m3), a physics material (friction overridden
per episode by the scene), the visual mesh with a steel shader, the convex
parts as convex-hull colliders (rendered as guides only) and the `tip` /
`bend` markers, and checks the visual extents, the mesh volume and the
collision set against `meta.json` again.

## Reference values

`assets/tools/retractor_rake/meta.json`, the build from the recipe source
(the recipe's `expected_meta` block repeats the checked values):

| key | value |
|---|---|
| `select` / `scale` / `pre_rotate` | `ovrsi_retractor_body` / 0.01474075 / `[[-1, 0, 0], [0, 1, 0], [0, 0, -1]]` |
| `align_z` / `coacd_threshold` | true / 0.015 |
| `extents_m` | [0.03059, 0.00967, 0.21631] |
| `longest_m` | 0.21631 |
| `watertight` / `volume_m3` | true / 6.611e-06 (6.61 cm3, about 52 g of steel) |
| `center_removed_pre_m` | [-0.01806, 0.00234, 0.0] |
| `n_collision_parts` / `collision_hull_volume_m3` | 21 / 7.848e-06 |
| `suggested_tip` | [-0.00677, 0.0012, 0.12624] |
| `suggested_bend` | [-0.0, 0.00033, 0.09956] |

`configs/assets/retractor_rake.yaml`, `fingerprint` block of the released
`visual.obj` (3152 vertices, 6308 faces):

| key | value |
|---|---|
| `face_sha256` (int64 faces, C-order bytes) | `7e88d550b79ee89005e815d9f42e7d4c26d4076b31461a385f4ecd7bb218d9b7` |
| `vertex_sha256` (float64 vertices rounded to 1e-6 m, C-order bytes) | `9f71ad5b457d967f3a9d623159dcbe37ed828eed0e292e020ed92266d7c61c34` |
| `vertex_grid_sha256` (1 um grid under `vertex_grid_phase`) | the identity check: reproduced by every build within 0.25 um of the released mesh |
| `extents_m` / `volume_m3` / `watertight` | [0.03059319, 0.00967014, 0.21631345] / 6.610871554e-06 / true |

The phase bits are one bit per coordinate (the second bit below the
micrometre) and carry no shape information. To fingerprint a mesh of your
own: `python -m simretract.sim.asset_recipe --obj FILE --print-fingerprint`.

The collision set is verified on one machine; CoACD is not guaranteed to
split identically across platforms or runs (see "Acquisition"), and a
count within 3 of the reference with a hull volume within 5% is the same
tool with a slightly different convex approximation, not a wrong asset. A
count outside that (a different CoACD build or threshold) fails the check;
`--accept-meta` records the local build as the reference if that is
intended (the visual mesh, extents, volume, fingerprints and grasp
candidates still have to match).

## Where the files land, and what never gets committed

| path | produced by | tracked |
|---|---|---|
| `configs/assets/retractor_rake.yaml` | the recipe: source, settings, expected values, fingerprints | yes |
| `assets/source/README.md` | the drop-in convention | yes |
| `assets/tools/retractor_rake/meta.json` | reference; `--from-mesh --accept-meta` rewrites it | yes |
| `assets/grasp_candidates/retractor_rake.json` | `--grasps` (verify with `--grasps --compare`) | yes |
| `assets/gmask_lut.npz` | `--lut` (gripper silhouettes; no tool geometry) | yes |
| `assets/source/<pack>/...` | you (the asset folder as downloaded) | never |
| `assets/tools/retractor_rake/visual.obj` | `--from-mesh` | never |
| `assets/tools/retractor_rake/collision/coll_NNN.obj` | `--from-mesh` | never |
| `assets/usd/retractor_rake.usd` | `--usd` | never |
| any other export, texture, scan or point cloud of the asset | | never |

`.gitignore` excludes everything under `assets/source/` except the README,
every geometry file under `assets/tools/`, all of `assets/usd/`, and
`*.fbx`, `*.blend`, `*.max`, `*.ply`, `*.stl`, `*.glb`, `*.gltf`, `*.usd*`
and `*.obj` anywhere, with the attributed gripper OBJ exports under
`assets/gripper/` as the only exception; `tests/unit/test_asset_exclusion.py`
and the release tier check that nothing matching those patterns would be
committed and that the tracked tool files hold poses and metadata only. Do
not add the source mesh, converted meshes, collision parts, USD files,
textures, scans or point clouds of the asset to the repository, to the
released datasets or to the hub repositories.

## Using a different asset

With another model (any retractor of your own):

```bash
simretract-build-assets --from-mesh /path/to/model.fbx --tool retractor_rake        # lists the node names when --select is missing
simretract-build-assets --from-mesh /path/to/model.fbx --tool retractor_rake \
    --select <node> --scale <units-to-metres> --pre-rotate identity --align-z --accept-meta   # new reference meta.json, no fingerprint check
simretract-build-assets --tool retractor_rake --usd --accept-eula
simretract-build-assets --tool retractor_rake --grasps                             # re-samples the candidates (overwrites the JSON)
```

A file that is not the recipe source is reported as such and converted
with the command-line settings, falling back to `meta.json`; pass
`--pre-rotate identity` unless the model needs a turn. Check the printed
longest extent (a hand retractor is about 0.2 m; the warning names the
scale that would land it there), the tip/bend suggestion and, after
`--usd`, the part count. Regenerated demonstrations then use the new
geometry; the shipped grasp candidates, metadata, datasets and checkpoints
are for the Fab model above and are not reproduced by a different model. To keep the
fingerprint check for your own model, record its block with
`python -m simretract.sim.asset_recipe --obj assets/tools/retractor_rake/visual.obj --print-fingerprint`
in a recipe of your own.

## Smoke tier

`scripts/smoke_test.sh sim` starts with the default build above (from the
drop-in file or the `tool_source` override), which converts, verifies the
fingerprints, builds the USD and compares the grasp candidates, then runs
the standalone `--verify` and `--grasps --compare` checks, records
episodes, builds caches and runs the benchmark. Without a source file the
build step is skipped with a message and the later steps run only if the
assets were built earlier on the machine; otherwise they skip too.
