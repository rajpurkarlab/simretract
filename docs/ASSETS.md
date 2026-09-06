# Tool asset: acquisition, identity and local preparation

The retractor the policy hands over is a third-party 3D model. The
repository never contains it in any form that carries its geometry: no
source file, no visual or collision mesh, no USD, no textures, no
reconstruction captures. What the repository tracks for the tool is
annotation and recipe only: `assets/tools/retractor_rake/meta.json`
(extents, volume, tip/bend points, the collision-set summary and the
settings of the build that produced them), `configs/assets/retractor_rake.yaml`
(which purchased file to convert, how, and the fingerprints the result must
reproduce) and `assets/grasp_candidates/retractor_rake.json` (the sampled
grasp poses the recorders draw from). Everything else is built on your
machine from your own licensed copy of the asset with
`simretract-build-assets`, and the build is checked against the tracked
references so you know whether your copy reproduces the released setup.

## Drop-in folder

Place the purchased pack under `assets/source/<pack>/` exactly as
downloaded, without renaming or converting anything. For the default
source:

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

### The default source: the Fab listing

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

Keep the purchase record and the terms that applied at purchase with your
own records. The converter verifies the checksum of the file it converts
and prints `[SOURCE] vretractor.fbx: recipe source 'vretractor' (sha256
verified)`; a later revision of the pack with a different digest is
reported and converted anyway, and the fingerprint check then decides
whether it still reproduces the released tool.

### The export the released data was recorded with

The released datasets, checkpoints and benchmark numbers were produced
with the rake retractor of the CGTrader pack "Set of 7 surgical
instruments" (<https://www.cgtrader.com/3d-models/science/medical/set-of-7-surgical-instruments>):
file `uploads_files_5268240_surginstruments.fbx`, 3,518,400 bytes, sha256
`8f7182fd5867b5da7c023b366ab37c73a37f13030ca119a0ff3011524bed6de3`, node
`so7si_retractor_2_body`, converted at scale 0.01 with no pre-rotation. The
tracked `meta.json` describes that build, and this provenance of the
recorded data stays as it is: the datasets are not relabelled as having
been made from the Fab file. The recipe lists this export under
`alternative_sources`; a copy of it is recognised by sha256 and converted
with its own settings, and it reproduces `visual.obj` byte for byte.

## Asset identity

The two files are two exports of the same model, and the Fab download
converted with the shipped recipe reproduces the released tool. Measured
with the repository's converter (`pyassimp` 5.2.5 with assimp 5.4, `trimesh`
4.5.1) on the reference machine:

- **Same mesh.** Both nodes hold 3152 vertices and 6308 triangles, and the
  face arrays are identical: same connectivity over the same vertex order.
- **Pack frames.** A vertex-wise similarity fit of the Fab node onto the
  recorded export's node gives a uniform scale of 1.474075 and a rotation
  within 1.1e-7 rad of an exact half-turn about y, with a residual of at
  most 1.9e-8 m (mean 6.5e-9 m) once both are in metres. The residual is
  the exports' own rounding: both FBX files store their vertex arrays as
  doubles whose values are exactly representable in single precision, each
  in its own frame, so the two coordinate sets differ by single-precision
  rounding and cannot be made bit-identical by any transform.
- **Recipe.** Total scale 0.01 x 1.474075 = 0.01474075 and the pre-rotation
  `[[-1, 0, 0], [0, 1, 0], [0, 0, -1]]`, applied to the Fab coordinates
  before the converter's own repair, centring and alignment steps, put the
  file into the pack frame of the recorded export; the converter then does
  to it what it did to that export. (The direct transform from the Fab
  pack frame to the released tool frame is scale 1.474075 with rotation
  `[[0, 0, 1], [0.0161, 0.9999, 0], [-0.9999, 0.0161, 0]]`, Fab x onto
  tool z with a 0.92 degree roll: the half-turn composed with the
  converter's alignment rotation. The recipe applies only the half-turn so
  that the alignment step, whose roll follows from the input orientation,
  runs on the same input as for the recorded export.)
- **Result.** The `visual.obj` built from the Fab file has the same 3152
  vertices and 6308 faces in the same order (`face_sha256` identical),
  every vertex within 2.9e-8 m of the released mesh (mean 9e-9 m), the
  same extents to 1e-8 m (0.03059319 x 0.00967014 x 0.21631345 m), the same
  volume to 5e-8 relative (6.61087e-06 m3), and the same tip and bend
  points at the 1e-5 m resolution of `meta.json`. Of its 9456 coordinates,
  26 fall on the other side of a 1 um rounding boundary, so the exact
  fingerprint of the released file (`vertex_sha256`, coordinates rounded to
  1e-6 m) is not reproduced and 2272 of the 9462 lines of the OBJ differ in
  their eighth decimal. The tolerant fingerprint (`vertex_grid_sha256`, the
  coordinates snapped to a 1 um grid with a per-coordinate phase that keeps
  every coordinate of the released mesh at least 0.25 um from a snapping
  boundary) is reproduced; the Fab build's coordinates stay at least 0.23 um
  from every boundary. The build prints `[VERIFY] vertex_sha256 differs,
  vertex_grid_sha256 OK` followed by `[VERIFY] ... OK`. A copy of the
  recorded export prints `[VERIFY] vertex_sha256: exact`.
- **Collision set.** CoACD (`coacd==1.0.11`, threshold 0.015) is not
  deterministic: on the reference machine the recorded export's mesh
  splits into 21 convex parts with a summed hull volume of 7.848 cm3 (the
  released collision set, recorded in `meta.json`) on one run and into 20
  parts with 7.898 cm3 on another, and the Fab build into 20 or 21 parts
  with 7.82 to 8.00 cm3. The reference checks therefore accept the
  collision set within 3 parts and 5% of the hull volume.
- **Grasp candidates.** All 444 candidates re-sample identically from the
  Fab build: the same surface samples, sites, order and random draws;
  centres within 5e-8 m, widths within 3e-8 m, standoffs exact, rotation
  matrices within 2.3e-5 (the closing axis of the flat-clamp family is a
  face normal, and a 3e-8 m shift of a vertex tilts the normal of a
  millimetre-sized triangle by that much). `--grasps --compare` uses 1 um on
  positions and 1e-4 on rotation entries.

Physics-driven outputs (regenerated demonstrations, benchmark rollouts) are
not bit-reproducible in any case (`README.md`, "Rebuild from scratch").

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

Every setting has a default from the recipe source the input file matches
(`--select`, `--scale`, `--pre-rotate`) or from the recipe (`--align-z`,
`--coacd-threshold`), then from `meta.json`; spelled out for the Fab file:

```bash
simretract-build-assets --from-mesh assets/source/vretractor/vretractor.fbx --tool retractor_rake \
    --select ovrsi_retractor_body --scale 0.01474075 --pre-rotate "-1,0,0, 0,1,0, 0,0,-1" \
    --align-z --coacd-threshold 0.015
```

Expected console lines: `[SOURCE] vretractor.fbx: recipe source 'vretractor'
(sha256 verified)`, `[MESH] check vs .../meta.json: OK (reference kept)`,
`[VERIFY] .../visual.obj vs configs/assets/retractor_rake.yaml: OK`,
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
   (metres per source unit; the recipe's per-source value) and by the
   recipe's `pre_rotate` matrix in one step. The pre-rotation brings a
   source into the pack frame of the recorded export so that the alignment
   below sees the same input; it is the identity for the recorded export.
   The converter warns and names the scale that would land a tool at about
   0.2 m when the result is far from that.
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
   recipe's pre-rotation matters. For the reference tool x comes out as the
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

`assets/tools/retractor_rake/meta.json`, the build from the recorded export
(the recipe's `expected_meta` block repeats the checked values):

| key | value |
|---|---|
| `select` / `scale` / `pre_rotate` | `so7si_retractor_2_body` / 0.01 / none (the recorded export; the recipe's `source` block holds the Fab values) |
| `align_z` / `coacd_threshold` | true / 0.015 |
| `extents_m` | [0.03059, 0.00967, 0.21631] |
| `longest_m` | 0.21631 |
| `watertight` / `volume_m3` | true / 6.611e-06 (6.61 cm3, about 52 g of steel) |
| `center_removed_pre_m` | [-0.02444, 0.00234, 0.11637] (the Fab file: [-0.01806, 0.00234, 0.0]) |
| `n_collision_parts` / `collision_hull_volume_m3` | 21 / 7.848e-06 |
| `suggested_tip` | [-0.00677, 0.0012, 0.12624] |
| `suggested_bend` | [-0.0, 0.00033, 0.09956] |

`configs/assets/retractor_rake.yaml`, `fingerprint` block of the released
`visual.obj` (3152 vertices, 6308 faces, byte for byte the mesh of the
reported experiments):

| key | value |
|---|---|
| `face_sha256` (int64 faces, C-order bytes) | `7e88d550b79ee89005e815d9f42e7d4c26d4076b31461a385f4ecd7bb218d9b7` |
| `vertex_sha256` (float64 vertices rounded to 1e-6 m, C-order bytes) | `9f71ad5b457d967f3a9d623159dcbe37ed828eed0e292e020ed92266d7c61c34` |
| `vertex_grid_sha256` (1 um grid under `vertex_grid_phase`) | reproduced by every build within 0.25 um of the released mesh |
| `extents_m` / `volume_m3` / `watertight` | [0.03059319, 0.00967014, 0.21631345] / 6.610871554e-06 / true |

The phase bits are one bit per coordinate (the second bit below the
micrometre) and carry no shape information. To fingerprint a mesh of your
own: `python -m simretract.sim.asset_recipe --obj FILE --print-fingerprint`.

The collision set is verified on one machine; CoACD is not guaranteed to
split identically across platforms or runs (see "Asset identity"), and a
count within 3 of the reference with a hull volume within 5% is the same
tool with a slightly different convex approximation, not a wrong asset. A
count outside that (a different CoACD build or threshold) fails the check;
`--accept-meta` records the local build as the reference if that is
intended (the visual mesh, extents, volume, fingerprints and grasp
candidates still have to match).

## Where the files land, and what never gets committed

| path | produced by | tracked |
|---|---|---|
| `configs/assets/retractor_rake.yaml` | the recipe: sources, settings, expected values, fingerprints | yes |
| `assets/source/README.md` | the drop-in convention | yes |
| `assets/tools/retractor_rake/meta.json` | reference; `--from-mesh --accept-meta` rewrites it | yes |
| `assets/grasp_candidates/retractor_rake.json` | `--grasps` (verify with `--grasps --compare`) | yes |
| `assets/gmask_lut.npz` | `--lut` (gripper silhouettes; no tool geometry) | yes |
| `assets/source/<pack>/...` | you (the purchased folder as downloaded) | never |
| `assets/tools/retractor_rake/visual.obj` | `--from-mesh` | never |
| `assets/tools/retractor_rake/collision/coll_NNN.obj` | `--from-mesh` | never |
| `assets/usd/retractor_rake.usd` | `--usd` | never |
| any other export, texture, scan or point cloud of the asset | | never |

`.gitignore` excludes everything under `assets/source/` except the README,
every geometry file under `assets/tools/`, all of `assets/usd/`, and
`*.fbx`, `*.blend`, `*.max`, `*.ply`, `*.stl`, `*.glb`, `*.gltf`, `*.usd*`
and `*.obj` anywhere, with the gripper's own OBJ files under
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

A file that matches no recipe source is reported as such and converted
with the command-line settings, falling back to `meta.json`; pass
`--pre-rotate identity` unless the model needs a turn. Check the printed
longest extent (a hand retractor is about 0.2 m; the warning names the
scale that would land it there), the tip/bend suggestion and, after
`--usd`, the part count. Regenerated demonstrations then use the new
geometry; the released datasets and checkpoints remain tied to the
recorded export and are not reproduced by a different model. To keep the
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
