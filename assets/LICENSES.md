# Asset files

| path | what | tracked |
|---|---|---|
| `source/README.md` | the drop-in convention for the asset folder | yes |
| `tools/retractor_rake/meta.json` | reference description of the tool (extents, volume, collision-set summary, tip/bend points, build settings); no geometry | yes |
| `grasp_candidates/retractor_rake.json` | the 444 grasp candidates (poses and finger contact points in the tool frame) the recorders draw from | yes |
| `gmask_lut.npz` | gripper silhouette look-up table rendered from the robot USD and the fingers below; no tool geometry | yes |
| `gripper/*` | mesh exports of the Source Robotics SSG-48 finger design, modified to fit the Trossen WidowX-AI; upstream Apache-2.0, see [attribution](gripper/README.md) | yes |
| `source/vretractor/vretractor.fbx`, `source/vretractor/textures/*.png` | the retractor asset as downloaded | never |
| `tools/retractor_rake/visual.obj`, `tools/retractor_rake/collision/coll_NNN.obj` | tool meshes built locally by `simretract-build-assets` | never |
| `usd/retractor_rake.usd` | tool USD built locally by `simretract-build-assets --usd` | never |

Source of the retractor asset: "Operating Volkman Retractor Surgical
Instrument" by HQ3DMOD on Fab
(<https://www.fab.com/listings/28243153-9617-4d27-a1c1-779d22e6f61a>), Fab
Standard License (<https://www.fab.com/eula>). Obtained separately by each
user and never redistributed; building it with the shipped recipe
(`configs/assets/retractor_rake.yaml`) reproduces the released tool geometry
within 0.25 um (grid fingerprint), and the released grasp candidates were
generated from it (`docs/ASSETS.md`).
