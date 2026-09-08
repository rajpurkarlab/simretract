# Gripper fingers

The finger design is adapted from **Source Robotics' SSG-48 adaptive electric
gripper**, with modifications to fit the **Trossen WidowX-AI**.

- Original project: [SSG-48 adaptive electric gripper](https://github.com/Source-Robotics/SSG-48-adaptive-electric-gripper).
- Upstream license: [Apache-2.0](https://github.com/Source-Robotics/SSG-48-adaptive-electric-gripper/blob/main/LICENSE), with a copy in [LICENSE](LICENSE).
- Modification: adaptation for the Trossen mounting, exported as the visual
  meshes and convex collision parts used by SimRetract.

| Files | Contents |
|---|---|
| `finger_left_visual.obj`, `finger_right_visual.obj` | Exported finger visual meshes |
| `collision_left/`, `collision_right/` | 12 convex collision meshes per side |
| `material.mtl`, `gripper_flatpad.png` | Material and reference image |

These are simulation mesh exports. Editable CAD source for the Trossen
adaptation and the complete SSG-48 assembly are not included in this folder.
The upstream project provides its original CAD separately.

Original design credit belongs to Source Robotics and the upstream
contributors. The project's pending code-license choice does not replace the
upstream terms for these derived assets.
