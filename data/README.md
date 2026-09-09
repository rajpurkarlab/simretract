# Data

[Simulation benchmark results](evaluation/) contain the paper
reference table and per-seed first-attempt outcomes.

## What ships

`data/selections/` holds the seed lists and episode lists that define every
training and test set (schemas in `docs/DATA_FORMAT.md`):

| file | content |
|---|---|
| `strict_500.json` | the strict training set: 500 seeds, 125 per 2.5 cm grasp-height bin, with per-seed grasp height and the recorder recipe |
| `recovery_train_2000.json` | the exact-4 recovery selection (1954 training episodes + 39 surplus) |
| `recovery_far1000.json` | the 1000 recovery episodes with the smallest perturbation fraction - the set the models train on |
| `strict_500_no_smoothing.json` | the no-smoothing strict set: 500 seeds of the same task recorded using a two-segment straight reach and the grasp candidate's own standoff, 125 per 2.5 cm grasp-height bin (`no_smooth` preset) |
| `recovery_far1000_no_smoothing.json` | its recovery set: 1000 far recovery draws (perturbation fraction at most 0.5) on those seeds |
| `strict_500_no_standoff.json` | the no-standoff strict set: 500 seeds recorded using the smoothed reach curving straight to the grasp, band-balanced over the grasp height (`no_standoff` preset) |
| `recovery_far1000_no_standoff.json` | its recovery set: the 1000 recovery draws with the smallest perturbation fraction on those seeds |
| `test_pool.json` | the frozen held-out pool (115 seeds); excluded from every cache |
| `test_core40.json`, `test_extra10.json`, `test_core50.json` | benchmark seed sets; `test_core50.json` is the protocol set |

`data/raw/` and `data/cache/` are git-ignored and are filled by the commands
below. The selection files, not the contents of a download, define a training
set.

The released datasets are robot-action and grasp-point data: joint states
and actions, the grasp pose, tool-masked RGB and depth renders, masks and
metadata per episode, plus the grasp-candidate database
(`assets/grasp_candidates/retractor_rake.json`). They were generated in
simulation with a third-party retractor model used under its publisher's
license. They carry no source file, mesh, collision part, USD, texture or
point cloud of that asset; the tool appears in the wrist-camera renders
(tool-masked RGB, depth images and masks) and as grasp poses. The asset
itself is obtained and prepared locally (`docs/ASSETS.md`). Terms:
`LICENSE-DATA.md`.

## Download the released raw sets

The hub repositories are named in `configs/hub.yaml` (`org: YOUR_HF_ORG` until
the release is published; the files carry the manifest contract described in
`docs/DATA_FORMAT.md`).

```bash
scripts/download_data.sh strict      # -> data/raw/strict_train      500 episode dirs, ~27 GB (5 shards of 100)
scripts/download_data.sh recovery    # -> data/raw/recovery_far1000  1000 episode dirs, ~55 GB (only the shards holding far-1000 members are fetched)
scripts/download_data.sh all         # both
# the data variants of the no_smooth / no_standoff presets (same sizes per set)
scripts/download_data.sh strict_nosmooth       # -> data/raw/strict_no_smoothing
scripts/download_data.sh recovery_nosmooth     # -> data/raw/recovery_no_smoothing
scripts/download_data.sh strict_nostandoff     # -> data/raw/strict_no_standoff
scripts/download_data.sh recovery_nostandoff   # -> data/raw/recovery_no_standoff
scripts/download_data.sh all_variants          # every set
```

`simretract-download data --set <name>` reads every `manifest.json` of the
set's repository (`configs/data.yaml` maps each set to a repository key of
`configs/hub.yaml`), extracts only the episode directories its selection
names, and reports the final count (500 for a strict set, 1000 for a
recovery set). Re-running skips shards that are already extracted.

## Build the cache

```bash
scripts/build_cache.sh strict --verify        # data/cache/strict_*    69,402 frames
scripts/build_cache.sh recovery --verify      # data/cache/recovery_*  106,868 frames
scripts/build_cache.sh strict_nosmooth        # data/cache/strict_nosmooth_*      (no recorded fingerprint)
scripts/build_cache.sh recovery_nosmooth      # data/cache/recovery_nosmooth_*
scripts/build_cache.sh strict_nostandoff      # data/cache/strict_nostandoff_*
scripts/build_cache.sh recovery_nostandoff    # data/cache/recovery_nostandoff_*
```

Each build writes the memmaps, the state/action/grasp arrays, the verdict
index, the plane-normal sidecar and a `meta.json` entry and spot-checks 20
frames bitwise against the PNGs. `--verify` additionally compares the frame
count and the array fingerprints with `configs/data.yaml` and exits non-zero
on a mismatch (use it for the downloaded sets only):

| set | frames | `_action.npy` md5[:8] | `_plane.npy` md5[:8] | `_gpose.npy` md5[:8] |
|---|---|---|---|---|
| strict | 69,402 | `7d0e2c3b` | `034cb12d` | `6a415e21` |
| recovery | 106,868 | `636630ba` | `93dda3b0` | `7f918202` |

Matching fingerprints mean the cache is frame-for-frame the one the released
checkpoints were trained on. A regenerated data set (below) produces different
fingerprints by construction. The no-smoothing and no-standoff sets build with
the same flags as the main pair (strict flags for the strict sets, recovery
flags for the recovery sets) but carry no recorded fingerprint
(`fingerprint: null` in `configs/data.yaml`): `--verify` reports that for
them and skips the comparison.

## Regenerate instead of downloading

```bash
scripts/build_assets.sh --accept-eula       # tool meshes + USD from the source file under assets/source/, checks (once; docs/ASSETS.md)
scripts/generate_strict.sh                  # data/raw/strict: quota mode, 125 kept episodes per bin
simretract-assemble strict --raw data/raw/strict --per-bin 125 \
    --out data/selections/strict_500.json --train-dir data/raw/strict_train
scripts/generate_recovery.sh                # data/raw/recovery: 4 draws per strict seed
simretract-assemble recovery --raw data/raw/recovery --strict-json data/selections/strict_500.json \
    --exact 4 --far 1000 --out-train data/selections/recovery_train_2000.json \
    --out-far data/selections/recovery_far1000.json --far-dir data/raw/recovery_far1000
scripts/build_cache.sh strict && scripts/build_cache.sh recovery   # no --verify: new data, new fingerprints
```

The strict recipe (`configs/generation/strict.yaml`) is the production recipe:
smoothed Bezier reach with a 3 cm standoff and a 6 cm on-axis control point,
speed scale 0.9, per-joint rate limit 2.5 deg per update, interference gate
(3 mm / 3 deg), pose keep rule (drop < 20 mm, presented-vs-released pose
< 10 mm / 10 deg), 20 candidate tries, grasp height U(-5, +5) cm, training
hand style. Quota mode pre-screens seeds without simulation and stops when
every bin holds 125 kept episodes; roughly half of the attempted seeds are
kept, so budget about 1000 attempted seeds. Bins that come up short can be
filled with `--fill-bin k --z-strict`.

The recovery recipe (`configs/generation/recovery.yaml`) replays each strict
seed's scene and stored grasp, interrupts the reach at a fraction
U(0.25, 0.75) of the path with a random excursion (3-8 cm, 8-25 deg, optional
partial grip close), and records the fresh reach back to the grasp. Most
draws are kept; `--topup --draws 8 --target-keeps 4` fills seeds that fall
short of four keeps. The far-1000 selection then takes the 1000 kept episodes
perturbed farthest from the grasp (perturbation fraction below about 0.5).

Expected counts for a regenerated set: 500 strict episodes (from roughly
1000-1300 attempts), about 2000 recovery draws of which about 1950 are kept,
1000 far-selected. Per-episode size is ~60 MB (~330 frames).

The no-smoothing and no-standoff sets were recorded by the same recorders
under different reach settings (no-smoothing: `blend false`, the grasp
candidate's own standoff, no joint-rate limit, speed scale 1.0; no-standoff:
`standoff 0`, no joint-rate limit, speed scale 1.0; the values are recorded in
the `recipe` of their strict selection files). The shipped generation
recipes and wrappers target the main pair; regenerating a variant means
running `simretract-generate` with those settings and assembling the
selections with the same commands.
