#!/usr/bin/env bash
# Build the simulation assets locally from your licensed copy of the retractor
# (docs/ASSETS.md): the source file under assets/source/<pack>/ (or the
# tool_source override) is checksummed and converted into the tool meshes,
# checked against meta.json and the recipe fingerprints, the tool USD is built
# (Isaac Sim required) and the tracked grasp candidates are re-sampled and
# compared. Any simretract-build-assets flag can follow, e.g. --from-mesh FILE
# for a copy kept elsewhere or --accept-meta for a different asset. The gripper
# LUT is tracked; re-render it with `simretract-build-assets --lut --accept-eula`
# when the fingers change.
# Usage: scripts/build_assets.sh [--accept-eula] [more simretract-build-assets flags]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_cli policy simretract.cli.build_assets --tool retractor_rake "$@"
