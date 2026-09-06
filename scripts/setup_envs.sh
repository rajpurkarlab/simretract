#!/usr/bin/env bash
# Create the conda environments (simretract-sim, simretract-policy, simretract-mask,
# optional simretract-fastwam), install the package into each, write configs/envs.local.yaml.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
exec bash "$SIMRETRACT_ROOT/envs/setup.sh" "$@"
