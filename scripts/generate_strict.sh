#!/usr/bin/env bash
# Record strict (expert hand-over) episodes with the production recipe
# (configs/generation/strict.yaml): quota mode fills 4 grasp-height bins with 125 kept
# episodes each. Usage: scripts/generate_strict.sh [--out data/raw/strict] [--seed-start N] [--parallel 3] ...
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_cli policy simretract.cli.generate strict --out data/raw/strict --quota 125 --bins 4 --z-range -0.05 0.05 --parallel 3 "$@"
