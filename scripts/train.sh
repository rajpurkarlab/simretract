#!/usr/bin/env bash
# Train one preset row on the cache with the reference H200 schedule
# (batch 16 x accum 4, 15k steps). Usage: scripts/train.sh <preset> [extra simretract-train flags]
# Presets: baseline no_dropout no_recovery half_data no_plane no_mask no_smooth no_standoff smolvla (docs/ABLATIONS.md)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
PRESET="${1:?usage: train.sh <preset> [flags]}"; shift
run_cli policy simretract.cli.train --preset "$PRESET" --cache data/cache --out "runs/$PRESET" "$@"
