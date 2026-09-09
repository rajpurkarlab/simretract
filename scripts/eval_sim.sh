#!/usr/bin/env bash
# Run the simulation benchmark (core-50 seeds, first-attempt protocol, reference settings:
# EXEC_H auto, smooth 0.45, no plane top-cut, test hand style).
# Usage: scripts/eval_sim.sh <checkpoint dir> [extra simretract-eval-sim flags]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
CKPT="${1:?usage: eval_sim.sh <checkpoint dir> [flags]}"; shift
NAME="$(basename "$CKPT")"; [ "$NAME" = final ] && NAME="$(basename "$(dirname "$CKPT")")"
run_cli policy simretract.cli.eval_sim --ckpt "$CKPT" --seeds data/selections/test_core50.json --passes 1 --exec-h auto --port 8767 --out "eval/$NAME" "$@"
