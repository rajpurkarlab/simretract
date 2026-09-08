#!/usr/bin/env bash
# Deploy a checkpoint on the robot: boots (or reuses) the policy server and the SAM 3
# mask server, verifies both, then runs the client. 'dry' = perception + inference
# only (arm read-only); 'go' = the arm moves (ENTER gate before motion).
# Launcher options (--preset, --plane-top-cut, --exec-h, ...) go before the mode and
# everything after the mode is passed to the client (e.g. --no-view --max-ticks 60 --yes).
# Without --preset the launcher uses a 1 cm metric crop. --plane-top-cut selects
# the legacy plane-only pixel fraction; --plane-top-cut-m 0 disables metric cropping.
# Usage: scripts/run_real.sh <checkpoint dir> [launcher flags] [go|dry] [client flags]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
CKPT="${1:?usage: run_real.sh <checkpoint dir> [launcher flags] [go|dry] [client flags]}"; shift
LAUNCH=(); MODE=""; CLIENT=()
for a in "$@"; do
  if [ -z "$MODE" ]; then
    case "$a" in
      go|dry) MODE="$a" ;;
      *) LAUNCH+=("$a") ;;
    esac
  else
    CLIENT+=("$a")
  fi
done
run_cli policy simretract.cli.run_real --ckpt "$CKPT" ${LAUNCH[@]+"${LAUNCH[@]}"} "${MODE:-dry}" ${CLIENT[@]+"${CLIENT[@]}"}
