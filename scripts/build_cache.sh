#!/usr/bin/env bash
# Build the flat frame cache (memmaps + plane-normal sidecar) for one set with the
# production flags. Pass --verify to compare the frame count and md5 fingerprints
# with the reference build in configs/data.yaml (only meaningful for the released
# raw sets; a regenerated set has different fingerprints by construction).
# Usage: scripts/build_cache.sh strict|recovery [--raw DIR] [--out data/cache] [--verify] [flags]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
SET="${1:?usage: build_cache.sh strict|recovery [flags]}"; shift
case "$SET" in
  strict)   RAW=data/raw/strict_train ;;
  recovery) RAW=data/raw/recovery_far1000 ;;
  *) echo "error: set must be strict or recovery" >&2; exit 2 ;;
esac
run_cli policy simretract.cli.build_cache --set "$SET" --raw "$RAW" --out data/cache "$@"
