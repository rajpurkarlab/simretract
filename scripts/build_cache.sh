#!/usr/bin/env bash
# Build the flat frame cache (memmaps + plane-normal sidecar) for one set with the
# production flags. Pass --verify to compare the frame count and md5 fingerprints
# with the reference build in configs/data.yaml (recorded for the main pair only;
# the other sets report that and skip the comparison; a regenerated set has
# different fingerprints by construction).
# Sets and their default raw directories (scripts/download_data.sh writes them):
#   strict / recovery                          data/raw/strict_train / data/raw/recovery_far1000
#   strict_nosmooth / recovery_nosmooth        data/raw/strict_no_smoothing / data/raw/recovery_no_smoothing
#   strict_nostandoff / recovery_nostandoff    data/raw/strict_no_standoff / data/raw/recovery_no_standoff
# Usage: scripts/build_cache.sh <set> [--raw DIR] [--out data/cache] [--verify] [flags]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
SET="${1:?usage: build_cache.sh <set> [flags]}"; shift
case "$SET" in
  strict)              RAW=data/raw/strict_train ;;
  recovery)            RAW=data/raw/recovery_far1000 ;;
  strict_nosmooth)     RAW=data/raw/strict_no_smoothing ;;
  recovery_nosmooth)   RAW=data/raw/recovery_no_smoothing ;;
  strict_nostandoff)   RAW=data/raw/strict_no_standoff ;;
  recovery_nostandoff) RAW=data/raw/recovery_no_standoff ;;
  *) echo "error: set must be one of strict recovery strict_nosmooth recovery_nosmooth strict_nostandoff recovery_nostandoff" >&2; exit 2 ;;
esac
run_cli policy simretract.cli.build_cache --set "$SET" --raw "$RAW" --out data/cache "$@"
