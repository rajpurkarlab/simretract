#!/usr/bin/env bash
# Download the released raw episode sets from the hub repos in configs/hub.yaml:
#   strict   -> data/raw/strict_train      (the strict-500 selection)
#   recovery -> data/raw/recovery_far1000  (only the shards holding far-1000 members)
# Usage: scripts/download_data.sh [strict|recovery|all] [extra simretract-download flags]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
SET="${1:-all}"; shift || true
case "$SET" in
  strict)   call_cli policy simretract.cli.download data --set strict --out data/raw/strict_train "$@" ;;
  recovery) call_cli policy simretract.cli.download data --set recovery --out data/raw/recovery_far1000 "$@" ;;
  all)
    call_cli policy simretract.cli.download data --set strict --out data/raw/strict_train "$@"
    call_cli policy simretract.cli.download data --set recovery --out data/raw/recovery_far1000 "$@"
    ;;
  *) echo "usage: scripts/download_data.sh [strict|recovery|all]" >&2; exit 2 ;;
esac
