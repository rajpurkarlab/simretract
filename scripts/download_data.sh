#!/usr/bin/env bash
# Download the released raw episode sets from the hub repos in configs/hub.yaml
# (only the shards holding selected episodes are fetched):
#   strict              -> data/raw/strict_train            (the strict-500 selection)
#   recovery            -> data/raw/recovery_far1000        (the far-1000 selection)
#   strict_nosmooth     -> data/raw/strict_no_smoothing     (no-smoothing strict-500)
#   recovery_nosmooth   -> data/raw/recovery_no_smoothing   (no-smoothing far-1000)
#   strict_nostandoff   -> data/raw/strict_no_standoff      (no-standoff strict-500)
#   recovery_nostandoff -> data/raw/recovery_no_standoff    (no-standoff far-1000)
#   all                 the main pair (strict + recovery); all_variants = every set
# Usage: scripts/download_data.sh [<set>|all|all_variants] [extra simretract-download flags]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
SET="${1:-all}"; shift || true
raw_dir() {
  case "$1" in
    strict)              echo data/raw/strict_train ;;
    recovery)            echo data/raw/recovery_far1000 ;;
    strict_nosmooth)     echo data/raw/strict_no_smoothing ;;
    recovery_nosmooth)   echo data/raw/recovery_no_smoothing ;;
    strict_nostandoff)   echo data/raw/strict_no_standoff ;;
    recovery_nostandoff) echo data/raw/recovery_no_standoff ;;
    *) return 1 ;;
  esac
}
case "$SET" in
  all)          SETS=(strict recovery) ;;
  all_variants) SETS=(strict recovery strict_nosmooth recovery_nosmooth strict_nostandoff recovery_nostandoff) ;;
  *)
    raw_dir "$SET" >/dev/null || { echo "usage: scripts/download_data.sh [strict|recovery|strict_nosmooth|recovery_nosmooth|strict_nostandoff|recovery_nostandoff|all|all_variants]" >&2; exit 2; }
    SETS=("$SET") ;;
esac
for s in "${SETS[@]}"; do
  call_cli policy simretract.cli.download data --set "$s" --out "$(raw_dir "$s")" "$@"
done
