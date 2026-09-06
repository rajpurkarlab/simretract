#!/usr/bin/env bash
# Record perturbation-recovery episodes (4 draws per strict seed) from the assembled
# strict set (configs/generation/recovery.yaml).
# Usage: scripts/generate_recovery.sh [--strict-json data/selections/strict_500.json] [--winners data/raw/strict_train] [--out data/raw/recovery] ...
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_cli policy simretract.cli.generate recovery --strict-json data/selections/strict_500.json --winners data/raw/strict_train --out data/raw/recovery --draws 4 --range 0:2000 --pert-frac-lo 0.25 --pert-frac-hi 0.75 --parallel 3 "$@"
