#!/usr/bin/env bash
# Download the released checkpoints listed in configs/hub.yaml into checkpoints/<row>/,
# verifying size and sha256. `all` fetches every row; --base-weights adds pi0_base.
# Needs configs/hub.yaml `org` filled in (the shipped placeholder is refused with one line).
# Usage: scripts/download_checkpoints.sh [baseline|no_dropout|no_recovery|half_data|no_plane|no_mask|no_smooth|no_standoff|smolvla|all] [--base-weights]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ROW="${1:-baseline}"; shift || true
BASE=0; EXTRA=()
for a in "$@"; do
  if [ "$a" = --base-weights ]; then BASE=1; else EXTRA+=("$a"); fi
done
if [ "$ROW" = all ]; then
  call_cli policy simretract.cli.download checkpoints ${EXTRA[@]+"${EXTRA[@]}"}
else
  call_cli policy simretract.cli.download checkpoints "$ROW" ${EXTRA[@]+"${EXTRA[@]}"}
fi
if [ "$BASE" = 1 ]; then
  call_cli policy simretract.cli.download base-weights pi0_base
fi
