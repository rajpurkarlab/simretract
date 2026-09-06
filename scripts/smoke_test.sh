#!/usr/bin/env bash
# Run the smoke-test tiers with pytest and print the wall time of each.
#   unit     no GPU, no Isaac                          (< 1 min)
#   gpu      GPU: synthetic cache, ACT smoke training, policy server (~ 1 min)
#   sim      Isaac Sim + GPU: asset build from the source file under assets/source/ or the
#            tool_source override (skips without one), generation, cache, benchmark
#            (~ 10 min, up to ~ 25)
#   real-dry client dry run; skipped without SAM 3 weights, robot address, D405
#   release  release gates (licence files, hub ids, asset exclusion, release wording);
#            fails until they are met
#   all      unit + gpu + sim
# Every tier's pytest process runs in the policy environment; the CLIs under
# test delegate simulator work to the sim interpreter recorded in
# configs/envs.local.yaml by envs/setup.sh. Requires bash >= 4.4.
# Usage: scripts/smoke_test.sh [unit|gpu|sim|real-dry|release|all] [extra pytest flags]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
TIER="${1:-unit}"; shift || true
EXTRA=("$@")
cd "$SIMRETRACT_ROOT"

run_tier() {  # run_tier ROLE DIR MARKER "expected wall time"
  local role="$1" dir="$2" marker="$3" eta="$4" py t0 rc
  py="$(interp "$role")" || exit 1
  echo "== smoke tier '$marker' ($dir) under $py -- expected wall time $eta"
  t0=$(date +%s)
  set +e
  "$py" -m pytest "$dir" -m "$marker" -q --durations=0 -p no:cacheprovider ${EXTRA[@]+"${EXTRA[@]}"}
  rc=$?
  set -e
  echo "== smoke tier '$marker': $([ $rc -eq 0 ] && echo PASS || echo "FAIL (rc $rc)") in $(( $(date +%s) - t0 )) s"
  return $rc
}

case "$TIER" in
  unit)     run_tier policy tests/unit unit "< 1 min" ;;
  gpu)      run_tier policy tests/gpu gpu "~ 1 min" ;;
  sim)      run_tier policy tests/sim sim "~ 10 min (up to ~ 25 min with a retried Kit worker)" ;;
  real-dry) run_tier policy tests/real_dry real_dry "~ 2 min (skips without hardware)" ;;
  release)  run_tier policy tests/release release "seconds" ;;
  all)
    run_tier policy tests/unit unit "< 1 min"
    run_tier policy tests/gpu gpu "~ 1 min"
    run_tier policy tests/sim sim "~ 10 min (up to ~ 25 min with a retried Kit worker)"
    ;;
  *) echo "usage: scripts/smoke_test.sh [unit|gpu|sim|real-dry|release|all]" >&2; exit 2 ;;
esac
echo "== smoke tier(s) '$TIER' passed"
