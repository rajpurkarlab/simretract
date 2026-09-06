#!/usr/bin/env bash
# Shared helpers for the scripts/ wrappers. Sourced, never executed directly.
#
# Resolves the repository root and the per-role interpreters the same way
# simretract.config does: SIMRETRACT_<ROLE>_PYTHON, then the machine-specific
# configs/envs.local.yaml written by envs/setup.sh, then the tracked template
# configs/envs.yaml.

SIMRETRACT_ROOT="${SIMRETRACT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export SIMRETRACT_ROOT

# interp ROLE -> prints the interpreter path for ROLE (sim|policy|mask|fastwam).
interp() {
  local role="$1" var value f
  var="SIMRETRACT_$(echo "$role" | tr '[:lower:]' '[:upper:]')_PYTHON"
  value="${!var:-}"
  for f in "$SIMRETRACT_ROOT/configs/envs.local.yaml" "$SIMRETRACT_ROOT/configs/envs.yaml"; do
    [ -n "$value" ] && break
    [ -f "$f" ] || continue
    value="$(sed -n "s/^${role}_python:[[:space:]]*['\"]\{0,1\}\([^'\"#]*\)['\"]\{0,1\}[[:space:]]*$/\1/p" \
             "$f" | head -1 | sed 's/[[:space:]]*$//')"
    [ "$value" = CHANGE_ME ] && value=""
  done
  if [ -z "$value" ] || [ ! -x "$value" ]; then
    echo "error: no usable ${role} interpreter (set ${var} or run envs/setup.sh to write configs/envs.local.yaml)" >&2
    return 1
  fi
  printf '%s\n' "$value"
}

# call_cli ROLE MODULE ARGS... -> run the CLI module under the ROLE interpreter (returns).
call_cli() {
  local role="$1" module="$2"; shift 2
  local py; py="$(interp "$role")" || return 1
  (cd "$SIMRETRACT_ROOT" && "$py" -m "$module" "$@")
}

# run_cli ROLE MODULE ARGS... -> exec the CLI module under the ROLE interpreter.
run_cli() {
  local role="$1" module="$2"; shift 2
  local py; py="$(interp "$role")" || exit 1
  cd "$SIMRETRACT_ROOT" || exit 1
  exec "$py" -m "$module" "$@"
}
