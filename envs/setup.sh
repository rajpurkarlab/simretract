#!/usr/bin/env bash
# Create (or register) the simretract interpreter environments, install the
# package into each, record the interpreter paths and print versions.
#
#   envs/setup.sh                                   create simretract-sim / -policy / -mask with conda
#   envs/setup.sh --with-fastwam                    also create simretract-fastwam (optional row)
#   envs/setup.sh --only policy                     (re)do a single role
#   envs/setup.sh --reuse sim=/path/to/python policy=/path/to/python mask=/path/to/python [fastwam=...]
#                                                   register existing interpreters instead of creating envs
#
# The script is idempotent: existing environments are kept, pip re-installs are
# no-ops when the pins are already satisfied. Output: configs/envs.local.yaml
# (git-ignored; never commit it); an existing tool_source entry in that file,
# or SIMRETRACT_TOOL_SOURCE in the environment, is carried over (docs/ASSETS.md).
# Requires conda (Miniconda) and bash >= 4.4.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
ENVS_OUT="$REPO/configs/envs.local.yaml"

WITH_FASTWAM=0
ONLY=""
declare -A REUSE=()
ROLES=(sim policy mask)

usage() { sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --with-fastwam) WITH_FASTWAM=1 ;;
    --only) ONLY="$2"; shift ;;
    --reuse)
      shift
      while [ $# -gt 0 ] && [[ "$1" == *=* ]]; do
        role="${1%%=*}"; py="${1#*=}"
        REUSE["$role"]="$py"
        [ "$role" = fastwam ] && WITH_FASTWAM=1
        shift
      done
      continue
      ;;
    -h|--help) usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 2 ;;
  esac
  shift
done
[ "$WITH_FASTWAM" = 1 ] && ROLES+=(fastwam)
if [ -n "$ONLY" ]; then ROLES=("$ONLY"); fi

# python version and marker module per role
pyver()  { case "$1" in sim) echo 3.11 ;; policy) echo 3.12 ;; mask) echo 3.11 ;; fastwam) echo 3.12 ;; esac; }
marker() { case "$1" in sim) echo isaacsim ;; policy) echo lerobot ;; mask) echo ultralytics ;; fastwam) echo lerobot ;; esac; }

conda_base() {
  if [ -n "${CONDA_EXE:-}" ]; then "$CONDA_EXE" info --base
  elif command -v conda >/dev/null 2>&1; then conda info --base
  else echo "conda not found: install Miniconda or use --reuse with existing interpreters" >&2; exit 1
  fi
}

ensure_env() {  # role -> prints the interpreter path
  local role="$1" name="simretract-$1" base py
  base="$(conda_base)"
  py="$base/envs/$name/bin/python"
  if [ ! -x "$py" ]; then
    echo "== creating conda env $name (python $(pyver "$role"))" >&2
    if ! "$base/bin/conda" create -y -n "$name" "python=$(pyver "$role")" >&2; then
      echo "conda create failed. Recent Miniconda releases refuse non-interactive use of the default" >&2
      echo "channel until its terms are accepted once:" >&2
      echo "  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main" >&2
      exit 1
    fi
  else
    echo "== conda env $name exists" >&2
  fi
  echo "$py"
}

install_role() {  # role python
  local role="$1" py="$2" req="$HERE/$1.txt" tmp
  echo "== installing requirements for $role into $py"
  "$py" -m pip install --upgrade pip >/dev/null
  case "$role" in
    sim)
      # Isaac Sim first from the NVIDIA index, then the pins on top of it with the
      # isaacsim line filtered out: isaacsim-kernel declares numpy==1.26.0, so
      # requesting both at once makes pip's resolver refuse the numpy 1.26.4 pin,
      # while installing the pins afterwards only prints a conflict warning.
      local isaac_spec
      isaac_spec="$(grep -E '^isaacsim' "$req")"
      "$py" -m pip install --no-cache-dir "$isaac_spec" --extra-index-url https://pypi.nvidia.com
      tmp="$(mktemp)"
      grep -vE '^isaacsim' "$req" > "$tmp"
      "$py" -m pip install -r "$tmp"
      rm -f "$tmp"
      ;;
    policy)
      # lerobot is installed separately with --no-deps (pinned commit, see LEROBOT_PIN.md).
      tmp="$(mktemp)"
      grep -vE '^lerobot' "$req" > "$tmp"
      "$py" -m pip install -r "$tmp"
      rm -f "$tmp"
      "$py" -m pip install --no-deps "$(grep -E '^lerobot' "$req")"
      ;;
    *)
      "$py" -m pip install -r "$req"
      ;;
  esac
  "$py" -m pip install --no-deps -e "$REPO"
}

declare -A PY=()
for role in "${ROLES[@]}"; do
  if [ -n "${REUSE[$role]:-}" ]; then
    py="${REUSE[$role]}"
    [ -x "$py" ] || { echo "interpreter for $role not executable: $py" >&2; exit 1; }
    echo "== registering existing $role interpreter $py"
    "$py" -m pip install --no-deps -e "$REPO"
  else
    py="$(ensure_env "$role")"
    install_role "$role" "$py"
  fi
  PY[$role]="$py"
done

# Preserve roles not touched in this invocation and the optional tool_source
# (the licensed tool mesh registered by hand or by SIMRETRACT_TOOL_SOURCE=...).
declare -A OLD=()
TOOL_SOURCE="${SIMRETRACT_TOOL_SOURCE:-}"
if [ -f "$ENVS_OUT" ]; then
  while IFS=': ' read -r k v; do
    [[ "$k" =~ ^([a-z]+)_python$ ]] && OLD["${BASH_REMATCH[1]}"]="$v"
    [ "$k" = tool_source ] && [ -z "$TOOL_SOURCE" ] && TOOL_SOURCE="$v"
  done < "$ENVS_OUT"
fi
{
  echo "# written by envs/setup.sh; overrides configs/envs.yaml (git-ignored)"
  for role in sim policy mask fastwam; do
    v="${PY[$role]:-${OLD[$role]:-}}"
    [ -n "$v" ] && echo "${role}_python: $v"
  done
  [ -n "$TOOL_SOURCE" ] && echo "tool_source: $TOOL_SOURCE"
  true
} > "$ENVS_OUT"
echo "== wrote $ENVS_OUT"

echo "== versions"
failed_roles=()
for role in "${ROLES[@]}"; do
  py="${PY[$role]}"
  if ! "$py" - "$role" "$(marker "$role")" <<'PYEOF'
import importlib, importlib.metadata as md, sys
role, marker = sys.argv[1], sys.argv[2]
def ver(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "-"
pkgs = {"sim": ["isaacsim", "numpy", "scipy", "trimesh", "opencv-python-headless"],
        "policy": ["torch", "torchvision", "lerobot", "transformers", "safetensors", "huggingface_hub", "numpy"],
        "mask": ["torch", "ultralytics", "clip", "timm", "opencv-python", "numpy"],
        "fastwam": ["torch", "lerobot", "transformers", "diffusers"]}[role]
line = " ".join(f"{p}={ver(p)}" for p in pkgs)
try:
    importlib.import_module(marker)
    if role == "mask":
        for dependency in ("clip", "timm"):
            importlib.import_module(dependency)
    ok = "ok"
except Exception as exc:  # noqa: BLE001
    ok = f"FAILED ({exc})"
print(f"{role:8s} python={sys.version.split()[0]} {line} import {marker}: {ok}")
sys.exit(0 if ok == "ok" else 1)
PYEOF
  then
    failed_roles+=("$role")
  fi
done
if [ "${#failed_roles[@]}" -gt 0 ]; then
  echo "error: environment validation failed for: ${failed_roles[*]}. Fix the imports above and rerun setup." >&2
  exit 1
fi
echo "== done."
echo "   The simretract-* and hf commands live in the policy environment; activate it first:"
echo "       conda activate simretract-policy"
echo "   (the scripts/*.sh wrappers pick the right interpreter themselves and need no activation)"
echo "   Isaac Sim needs OMNI_KIT_ACCEPT_EULA=YES in the environment of every simulation command."
echo "   $ENVS_OUT holds machine-specific absolute paths: it is git-ignored and must never be committed."
