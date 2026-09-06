"""Central configuration for the ``simretract`` package.

Every filesystem location, interpreter path and Hugging Face hub identifier
used by the package comes from the YAML files under ``configs/``:

* ``configs/paths.yaml``          storage roots (raw data, caches, checkpoints, runs, eval)
* ``configs/envs.yaml``           interpreter paths per role and the optional ``tool_source``
                                  override (template; ``envs/setup.sh`` writes the
                                  machine-specific ``configs/envs.local.yaml``)
* ``configs/assets/<tool>.yaml``  tool asset recipe: the purchased source file, its conversion
                                  and the fingerprints of the released mesh
* ``configs/hub.yaml``            hub organisation, repository ids, checkpoint and
                                  base-weight names
* ``configs/data.yaml``           dataset ids, selection files, cache flags, fingerprints
* ``configs/train/presets.yaml``  named training rows (the reference rows and optional rows)
* ``configs/generation/*.yaml``   expert-recorder recipes
* ``configs/eval/sim.yaml``       simulation benchmark protocol

Environment variables override individual entries so one checkout can serve
several storage layouts: ``SIMRETRACT_ROOT`` relocates every storage root,
``SIMRETRACT_DATA_RAW`` (and friends) relocate one root,
``SIMRETRACT_<ROLE>_PYTHON`` overrides one interpreter and
``SIMRETRACT_TOOL_SOURCE`` overrides the licensed tool mesh to convert locally.

The module has no dependency beyond PyYAML so it can be imported in every
role environment, including the Isaac Sim one.
"""

from __future__ import annotations

import functools
import importlib.util
import logging
import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, TypeVar

import yaml

log = logging.getLogger(__name__)

T = TypeVar("T")

#: Process roles and the module whose importability proves the interpreter is right.
ROLES: dict[str, str] = {
    "sim": "isaacsim",
    "policy": "lerobot",
    "mask": "ultralytics",
    "fastwam": "lerobot",
}

#: Keys of ``paths()`` that are storage roots (relocatable with ``SIMRETRACT_ROOT``).
STORAGE_KEYS: tuple[str, ...] = ("data_raw", "data_cache", "checkpoints", "runs", "eval")
#: Keys of ``paths()`` that always live inside the repository checkout.
REPO_KEYS: tuple[str, ...] = ("assets", "selections", "third_party", "configs")

#: Keys of a resolved preset that are descriptive rather than trainer flags.
PRESET_META_KEYS: frozenset[str] = frozenset({"name", "description", "reference"})

_PLACEHOLDER = "CHANGE_ME"
_SUBST_RE = re.compile(r"\$\{([A-Za-z0-9_.]+)\}")


class ConfigError(RuntimeError):
    """A configuration file or environment variable is missing or invalid."""


# --------------------------------------------------------------------------- roots


def repo_root() -> Path:
    """Return the repository checkout that contains this package.

    The package is installed editable, so the checkout is the parent of the
    package directory.  ``SIMRETRACT_REPO`` overrides the discovery (useful when
    the package is run from a wheel); the override must contain ``configs/``.
    """
    override = os.environ.get("SIMRETRACT_REPO")
    root = Path(override).expanduser().resolve() if override else Path(__file__).resolve().parent.parent
    if not (root / "configs").is_dir():
        raise ConfigError(
            f"repository root {root} has no configs/ directory; "
            "install the package editable from a checkout or set SIMRETRACT_REPO"
        )
    return root


def storage_root() -> Path:
    """Return the directory relative storage roots resolve against.

    Defaults to the repository root; ``SIMRETRACT_ROOT`` moves raw data, caches,
    checkpoints, runs and evaluation outputs to another disk in one go.
    """
    override = os.environ.get("SIMRETRACT_ROOT")
    return Path(override).expanduser().resolve() if override else repo_root()


def config_dir() -> Path:
    """Return ``<repo>/configs``."""
    return repo_root() / "configs"


def config_path(rel: str) -> Path:
    """Return the absolute path of a file under ``configs/`` (``.yaml`` appended if missing).

    The suffix is appended, never substituted, so dotted names such as
    ``envs.local`` resolve to ``envs.local.yaml``.
    """
    rel_str = str(rel)
    if not rel_str.endswith(".yaml"):
        rel_str += ".yaml"
    return config_dir() / rel_str


# --------------------------------------------------------------------------- yaml


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML mapping; an empty file yields ``{}``.

    Raises :class:`ConfigError` when the file is missing or its top level is not
    a mapping, so callers get one actionable message instead of a stack of
    ``KeyError``.
    """
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"configuration file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{p}: top level must be a mapping, got {type(data).__name__}")
    return data


def load_config(rel: str) -> dict[str, Any]:
    """Load ``configs/<rel>.yaml`` (fresh read; callers cache if they need to)."""
    return load_yaml(config_path(rel))


# --------------------------------------------------------------------------- env helpers


def _parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"not a boolean: {value!r}")


def env_or(name: str | Sequence[str], default: T, cast: Callable[[str], Any] | None = None) -> T | Any:
    """Return an environment override for ``name`` or ``default``.

    ``name`` may be a single variable or a sequence of variables tried in order
    (the first one that is set wins), which is how CLI flags honour both the
    ``SIMRETRACT_*`` name and the legacy environment name of the same knob::

        standoff = env_or(("SIMRETRACT_STANDOFF", "STANDOFF"), 0.03, float)

    ``cast`` converts the raw string; when omitted it is inferred from the type
    of ``default`` (``bool`` accepts 1/0/true/false/yes/no/on/off).  A value that
    fails to cast raises :class:`ConfigError` naming the variable.
    """
    names: Iterable[str] = (name,) if isinstance(name, str) else name
    for n in names:
        raw = os.environ.get(n)
        if raw is None:
            continue
        caster: Callable[[str], Any] | None = cast
        if caster is None:
            if isinstance(default, bool):
                caster = _parse_bool
            elif default is not None:
                caster = type(default)
        try:
            return caster(raw) if caster is not None else raw
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"environment variable {n}={raw!r}: {exc}") from exc
    return default


def require_kit_eula() -> None:
    """Fail fast when the Isaac Sim EULA has not been accepted.

    Headless Isaac Sim refuses to boot without ``OMNI_KIT_ACCEPT_EULA=YES`` and
    the error it prints is buried in the Kit log; every Kit-side module calls
    this before creating ``SimulationApp`` so the user sees one clear line.
    """
    if os.environ.get("OMNI_KIT_ACCEPT_EULA", "").upper() != "YES":
        raise ConfigError(
            "Isaac Sim needs the NVIDIA Omniverse EULA accepted: export OMNI_KIT_ACCEPT_EULA=YES "
            "before running any simulation command"
        )


# --------------------------------------------------------------------------- substitution


def _substitute(value: Any, context: dict[str, Any]) -> Any:
    """Expand ``${a.b.c}`` references in strings against a nested context.

    Used for hub ids (``${org}/repo``) and preset values such as
    ``${paths.checkpoints}/pi0_base`` so YAML files never carry absolute paths.
    """
    if isinstance(value, str):

        def repl(m: re.Match[str]) -> str:
            cur: Any = context
            for part in m.group(1).split("."):
                if not isinstance(cur, dict) or part not in cur:
                    raise ConfigError(f"unknown reference ${{{m.group(1)}}}")
                cur = cur[part]
            return str(cur)

        return _SUBST_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _substitute(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, context) for v in value]
    return value


# --------------------------------------------------------------------------- paths


def paths() -> dict[str, Path]:
    """Return every configured location as an absolute :class:`Path`.

    Keys: ``data_raw``, ``data_cache``, ``checkpoints``, ``runs``, ``eval``
    (storage roots; relative entries resolve against :func:`storage_root`,
    each overridable by ``SIMRETRACT_<KEY>``) and ``assets``, ``selections``,
    ``third_party``, ``configs`` (always inside the checkout).  Directories are
    not created here; the CLI that writes to a root creates it.
    """
    cfg = load_config("paths")
    storage = cfg.get("storage", {})
    repo = cfg.get("repo", {})
    out: dict[str, Path] = {}
    for key in STORAGE_KEYS:
        raw = os.environ.get(f"SIMRETRACT_{key.upper()}") or storage.get(key)
        if raw is None:
            raise ConfigError(f"configs/paths.yaml: storage.{key} is missing")
        p = Path(raw).expanduser()
        out[key] = p if p.is_absolute() else (storage_root() / p)
    for key in REPO_KEYS:
        raw = repo.get(key, key)
        p = Path(raw).expanduser()
        out[key] = p if p.is_absolute() else (repo_root() / p)
    return {k: v.resolve() for k, v in out.items()}


def selection_path(name: str) -> Path:
    """Return ``data/selections/<name>`` (``.json`` added if missing)."""
    n = name if name.endswith(".json") else name + ".json"
    return paths()["selections"] / n


# --------------------------------------------------------------------------- hub


def hub() -> dict[str, Any]:
    """Return ``configs/hub.yaml`` with ``${org}`` references expanded.

    The organisation placeholder is left as-is (``YOUR_HF_ORG``) so download
    commands can detect an unconfigured hub and say so.
    """
    cfg = load_config("hub")
    return _substitute(cfg, cfg)


def hub_is_configured() -> bool:
    """True when the hub organisation placeholder has been replaced."""
    return str(hub().get("org", "")).strip() not in {"", "YOUR_HF_ORG", _PLACEHOLDER}


# --------------------------------------------------------------------------- data


def data_config() -> dict[str, Any]:
    """Return ``configs/data.yaml`` (dataset ids, selection files, fingerprints)."""
    return load_config("data")


def generation_recipe(kind: str) -> dict[str, Any]:
    """Return the expert-recorder recipe ``configs/generation/<kind>.yaml`` (strict|recovery)."""
    if kind not in {"strict", "recovery"}:
        raise ConfigError(f"unknown generation recipe {kind!r}; expected strict or recovery")
    return load_config(f"generation/{kind}")


def eval_protocol() -> dict[str, Any]:
    """Return the simulation benchmark protocol ``configs/eval/sim.yaml``."""
    return load_config("eval/sim")


# --------------------------------------------------------------------------- interpreters


def _envs_file() -> Path:
    """The machine-specific interpreter table, falling back to the template."""
    local = config_path("envs.local")
    return local if local.is_file() else config_path("envs")


@functools.lru_cache(maxsize=None)
def _validate_interpreter(python: str, module: str) -> str | None:
    """Return None when ``python`` can import ``module``, else the failure text."""
    if Path(python).resolve() == Path(sys.executable).resolve():
        return None if importlib.util.find_spec(module) is not None else f"cannot import {module}"
    try:
        proc = subprocess.run(
            [python, "-c", f"import {module}"],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except OSError as exc:
        return str(exc)
    except subprocess.TimeoutExpired:
        return f"import {module} timed out"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        return tail[-1] if tail else f"import {module} failed"
    return None


def interpreter(role: str, *, validate: bool = True) -> Path:
    """Return the Python interpreter for ``role`` (``sim``, ``policy``, ``mask``, ``fastwam``).

    Resolution order: ``SIMRETRACT_<ROLE>_PYTHON``, then ``configs/envs.local.yaml``
    (written by ``envs/setup.sh``), then the tracked template ``configs/envs.yaml``.
    With ``validate`` the interpreter must import the role's marker module
    (``isaacsim`` / ``lerobot`` / ``ultralytics``); the check runs once per
    process and failures raise :class:`ConfigError` with the fix.
    """
    if role not in ROLES:
        raise ConfigError(f"unknown interpreter role {role!r}; expected one of {sorted(ROLES)}")
    env_name = f"SIMRETRACT_{role.upper()}_PYTHON"
    raw = os.environ.get(env_name)
    source = env_name
    if not raw:
        envs_file = _envs_file()
        raw = load_yaml(envs_file).get(f"{role}_python")
        source = str(envs_file)
    if not raw or str(raw).strip() in {"", _PLACEHOLDER}:
        raise ConfigError(
            f"no {role} interpreter configured ({role}_python is unset in {source}): "
            f"run envs/setup.sh (or envs/setup.sh --reuse {role}=/path/to/python), "
            f"or export {env_name}"
        )
    py = Path(str(raw)).expanduser()
    if not py.is_file():
        raise ConfigError(f"{role} interpreter {py} (from {source}) does not exist; re-run envs/setup.sh")
    if validate:
        problem = _validate_interpreter(str(py), ROLES[role])
        if problem is not None:
            raise ConfigError(
                f"{role} interpreter {py} cannot import {ROLES[role]} ({problem}): "
                f"re-run envs/setup.sh --only {role} or point {env_name} at the right environment"
            )
    return py


# --------------------------------------------------------------------------- tool source


def tool_source() -> Path | None:
    """Return the ``tool_source`` override (a licensed tool mesh kept outside the
    drop-in folder), or ``None`` when none is configured.

    Resolution order: ``SIMRETRACT_TOOL_SOURCE``, then ``tool_source`` in
    ``configs/envs.local.yaml`` (git-ignored), then the tracked template
    ``configs/envs.yaml``.  Without an override the asset build takes the
    file the tool's recipe names under ``assets/source/<pack>/``
    (:func:`simretract.sim.asset_recipe.resolve_source`).  The file is never
    part of the repository: ``simretract-build-assets`` converts it locally
    into ``assets/tools/<tool>/`` (see ``docs/ASSETS.md``).  An unset or
    placeholder value yields ``None``; a configured path that does not exist
    raises :class:`ConfigError`.
    """
    raw = os.environ.get("SIMRETRACT_TOOL_SOURCE")
    source = "SIMRETRACT_TOOL_SOURCE"
    if not raw:
        envs_file = _envs_file()
        raw = load_yaml(envs_file).get("tool_source")
        source = str(envs_file)
    if not raw or str(raw).strip() in {"", _PLACEHOLDER}:
        return None
    p = Path(str(raw)).expanduser()
    if not p.is_file():
        raise ConfigError(
            f"tool_source {p} (from {source}) does not exist; point it at your copy of the "
            "retractor pack or unset it (docs/ASSETS.md)"
        )
    return p


# --------------------------------------------------------------------------- vendor robot


def trossen_root() -> Path:
    """Return the ``trossen_ai_isaac`` submodule checkout."""
    root = paths()["third_party"] / "trossen_ai_isaac"
    if not (root / "scripts" / "controller.py").is_file():
        raise ConfigError(
            f"vendor robot package missing at {root}: run `git submodule update --init --recursive`"
        )
    return root


def trossen_scripts_path() -> Path:
    """Return the directory that holds the vendor ``controller.py`` (TrossenAIController)."""
    return trossen_root() / "scripts"


def trossen_robot_usd() -> Path:
    """Return the vendor WidowX-AI follower USD used by the simulation scene."""
    usd = trossen_root() / "assets" / "robots" / "wxai" / "wxai_follower.usd"
    if not usd.is_file():
        raise ConfigError(f"robot USD missing: {usd} (submodule incomplete?)")
    return usd


def ensure_trossen_on_path() -> Path:
    """Make ``import controller`` resolve to the vendor controller; returns the scripts dir."""
    scripts = trossen_scripts_path()
    s = str(scripts)
    if s not in sys.path:
        sys.path.insert(0, s)
    return scripts


# --------------------------------------------------------------------------- presets


def _preset_context() -> dict[str, Any]:
    return {
        "paths": {k: str(v) for k, v in paths().items()},
        "hub": hub(),
    }


def preset_names() -> list[str]:
    """Return the preset names defined in ``configs/train/presets.yaml`` in file order."""
    return list(load_config("train/presets").get("presets", {}).keys())


def preset(name: str) -> dict[str, Any]:
    """Resolve a named training row to its complete flag set.

    The result is ``defaults`` from ``configs/train/presets.yaml`` updated with
    the preset's overrides, with ``${paths.*}`` / ``${hub.*}`` references
    expanded, plus the descriptive keys in :data:`PRESET_META_KEYS`.  Every
    trainer flag is present in every preset, so consumers never need to know
    the defaults.
    """
    cfg = load_config("train/presets")
    presets = cfg.get("presets", {})
    if name not in presets:
        raise KeyError(f"unknown preset {name!r}; available: {', '.join(presets)}")
    flags: dict[str, Any] = dict(cfg.get("defaults", {}))
    flags.update(presets[name] or {})
    flags["name"] = name
    unknown = set(flags) - set(cfg.get("defaults", {})) - PRESET_META_KEYS
    if unknown:
        raise ConfigError(f"preset {name!r} sets keys missing from defaults: {sorted(unknown)}")
    return _substitute(flags, _preset_context())


def preset_flags(name: str) -> dict[str, Any]:
    """Like :func:`preset` but without the descriptive keys."""
    return {k: v for k, v in preset(name).items() if k not in PRESET_META_KEYS}
