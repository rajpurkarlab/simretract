"""Downloads from the model/data hub: raw episode shards for a selection,
checkpoint directories (final rows and the pi0 base weights) and
third-party base weights / tokenizers.

Every repository id comes from ``configs/hub.yaml`` through
``simretract.config.hub()``::

    org: YOUR_HF_ORG
    repos:
      checkpoints:  <org>/...      # model repo: one directory per row (+ pi0_base)
      strict_raw:   <org>/...      # dataset repo: shards + manifest.json
      recovery_raw: <org>/...
    checkpoints:
      baseline: {path: baseline, files: [config.json, model.safetensors, norm.json],
                 sha256: {model.safetensors: <hex>}, sizes: {model.safetensors: <bytes>}}
    base_weights:
      smolvla: lerobot/smolvla_base          # third-party ids (some gated)
      paligemma_tokenizer: google/paligemma-3b-pt-224

Raw dataset repos follow the shard contract: ``manifest.json`` maps
``shard_NNNN.tar`` to the episode directory names it contains
(``{"shards": {"shard_0000.tar": ["ep_200005", ...]}}``) and each tar holds
those directories at its top level. A repository may carry further
``<dir>/manifest.json`` files (later additions) whose shard names are
relative to that directory; :func:`download_raw` reads every manifest,
fetches only the shards intersecting the wanted episodes and extracts only
the wanted members, so a far-1000 recovery subset never pulls the whole
repository. ``simretract-download data --set <name>`` takes any cache set
of ``configs/data.yaml`` (``strict``, ``recovery``, ``strict_nosmooth``,
``recovery_nosmooth``, ``strict_nostandoff``, ``recovery_nostandoff``); the
set's ``hub_repo`` names the repos key and its ``selection`` the episodes.

Checkpoint verification compares sizes and sha256 with the entries of the
row when they are filled in and warns (without failing) when they are
absent, because the tables are completed after the release artefacts are
frozen. Gated third-party assets are fetched with the user's own hub login
(``huggingface-cli login`` after accepting the licence); no mirror exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import tarfile
from pathlib import Path
from typing import Any, Iterable

from simretract.config import ConfigError

log = logging.getLogger(__name__)

PLACEHOLDER_ORG = "YOUR_HF_ORG"


def load_hub_config() -> dict[str, Any]:
    """``configs/hub.yaml`` with ``${org}`` expanded."""
    from simretract import config as _cfg

    return _cfg.hub()


def _require_repo(cfg: dict[str, Any], key: str) -> str:
    repo = (cfg.get("repos") or {}).get(key)
    if not repo or PLACEHOLDER_ORG in str(repo) or str(cfg.get("org", "")).strip() in {"", PLACEHOLDER_ORG}:
        raise ConfigError(f"configs/hub.yaml: repos.{key} is not configured (set `org` to the "
                          "organisation that hosts the release; the shipped value is a placeholder)")
    return str(repo)


def wanted_episodes(selection: dict[str, Any] | None) -> set[str] | None:
    """Episode dir names named by a selection json: ``episodes`` entries
    (strings or dicts with ``name``/``episode``/``seed``) or ``seeds``
    (-> ``ep_<seed>``). ``None`` means every episode."""
    if selection is None:
        return None
    if "episodes" in selection:
        out: set[str] = set()
        for e in selection["episodes"]:
            if isinstance(e, dict):
                name = e.get("name") or e.get("episode") or (f"ep_{e['seed']}" if "seed" in e else None)
                if name is None:
                    raise ValueError(f"cannot name episode from {e}")
                out.add(str(name))
            else:
                out.add(str(e))
        return out
    if "seeds" in selection:
        return {f"ep_{int(s)}" for s in selection["seeds"]}
    raise ValueError("selection json needs an 'episodes' or 'seeds' list")


def _hf_download(repo_id: str, filename: str, local_dir: Path, repo_type: str = "dataset") -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id, filename, repo_type=repo_type, local_dir=str(local_dir)))


def extract_members(tar_path: Path, out_dir: Path, wanted: set[str] | None) -> int:
    """Extract the top-level episode dirs in ``wanted`` (all when None); returns the count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with tarfile.open(tar_path) as tf:
        members = []
        for m in tf.getmembers():
            top = m.name[2:] if m.name.startswith("./") else m.name
            top = top.split("/", 1)[0]
            if wanted is None or top in wanted:
                members.append(m)
                seen.add(top)
        if hasattr(tarfile, "data_filter"):
            tf.extractall(out_dir, members=members, filter="data")
        else:
            tf.extractall(out_dir, members=members)
    return len(seen)


MANIFEST = "manifest.json"


def repo_manifests(repo_id: str) -> list[str]:
    """Every manifest in a raw dataset repo: the top-level ``manifest.json``
    plus any ``<dir>/manifest.json`` of later additions, sorted."""
    from huggingface_hub import list_repo_files

    names = list_repo_files(repo_id, repo_type="dataset")
    found = sorted(n for n in names if n == MANIFEST or n.endswith("/" + MANIFEST))
    if not found:
        raise RuntimeError(f"{repo_id}: no {MANIFEST} in the repository")
    return found


def merge_manifests(manifests: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Merge ``{manifest path: parsed manifest}`` into one ``{shard path:
    [episode names]}`` table; shard names are relative to their manifest's
    directory (``testfix/manifest.json`` + ``shard_0000.tar`` ->
    ``testfix/shard_0000.tar``)."""
    shards: dict[str, list[str]] = {}
    for path, manifest in sorted(manifests.items()):
        prefix = path[: -len(MANIFEST)]
        for shard, eps in (manifest.get("shards") or {}).items():
            shards[prefix + shard] = [str(e) for e in eps]
    return shards


def load_shards(repo_id: str, stage: Path) -> dict[str, list[str]]:
    """Download every manifest of ``repo_id`` into ``stage`` and merge them."""
    return merge_manifests({m: json.load(open(_hf_download(repo_id, m, stage))) for m in repo_manifests(repo_id)})


def download_raw(
    repo_id: str,
    out_dir: str | Path,
    selection: str | Path | dict[str, Any] | None = None,
    *,
    keep_tars: bool = False,
    stage_dir: str | Path | None = None,
) -> int:
    """Download and extract the raw episodes of ``repo_id`` named by ``selection``.

    Resume-safe: episodes whose ``meta.npz`` already exists are not fetched
    again and shards holding nothing new are skipped. Returns the number of
    wanted episodes present after the call.
    """
    out_dir = Path(out_dir)
    stage = Path(stage_dir) if stage_dir else out_dir / "_shards"
    if isinstance(selection, (str, Path)):
        selection = json.load(open(selection))
    wanted = wanted_episodes(selection)
    shards = load_shards(repo_id, stage)
    in_repo = {e for eps in shards.values() for e in eps}
    total_wanted = wanted if wanted is not None else in_repo
    missing = total_wanted - in_repo
    if missing:
        log.warning("[HUB] %d wanted episodes are not in %s (e.g. %s)", len(missing), repo_id, sorted(missing)[:3])
    have = {d.name for d in out_dir.glob("ep_*") if (d / "meta.npz").exists()}
    for shard, eps in sorted(shards.items()):
        need = [e for e in eps if e in total_wanted and e not in have]
        if not need:
            continue
        p = _hf_download(repo_id, shard, stage)
        n = extract_members(p, out_dir, set(need))
        if not keep_tars:
            p.unlink(missing_ok=True)
        have |= set(need)
        log.info("[HUB] %s: extracted %d episodes", shard, n)
    n_have = len([d for d in out_dir.glob("ep_*") if d.name in total_wanted and (d / "meta.npz").exists()])
    log.info("[HUB] %s: %d/%d wanted episodes present in %s", repo_id, n_have, len(total_wanted), out_dir)
    return n_have


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_files(root: Path, rel_paths: Iterable[str], entry: dict[str, Any]) -> bool:
    """Check size / sha256 of files under ``root`` against a checkpoint entry
    (``sizes`` / ``sha256`` maps keyed by file name). Returns False on a
    mismatch or a missing file; an absent table value only warns."""
    sizes = entry.get("sizes") or {}
    shas = entry.get("sha256") or {}
    ok = True
    for rel in rel_paths:
        p = root / rel
        if not p.exists():
            log.error("[HUB] missing %s", p)
            ok = False
            continue
        if sizes.get(rel):
            good = p.stat().st_size == int(sizes[rel])
            ok &= good
            log.info("[HUB] size %s %d %s", rel, p.stat().st_size, "OK" if good else "MISMATCH")
        digest = sha256_of(p)
        if shas.get(rel):
            good = digest == str(shas[rel]).lower()
            ok &= good
            log.info("[HUB] sha256 %s %s %s", rel, digest[:16], "OK" if good else "MISMATCH")
        else:
            log.warning("[HUB] no sha256 for %s in configs/hub.yaml: unverified (sha256 %s)", rel, digest[:16])
    return ok


def download_checkpoint(name: str, out_root: str | Path, cfg: dict[str, Any] | None = None) -> Path:
    """Download checkpoint directory ``name`` (a row or ``pi0_base``) from the
    checkpoints repo into ``<out_root>/<name>`` and verify it."""
    cfg = cfg or load_hub_config()
    entry = (cfg.get("checkpoints") or {}).get(name)
    if entry is None:
        raise KeyError(f"'{name}' is not listed under checkpoints: in configs/hub.yaml")
    entry = {"path": entry} if isinstance(entry, str) else dict(entry)
    path = entry.get("path", name)
    repo = entry.get("repo") or _require_repo(cfg, "checkpoints")
    from huggingface_hub import snapshot_download

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo, repo_type="model", local_dir=str(out_root), allow_patterns=[f"{path}/*"])
    src, dst = out_root / path, out_root / name
    if src != dst and src.exists() and not dst.exists():
        src.rename(dst)
    files = list(entry.get("files") or []) or sorted(str(p.relative_to(dst)) for p in dst.rglob("*") if p.is_file())
    if not files:
        raise RuntimeError(f"no files downloaded for {repo}/{path}")
    if not verify_files(dst, files, entry):
        raise RuntimeError(f"checkpoint {name} failed verification")
    log.info("[HUB] checkpoint %s ready at %s", name, dst)
    return dst


def download_base_weights(key: str, out_root: str | Path | None = None, cfg: dict[str, Any] | None = None) -> Path:
    """Fetch base weights by key: a ``checkpoints:`` entry (``pi0_base``) is
    downloaded and verified into ``<out_root>/<key>``; a ``base_weights:``
    entry is a third-party hub id fetched into the hub cache (or
    ``<out_root>/<key>`` when ``out_root`` is given)."""
    cfg = cfg or load_hub_config()
    if key in (cfg.get("checkpoints") or {}):
        if out_root is None:
            raise ValueError("out_root is required for a checkpoints-repo entry")
        return download_checkpoint(key, out_root, cfg)
    entry = (cfg.get("base_weights") or {}).get(key)
    if entry is None:
        raise KeyError(f"'{key}' is neither under checkpoints: nor base_weights: in configs/hub.yaml")
    repo = entry if isinstance(entry, str) else entry["repo"]
    if str(repo).startswith("torchvision:"):
        raise ValueError(f"{key} ({repo}) is fetched by torchvision on first use; nothing to download")
    patterns = None if isinstance(entry, str) else entry.get("allow_patterns")
    from huggingface_hub import snapshot_download

    kw: dict[str, Any] = {"allow_patterns": patterns}
    if out_root is not None:
        Path(out_root, key).mkdir(parents=True, exist_ok=True)
        kw["local_dir"] = str(Path(out_root, key))
    path = Path(snapshot_download(str(repo), **kw))
    log.info("[HUB] base weights %s (%s) at %s", key, repo, path)
    return path


FALLBACK_SETS = ("strict", "recovery")


def data_sets() -> dict[str, dict[str, Any]]:
    """``sets:`` of ``configs/data.yaml`` (cache set name -> entry); the main
    pair alone when the config cannot be read."""
    try:
        from simretract import config as _cfg

        sets = dict(_cfg.data_config().get("sets") or {})
    except Exception:  # noqa: BLE001 - unreadable config: main pair only
        sets = {}
    return sets or {k: {} for k in FALLBACK_SETS}


def _default_selection(cfg_key: str) -> Path:
    from simretract import config as _cfg

    sel = (data_sets().get(cfg_key) or {}).get("selection")
    if not sel:
        raise KeyError(f"configs/data.yaml: sets.{cfg_key}.selection is missing")
    return _cfg.selection_path(sel)


def _repo_key(cfg_key: str) -> str:
    """The ``repos:`` key of a cache set (``sets.<name>.hub_repo``, default ``<name>_raw``)."""
    return str((data_sets().get(cfg_key) or {}).get("hub_repo") or f"{cfg_key}_raw")


def _checkpoints_root(override: str | None) -> Path:
    if override:
        return Path(override)
    from simretract import config as _cfg

    return _cfg.paths()["checkpoints"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download raw data, checkpoints or base weights from the hub (configs/hub.yaml).")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data", help="raw episode shards of a selection")
    d.add_argument("--set", choices=tuple(data_sets()), required=True,
                   help="cache set of configs/data.yaml: picks the raw dataset repo (sets.<name>.hub_repo) and the selection")
    d.add_argument("--selection", default=None,
                   help="selection json (seeds or episodes); default: the shipped selection of --set from configs/data.yaml")
    d.add_argument("--out", required=True, help="raw output directory (ep_* dirs)")
    d.add_argument("--keep-tars", action="store_true", help="keep the downloaded shard tars under <out>/_shards")
    c = sub.add_parser("checkpoints", help="one or more checkpoint rows")
    c.add_argument("names", nargs="*", help="row names from configs/hub.yaml (default: every row except pi0_base)")
    c.add_argument("--out", default=None, help="checkpoints root (default: paths.yaml checkpoints)")
    b = sub.add_parser("base-weights", help="base weights: pi0_base (checkpoints repo) or a third-party key")
    b.add_argument("keys", nargs="*", default=["pi0_base"], help="keys from configs/hub.yaml (default pi0_base)")
    b.add_argument("--out", default=None, help="checkpoints root for pi0_base / local dir root for third-party keys")
    b.add_argument("--hub-cache", action="store_true", help="leave third-party weights in the hub cache (no local copy)")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    a = build_parser().parse_args(argv)
    try:
        cfg = load_hub_config()
        if a.cmd == "data":
            repo = _require_repo(cfg, _repo_key(a.set))
            download_raw(repo, a.out, a.selection or _default_selection(a.set), keep_tars=a.keep_tars)
        elif a.cmd == "checkpoints":
            names = a.names or [n for n in (cfg.get("checkpoints") or {}) if n != "pi0_base"]
            for n in names:
                download_checkpoint(n, _checkpoints_root(a.out), cfg)
        elif a.cmd == "base-weights":
            for k in a.keys:
                root = None if (a.hub_cache and k not in (cfg.get("checkpoints") or {})) else _checkpoints_root(a.out)
                download_base_weights(k, root, cfg)
    except ConfigError as exc:
        log.error("error: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
