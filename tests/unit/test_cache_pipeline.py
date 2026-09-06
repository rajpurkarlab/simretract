"""End-to-end unit test of the data pipeline on a tiny synthetic raw set:
2 strict episodes + 1 recovery draw (with a perturbation window) written in
the documented raw layout -> strict and recovery caches with the production
flags -> plane sidecar -> alignment gate. Also checks the selection rules
(held-out seed exclusion, missing perturb window, empty mask) and that
``perturb_trim`` drops every frame up to and including the excursion end.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from simretract.data import cache as C
from simretract.data import synthetic
from simretract.data.plane_sidecar import precompute_plane

H, W = C.FRAME_H, C.FRAME_W
T_STRICT = synthetic.T_STRICT
T_RECOVERY = synthetic.T_RECOVERY
PERTURB_END = synthetic.PERTURB_END
write_episode = synthetic.write_episode


@pytest.fixture(scope="module")
def raw(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("raw")
    rng = np.random.default_rng(0)
    strict = root / "strict"
    recovery = root / "recovery"
    st = {}
    st["ep_200011"] = write_episode(strict, "ep_200011", 200011, T_STRICT, rng)
    st["ep_200022"] = write_episode(strict, "ep_200022", 200022, T_STRICT, rng)
    write_episode(strict, "ep_200033", 200033, T_STRICT, rng)                    # held-out seed
    write_episode(strict, "ep_100044", 100044, T_STRICT, rng)                    # below seed_min
    write_episode(strict, "ep_200055", 200055, T_STRICT, rng, empty_mask=True)   # broken render
    st["ep_200011_p0"] = write_episode(
        recovery, "ep_200011_p0", 200011, T_RECOVERY, rng, close_at=70,
        info={"perturb_win": {"start": 0, "end": PERTURB_END}, "perturb": {"frac": 0.3}})
    write_episode(recovery, "ep_200022_p0", 200022, T_RECOVERY, rng, close_at=70)  # no perturb window
    pool = root / "test_pool.json"
    json.dump({"name": "pool", "seeds": [200033]}, open(pool, "w"))
    return {"strict": strict, "recovery": recovery, "pool": pool, "states": st, "cache": root / "cache"}


def cfg_for(raw: dict, set_key: str) -> C.CacheConfig:
    args = C.build_parser().parse_args(
        ["--set", set_key, "--raw", str(raw[set_key]), "--out", str(raw["cache"]),
         "--exclude-seeds", str(raw["pool"]), "--workers", "2"])
    cfg = C.config_from_args(args)
    assert cfg.flags()[set_key and "pretrim"] is True
    return cfg


def test_production_presets():
    strict = dict(pretrim=True, keep_only=True, seed_min=200000, skip_gmk=True, perturb_trim=False)
    recovery = dict(pretrim=True, keep_only=True, seed_min=200000, skip_gmk=True, perturb_trim=True)
    assert C.SET_PRESETS["strict"] == strict and C.SET_PRESETS["recovery"] == recovery
    # the no-smoothing / no-standoff pairs build with the same flags as the main pair
    for name in ("strict_nosmooth", "strict_nostandoff"):
        assert C.SET_PRESETS[name] == strict, name
    for name in ("recovery_nosmooth", "recovery_nostandoff"):
        assert C.SET_PRESETS[name] == recovery, name
    assert set(C.SET_PRESETS) == {"strict", "recovery", "strict_nosmooth", "recovery_nosmooth",
                                  "strict_nostandoff", "recovery_nostandoff"}
    b = C.build_parser().parse_args(["--set", "recovery_nosmooth", "--raw", "r", "--out", "o", "--no-exclude-seeds"])
    cfg_b = C.config_from_args(b)
    assert cfg_b.name == "recovery_nosmooth" and cfg_b.perturb_trim is True
    # no fingerprint is recorded for the variant sets: --verify has nothing to compare and says so
    assert C.reference_fingerprint("strict_nosmooth") is None
    assert C.reference_fingerprint("strict")["frames"] == 69402
    a = C.build_parser().parse_args(["--set", "recovery", "--raw", "r", "--out", "o", "--no-exclude-seeds", "--no-perturb-trim"])
    cfg = C.config_from_args(a)
    assert cfg.perturb_trim is False and cfg.keep_only is True and cfg.name == "recovery"


def test_strict_cache(raw):
    cfg = cfg_for(raw, "strict")
    summary = C.build_cache(cfg)
    assert summary["episodes"] == 2 and summary["verify_bad"] == 0
    meta = json.load(open(raw["cache"] / "meta.json"))["strict"]
    assert meta["prekept"] is True and meta["N"] == summary["N"]
    assert meta["flags"]["perturb_trim"] is False and meta["flags"]["seed_min"] == 200000
    assert meta["n_excluded_seeds"] == 1
    assert meta["source_episodes"] == ["ep_200011", "ep_200022"]
    verdicts = json.load(open(raw["cache"] / "strict_verdicts.json"))
    assert [v["seed"] for v in verdicts.values()] == [200011, 200022]
    assert all(v["strict_pass"] is False and v["battery_skipped"] for v in verdicts.values())  # skipped battery -> false, as the reference build wrote it
    # frames = kept frames of the pretrimmed episodes; state rows equal the raw states
    st = raw["states"]["ep_200011"]
    cut = C.close_cutoff_frames(st)
    assert cut == int(T_STRICT * 0.6) + C.CLOSE_MARGIN - 0 or cut == min(T_STRICT, int(T_STRICT * 0.6) + C.CLOSE_MARGIN)
    kept = C.kept_idx(st[:cut])
    a0, a1 = meta["episodes"][0]
    assert a1 - a0 == len(kept)
    state = np.load(raw["cache"] / "strict_state.npy")
    assert np.array_equal(state[a0:a1], st[kept])
    assert np.all(np.diff(kept) > 0) and kept[-1] == cut - 1
    # the stall was collapsed: fewer kept frames than raw frames
    assert len(kept) < cut
    # gripper mask left sparse
    gmk = np.load(raw["cache"] / "strict_gmk.npy", mmap_mode="r")
    assert gmk.shape == (meta["N"], H, W) and not gmk[a0].any()
    rgb = np.load(raw["cache"] / "strict_rgb.npy", mmap_mode="r")
    msk = np.load(raw["cache"] / "strict_msk.npy", mmap_mode="r")
    assert rgb.shape == (meta["N"], H, W, 3) and msk[a0].max() == 255


def test_recovery_cache_perturb_trim(raw):
    cfg = cfg_for(raw, "recovery")
    summary = C.build_cache(cfg)
    assert summary["episodes"] == 1 and summary["verify_bad"] == 0
    meta = json.load(open(raw["cache"] / "meta.json"))
    assert set(meta) == {"strict", "recovery"}  # shared dir merged
    rec = meta["recovery"]
    assert rec["flags"]["perturb_trim"] is True and rec["source_episodes"] == ["ep_200011_p0"]
    st = raw["states"]["ep_200011_p0"]
    s0 = PERTURB_END + 1
    cut = C.close_cutoff_frames(st)
    kept = s0 + C.kept_idx(st[s0:cut])
    a0, a1 = rec["episodes"][0]
    assert a1 - a0 == len(kept) and kept[0] == s0
    state = np.load(raw["cache"] / "recovery_state.npy")
    assert np.array_equal(state[a0:a1], st[kept])
    # no cached frame comes from the perturbation window
    assert not any(np.array_equal(state[k], st[i]) for k in range(a0, a1) for i in range(s0))
    dep = np.load(raw["cache"] / "recovery_dep.npy", mmap_mode="r")
    first = np.asarray(Image.open(raw["recovery"] / "ep_200011_p0" / f"depth_{s0:04d}.png"))
    assert np.array_equal(dep[a0], first)


def test_plane_sidecar_and_fingerprint(raw):
    for name in ("strict", "recovery"):
        n_fit, n = precompute_plane(raw["cache"], name)
        plane = np.load(raw["cache"] / f"{name}_plane.npy")
        conf = np.load(raw["cache"] / f"{name}_plane_conf.npy")
        assert plane.shape == (n, 3) and conf.shape == (n,) and n_fit == n
        assert np.allclose(np.linalg.norm(plane, axis=1), 1, atol=1e-5) and (plane[:, 2] < 0).all()
    fp = C.fingerprint(raw["cache"], "strict")
    assert set(fp["md5"]) == {"action", "plane", "gpose"} and all(len(v) == 8 for v in fp["md5"].values())
    assert C.verify_fingerprint(raw["cache"], "strict", fp)
    assert not C.verify_fingerprint(raw["cache"], "strict", {"frames": fp["frames"] + 1, "md5": fp["md5"]})
    assert C.reference_fingerprint("recovery")["frames"] == 106868


def test_verify_align(raw):
    pytest.importorskip("torch")
    try:
        from simretract.train import train  # noqa: F401
    except ImportError:
        pytest.skip("trainer module not importable in this environment")
    from simretract.data.verify_align import verify_alignment

    assert verify_alignment(raw["cache"], ["strict"], n_episodes=2, state_mode="full", float_depth=False)
    assert verify_alignment(raw["cache"], ["strict"], n_episodes=2, state_mode="qpos_plane", float_depth=True, nogrip=True)
    assert verify_alignment(raw["cache"], ["recovery"], n_episodes=1, state_mode="qpos_plane", float_depth=True, nogrip=True)


def test_cli_and_log_lines(raw, tmp_path, caplog):
    import logging

    out = tmp_path / "cache2"
    with caplog.at_level(logging.INFO):
        rc = C.main(["--set", "strict", "--raw", str(raw["strict"]), "--out", str(out),
                     "--exclude-seeds", str(raw["pool"]), "--workers", "1"])
    assert rc == 0
    text = caplog.text
    assert "VERIFY PASS" in text and "CACHE FROM RAW DONE" in text and "EXCLUDED" in text
    assert (out / "strict_plane.npy").exists()


def test_presets_match_data_yaml():
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(open(Path(__file__).resolve().parents[2] / "configs" / "data.yaml"))
    assert set(C.SET_PRESETS) == set(data["sets"])
    for key, preset in C.SET_PRESETS.items():
        assert data["sets"][key]["cache_flags"] == preset
        ref = C.reference_fingerprint(key)
        want = data["sets"][key]["fingerprint"]
        if want is None:  # no reference build recorded: --verify skips the set
            assert ref is None, key
            continue
        assert ref["frames"] == want["frames"]
        assert ref["md5"] == want["md5_8"]


def test_hub_shard_contract(tmp_path, raw):
    """Only the wanted members of a shard are extracted; selections resolve to episode names."""
    import tarfile

    from simretract.data import hub

    assert hub.wanted_episodes({"seeds": [200011, 200022]}) == {"ep_200011", "ep_200022"}
    assert hub.wanted_episodes({"episodes": ["ep_1_p0", {"name": "ep_2_p1"}, {"seed": 3}]}) == {"ep_1_p0", "ep_2_p1", "ep_3"}
    assert hub.wanted_episodes(None) is None
    merged = hub.merge_manifests({
        "manifest.json": {"shards": {"shard_0000.tar": ["ep_1", "ep_2"], "shard_0001.tar": ["ep_3"]}},
        "later/manifest.json": {"shards": {"shard_0000.tar": ["ep_4"]}},
    })
    assert merged == {"shard_0000.tar": ["ep_1", "ep_2"], "shard_0001.tar": ["ep_3"], "later/shard_0000.tar": ["ep_4"]}
    assert set(hub.data_sets()) >= {"strict", "recovery", "strict_nosmooth", "recovery_nosmooth",
                                    "strict_nostandoff", "recovery_nostandoff"}
    shard = tmp_path / "shard_0000.tar"
    with tarfile.open(shard, "w") as tf:
        for name in ("ep_200011", "ep_200022"):
            tf.add(raw["strict"] / name, arcname=name)
    out = tmp_path / "extract"
    assert hub.extract_members(shard, out, {"ep_200022"}) == 1
    assert (out / "ep_200022" / "meta.npz").exists() and not (out / "ep_200011").exists()
    assert hub.extract_members(shard, out, None) == 2
    entry = {"files": ["meta.npz"], "sha256": {"meta.npz": hub.sha256_of(out / "ep_200011" / "meta.npz")},
             "sizes": {"meta.npz": (out / "ep_200011" / "meta.npz").stat().st_size}}
    assert hub.verify_files(out / "ep_200011", ["meta.npz"], entry)
    assert not hub.verify_files(out / "ep_200011", ["meta.npz"], {"sha256": {"meta.npz": "00" * 32}})
    assert hub.verify_files(out / "ep_200011", ["meta.npz"], {})  # unfilled table: warn only
    assert not hub.verify_files(out / "ep_200011", ["missing.bin"], {})
