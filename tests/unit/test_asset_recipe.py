"""The tool asset recipe (``configs/assets/<tool>.yaml``) and the mesh
fingerprint verifier (``simretract.sim.asset_recipe``).

The recipe must load, name the drop-in source under ``assets/source/``,
carry the fingerprints of the released mesh and agree with the tracked
``meta.json``. The verifier is exercised on synthetic meshes: fingerprint ->
compare OK; a moved vertex, a reordered face or a missing face is reported;
a sub-margin perturbation passes the grid hash while the exact hash
changes; the source identification matches by sha256, then by name.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from simretract import config
from simretract.sim import asset_recipe as ar

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
TOOL = "retractor_rake"


# --------------------------------------------------------------------------- the tracked recipe


@pytest.fixture
def recipe() -> dict:
    rec = ar.load_recipe(TOOL)
    assert rec is not None
    return rec


@pytest.fixture
def meta() -> dict:
    return json.loads((REPO / "assets/tools" / TOOL / "meta.json").read_text())


def test_recipe_names_the_drop_in_source(recipe: dict) -> None:
    src = recipe["source"]
    assert src["pack"] == "vretractor" and src["file"] == "vretractor.fbx"
    assert src["sha256"] == "df9689bc33e997417d2898406cdb7214cbabde58744513b2ddd0013950061cbb"
    assert src["size_bytes"] == 210860
    assert src["select"] == "ovrsi_retractor_body"
    assert src["scale"] == pytest.approx(0.01474075)
    assert src["listing_url"].startswith("https://www.fab.com/listings/")
    assert src["license_url"] == "https://www.fab.com/eula"
    assert "Allows usage with AI: Yes" in src["ai_usage"]
    assert ar.default_source_path(recipe) == config.paths()["assets"] / "source" / "vretractor" / "vretractor.fbx"
    R = ar.parse_pre_rotate(src["pre_rotate"])
    assert R is not None and np.array_equal(R, np.diag([-1.0, 1.0, -1.0]))


def test_recipe_names_one_source_only(recipe: dict) -> None:
    assert "alternative_sources" not in recipe
    assert ar.source_entry(recipe)["label"] == "vretractor"


def test_recipe_fingerprint_is_the_released_mesh(recipe: dict) -> None:
    fp = recipe["fingerprint"]
    assert fp["n_vertices"] == 3152 and fp["n_faces"] == 6308
    assert fp["vertex_sha256"] == "9f71ad5b457d967f3a9d623159dcbe37ed828eed0e292e020ed92266d7c61c34"
    assert fp["face_sha256"] == "7e88d550b79ee89005e815d9f42e7d4c26d4076b31461a385f4ecd7bb218d9b7"
    assert fp["extents_m"] == pytest.approx([0.03059319, 0.00967014, 0.21631345], abs=1e-8)
    assert fp["volume_m3"] == pytest.approx(6.6109e-06, rel=1e-4)
    assert fp["watertight"] is True
    bits = ar.decode_phase(fp["vertex_grid_phase"], 3 * fp["n_vertices"])
    assert bits.shape == (3 * 3152,) and set(np.unique(bits)) <= {0, 1}


def test_recipe_agrees_with_meta_json(recipe: dict, meta: dict) -> None:
    exp = recipe["expected_meta"]
    for key in ("extents_m", "longest_m", "watertight", "volume_m3", "n_collision_parts",
                "collision_hull_volume_m3", "suggested_tip", "suggested_bend"):
        assert exp[key] == meta[key], key
    assert recipe["conversion"] == {"align_z": meta["align_z"], "coacd_threshold": meta["coacd_threshold"]}
    # meta.json describes the build from the recipe source
    src = recipe["source"]
    assert (meta["select"], meta["scale"]) == (src["select"], src["scale"])
    assert np.array_equal(ar.parse_pre_rotate(meta["pre_rotate"]), ar.parse_pre_rotate(src["pre_rotate"]))
    assert meta["center_removed_pre_m"] == src["expected_center_removed_pre_m"]
    assert "source" not in meta and "/" not in json.dumps(meta)


def test_settings_for_source(recipe: dict) -> None:
    s = ar.settings_for(ar.source_entry(recipe), recipe)
    assert s["select"] == "ovrsi_retractor_body" and s["scale"] == pytest.approx(0.01474075)
    assert np.array_equal(s["pre_rotate"], np.diag([-1.0, 1.0, -1.0]))
    assert s["align_z"] is True and s["coacd_threshold"] == 0.015
    assert ar.settings_for(None, recipe) == {"align_z": True, "coacd_threshold": 0.015}


def test_recipe_carries_no_paths_or_dates() -> None:
    text = ar.recipe_path(TOOL).read_text(encoding="utf-8")
    assert "/ho" + "me/" not in text and "20" + "26-" not in text
    assert all(ord(ch) < 128 for ch in text)


def test_missing_recipe_is_none_and_malformed_is_actionable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert ar.load_recipe("no_such_tool") is None
    cfg = tmp_path / "configs" / "assets"
    cfg.mkdir(parents=True)
    (tmp_path / "configs" / "paths.yaml").write_text(config.config_path("paths").read_text(encoding="utf-8"))
    (cfg / "bad.yaml").write_text("tool: bad\nsource: {pack: p}\n", encoding="utf-8")
    monkeypatch.setenv("SIMRETRACT_REPO", str(tmp_path))
    with pytest.raises(config.ConfigError, match="source"):
        ar.load_recipe("bad")
    (cfg / "bad.yaml").write_text("tool: other\nsource: {pack: p}\n", encoding="utf-8")
    with pytest.raises(config.ConfigError, match="tool"):
        ar.load_recipe("bad")


# --------------------------------------------------------------------------- source identification


def _mini_recipe(tmp_path: Path, payload: bytes) -> tuple[dict, Path]:
    f = tmp_path / "pack" / "model.fbx"
    f.parent.mkdir()
    f.write_bytes(payload)
    rec = {"tool": "t", "source": {"pack": "pack", "file": "model.fbx", "sha256": hashlib.sha256(payload).hexdigest(),
                                   "size_bytes": len(payload), "select": "n", "scale": 0.5,
                                   "pre_rotate": [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]}}
    return rec, f


def test_identify_source_by_sha256_then_name(tmp_path: Path) -> None:
    rec, f = _mini_recipe(tmp_path, b"model bytes")
    entry, how = ar.identify_source(f, rec)
    assert how == "sha256" and entry["label"] == "pack"
    f.write_bytes(b"model bytes, revised")
    entry, how = ar.identify_source(f, rec)
    assert how == "name" and entry["label"] == "pack"
    assert entry["actual_sha256"] == hashlib.sha256(b"model bytes, revised").hexdigest()
    other = tmp_path / "pack" / "unrelated.obj"
    other.write_bytes(b"v 0 0 0\n")
    assert ar.identify_source(other, rec) == (None, "unknown")


def test_resolve_source_order(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    explicit = tmp_path / "x.obj"
    assert ar.resolve_source(TOOL, explicit)[0] == explicit
    override = tmp_path / "override.fbx"
    override.write_bytes(b"x")
    clean_env.setenv("SIMRETRACT_TOOL_SOURCE", str(override))
    path, origin = ar.resolve_source(TOOL)
    assert path == override and "tool_source" in origin
    clean_env.delenv("SIMRETRACT_TOOL_SOURCE")
    clean_env.setattr(config, "_envs_file", lambda: config.config_path("envs"))  # template: no override
    path, origin = ar.resolve_source(TOOL)
    drop_in = config.paths()["assets"] / "source" / "vretractor" / "vretractor.fbx"
    if drop_in.is_file():
        assert path == drop_in and "recipe default" in origin
    else:
        assert path is None and "assets/source/vretractor/" in origin
    path, origin = ar.resolve_source("no_such_tool")
    assert path is None and "no recipe" in origin


def test_parse_and_format_pre_rotate() -> None:
    assert ar.parse_pre_rotate(None) is None
    assert ar.parse_pre_rotate("identity") is None
    R = ar.parse_pre_rotate("-1,0,0, 0,1,0, 0,0,-1")
    assert np.array_equal(R, np.diag([-1.0, 1.0, -1.0]))
    assert ar.parse_pre_rotate(ar.format_pre_rotate(R)).tolist() == R.tolist()
    assert ar.format_pre_rotate(None) == "identity"
    assert np.array_equal(ar.parse_pre_rotate([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]), np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))
    with pytest.raises(ValueError, match="9 numbers"):
        ar.parse_pre_rotate("1,0,0")
    with pytest.raises(ValueError, match="proper rotation"):
        ar.parse_pre_rotate("1,0,0, 0,1,0, 0,0,-1")   # a reflection
    with pytest.raises(ValueError, match="proper rotation"):
        ar.parse_pre_rotate("2,0,0, 0,1,0, 0,0,1")


# --------------------------------------------------------------------------- fingerprints on synthetic meshes


def _box(size=(0.02, 0.01, 0.2), offset=(0.0123456789, -0.0042, 0.00007)) -> tuple[np.ndarray, np.ndarray]:
    """A closed, outward-wound box with irrational-looking coordinates."""
    sx, sy, sz = (s / 2 for s in size)
    V = np.array([[-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
                  [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz]], dtype=np.float64)
    V += np.asarray(offset)
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.int64)
    return V, F


def test_fingerprint_roundtrip_and_values() -> None:
    V, F = _box()
    fp = ar.mesh_fingerprint(V, F)
    assert fp["n_vertices"] == 8 and fp["n_faces"] == 12 and fp["watertight"] is True
    assert fp["volume_m3"] == pytest.approx(0.02 * 0.01 * 0.2, rel=1e-12)
    assert fp["extents_m"] == pytest.approx([0.02, 0.01, 0.2], abs=1e-9)
    assert len(fp["vertex_sha256"]) == 64 and len(fp["face_sha256"]) == 64
    assert ar.compare_fingerprint(V, F, fp) == ([], ["vertex_sha256: exact (coordinates identical to 1e-6 m)"])
    # the exact vertex hash is what the recipe documents: float64 rounded to 1e-6 m, C-order bytes
    assert fp["vertex_sha256"] == hashlib.sha256(np.round(V, 6).tobytes()).hexdigest()
    assert fp["face_sha256"] == hashlib.sha256(F.astype(np.int64).tobytes()).hexdigest()


def test_fingerprint_yaml_block_parses(tmp_path: Path) -> None:
    import yaml

    V, F = _box()
    block = yaml.safe_load(ar.fingerprint_yaml(ar.mesh_fingerprint(V, F)))["fingerprint"]
    assert ar.compare_fingerprint(V, F, block)[0] == []


def test_obj_loader_matches_arrays(tmp_path: Path) -> None:
    V, F = _box()
    obj = tmp_path / "m.obj"
    lines = ["# header"] + [f"v {x:.8f} {y:.8f} {z:.8f}" for x, y, z in V] + [f"f {a + 1}//1 {b + 1}//1 {c + 1}//1" for a, b, c in F]
    obj.write_text("\n".join(lines) + "\n")
    V2, F2 = ar.load_obj(obj)
    assert np.array_equal(F2, F) and np.allclose(V2, V, atol=5e-9)
    quad = tmp_path / "q.obj"
    quad.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n")
    with pytest.raises(ValueError, match="triangulated"):
        ar.load_obj(quad)


def test_sub_margin_noise_keeps_the_grid_hash_only() -> None:
    V, F = _box(offset=(0.0123455, -0.0042, 0.00007))   # x coordinates sit on a 1 um rounding boundary
    fp = ar.mesh_fingerprint(V, F)
    rng = np.random.default_rng(0)
    # a few 1e-8 m of noise: well inside the 0.25 um grid margin, the 0.1 um
    # extents tolerance and the 1e-6 volume tolerance
    noisy = V + rng.uniform(-3e-8, 3e-8, V.shape)
    assert ar.vertex_sha256(noisy) != fp["vertex_sha256"]   # some coordinate crossed the boundary
    problems, notes = ar.compare_fingerprint(noisy, F, fp)
    assert problems == []
    assert any("vertex_grid_sha256 OK" in n for n in notes)
    # a coordinate moved by more than the margin's outer bound fails the grid hash
    far = V.copy()
    far[3, 1] += 1.0e-6
    problems, _ = ar.compare_fingerprint(far, F, fp)
    assert any("vertex_grid_sha256" in p for p in problems)


def test_phase_bits_guarantee_the_margin() -> None:
    rng = np.random.default_rng(1)
    V = rng.uniform(-0.1, 0.1, (500, 3))
    phase = ar.grid_phase(V)
    codes = ar.grid_codes(V, phase)
    for delta in (ar.GRID_MARGIN_M * 0.99, -ar.GRID_MARGIN_M * 0.99):
        assert np.array_equal(ar.grid_codes(V + delta, phase), codes)
    q = V.reshape(-1) / ar.GRID_M
    frac = q - np.floor(q)
    dist = np.where(phase == 1, np.minimum(frac, 1 - frac), np.abs(frac - 0.5))
    assert dist.min() >= 0.25 - 1e-9
    assert ar.decode_phase(ar.encode_phase(phase), phase.size).tolist() == phase.tolist()
    with pytest.raises(ValueError):
        ar.decode_phase(base64.b64encode(b"\x00").decode(), 500 * 3)


def test_mismatches_are_reported() -> None:
    V, F = _box()
    fp = ar.mesh_fingerprint(V, F)
    moved = V.copy()
    moved[0] -= 1e-3      # the (-x, -y, -z) corner moved outward
    problems, _ = ar.compare_fingerprint(moved, F, fp)
    assert any("vertex_grid_sha256" in p for p in problems) and any("extents_m" in p for p in problems)
    assert any("volume_m3" in p for p in problems)
    swapped = F[[1, 0] + list(range(2, 12))]
    problems, _ = ar.compare_fingerprint(V, swapped, fp)
    assert problems == ["face_sha256 differs (different triangles or vertex order)"]
    open_mesh = F[:-1]
    problems, _ = ar.compare_fingerprint(V, open_mesh, fp)
    assert any("faces" in p for p in problems)      # count check stops early
    fp_open = ar.mesh_fingerprint(V, open_mesh)
    assert fp_open["watertight"] is False and fp_open["volume_m3"] is None
    problems, _ = ar.compare_fingerprint(V, F[:-1], dict(fp, n_faces=11))
    assert any("watertight" in p for p in problems)


def test_verify_obj_uses_the_recipe_block(tmp_path: Path) -> None:
    V, F = _box()
    obj = tmp_path / "visual.obj"
    obj.write_text("\n".join([f"v {float(x)!r} {float(y)!r} {float(z)!r}" for x, y, z in V]
                             + [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in F]) + "\n")
    rec = {"tool": "t", "fingerprint": ar.mesh_fingerprint(V, F)}
    problems, notes = ar.verify_obj(obj, rec)
    assert problems == [] and notes
    with pytest.raises(config.ConfigError, match="fingerprint"):
        ar.verify_obj(obj, {"tool": "t"})
