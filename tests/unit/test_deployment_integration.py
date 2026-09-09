"""Cross-module deployment behavior."""
from types import SimpleNamespace

import numpy as np
import pytest

from simretract.cli import run_real
from simretract.real import client, load_real_config, preset_client_args, resolve_preset
from simretract.serve import metric
from simretract.serve.policy_server import NormSpec, PolicyEngine, Preprocessor, ServeConfig, prepare_metric_inputs


def capture_engine(state_dim=7):
    import torch

    class Policy:
        def predict_action_chunk(self, batch):
            self.batch = batch
            return torch.zeros((1, 50, 7))

    e = object.__new__(PolicyEngine)
    e.cfg = ServeConfig(device="cpu")
    e._aug_rng = None
    e.nogrip, e.input_mode, e.state_dim = True, "masked_rgbd", state_dim
    e.drop_aug, e.prompt_tok = None, None
    e.norm = NormSpec()
    e.pre = Preprocessor("cpu", state_dim, "masked_rgbd", True, True)
    e.policy_type, e.device = "fastwam", "cpu"
    e.policy = Policy()
    e.last_plane = {"n": None, "c": None}
    return e


def message():
    mask = np.zeros((480, 640), np.uint8)
    mask[130:350, 300:340] = 255
    return dict(mask=mask, rgb=np.full((480, 640, 3), 100, np.uint8),
                depth_mm=np.full((480, 640), 250, np.uint16), gmask=np.zeros_like(mask),
                state=np.zeros(14, np.float32), plane_top_cut_m=.01)


def test_no_plane_legacy_reply_does_not_reuse_previous_metric_plane():
    e = capture_engine()
    msg = message()
    e.infer(msg)
    assert e.last_plane["n"] is not None
    e.infer(dict(msg, plane_top_cut_m=0))
    assert e.last_plane == {"n": None, "c": None}
    assert e.last_metric_input is None


@pytest.mark.parametrize("length", [-.01, float("nan"), float("inf")])
def test_policy_rejects_invalid_metric_lengths(length):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        prepare_metric_inputs(dict(message(), plane_top_cut_m=length))


@pytest.mark.parametrize("mode", ["dry", "go", "servers-only"])
def test_bare_metric_launcher_flag_before_mode(mode):
    args = run_real.build_parser().parse_args(["--ckpt", "weights", "--plane-top-cut-m", mode])
    assert args.mode == mode and args.plane_top_cut_m == .01


@pytest.mark.parametrize("head,tail,expected", [
    (["--plane-top-cut", ".2"], [], (.2, 0)),
    ([], ["--plane-top-cut", ".2"], (.2, 0)),
    (["--plane-top-cut-m", ".03"], ["--plane-top-cut", ".2"], (.2, .03)),
    (["--plane-top-cut", ".2"], ["--plane-top-cut-m", ".03"], (.2, .03)),
    (["--plane-top-cut-m", ".03"], ["--plane-top-cut-m", "0"], (0, 0)),
])
def test_launcher_resolved_arguments_reach_client_unchanged(head, tail, expected):
    args = run_real.build_parser().parse_args(["--ckpt", "weights", *head, "dry", *tail])
    preset = run_real.resolve_cut_options(resolve_preset(load_real_config(), "default"),
                                          args.plane_top_cut, args.plane_top_cut_m, args.client_args)
    actual = client.build_parser().parse_args([*preset_client_args(preset), *args.client_args])
    assert (actual.plane_top_cut, actual.plane_top_cut_m) == expected


def test_client_binary_aux_preview_matches_preprocessor_mode(monkeypatch):
    msg = message()
    result = metric.metric_plane_inputs(msg["mask"], msg["depth_mm"], .01)
    snapshot = dict(observation=(msg["rgb"], msg["depth_mm"], msg["mask"]),
                    source_time=10., result_time=10.1, metric_inputs=result, plane=None)
    stream = SimpleNamespace(top_cut_m=.01, input_mode="masked_rgb", nogrip=True,
                             gmask_fn=lambda: msg["gmask"], tracking_snapshot=lambda: snapshot)
    captured = []
    original = client.cv2.applyColorMap

    def colorize(depth, color):
        captured.append(depth.copy())
        return original(depth, color)

    monkeypatch.setattr(client.cv2, "applyColorMap", colorize)
    client._live_panel(msg["rgb"], msg["depth_mm"], msg["mask"], stream, 0, 0)
    assert set(np.unique(captured[0])).issubset({0, 255})


def test_client_rgb_only_preview_shows_rgb_without_inventing_an_aux_stream(monkeypatch):
    msg = message()
    result = metric.metric_plane_inputs(msg["mask"], msg["depth_mm"], .01)
    snapshot = dict(observation=(msg["rgb"], msg["depth_mm"], msg["mask"]),
                    source_time=10., result_time=10.1, metric_inputs=result, plane=None)
    stream = SimpleNamespace(top_cut_m=.01, input_mode="masked_rgb_nodepth", nogrip=True,
                             gmask_fn=lambda: msg["gmask"], tracking_snapshot=lambda: snapshot)

    def no_depth_color_map(*args):
        raise AssertionError("RGB-only model has no depth or binary auxiliary stream")

    monkeypatch.setattr(client.cv2, "applyColorMap", no_depth_color_map)
    panel = client._live_panel(msg["rgb"], msg["depth_mm"], msg["mask"], stream, 0, 0)
    np.testing.assert_array_equal(panel[250, 640 + 320], msg["rgb"][250, 320, ::-1])


def test_disabling_direct_client_metric_default_selects_uncut_legacy(monkeypatch):
    for key in ("SIMRETRACT_PLANE_TOP_CUT", "PLANE_TOP_CUT"):
        monkeypatch.delenv(key, raising=False)
    args = client.build_parser().parse_args(["--plane-top-cut-m", "0"])
    assert args.plane_top_cut == 0
