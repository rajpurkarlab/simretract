"""Model-input and deployment contracts for the optional metric path."""
from unittest.mock import patch
import numpy as np
import pytest

from simretract.cli import run_real
from simretract.real import load_real_config, resolve_preset
from simretract.serve import metric
from simretract.serve.policy_server import (NormSpec, PolicyEngine, Preprocessor,
                                            ServeConfig, prepare_metric_inputs)
from simretract.serve.protocol import ensure_wire_safe


def observation():
    mask = np.zeros((480, 640), np.uint8)
    mask[130:350, 307:333] = 255
    rgb = np.full((480, 640, 3), 127, np.uint8)
    depth = np.where(mask, 250, 600).astype(np.uint16)
    return dict(rgb=rgb, depth_mm=depth, mask=mask, gmask=np.zeros_like(mask),
                state=np.arange(14, dtype=np.float32) * .01, plane_top_cut_m=.01)


def engine(mode='masked_rgbd', state_dim=10, float_depth=True, nogrip=True):
    import torch
    class CapturePolicy:
        def predict_action_chunk(self, batch):
            self.batch = batch
            return torch.zeros((1, 50, 7))
    e = object.__new__(PolicyEngine)
    e.cfg = ServeConfig(device='cpu')
    e._aug_rng = None
    e.nogrip, e.input_mode, e.state_dim = nogrip, mode, state_dim
    e.drop_aug = None
    e.norm = NormSpec()
    e.pre = Preprocessor('cpu', state_dim, mode, float_depth, nogrip)
    e.prompt_tok = None
    e.policy_type = 'fastwam'  # Own precision; the capture policy stays on CPU.
    e.policy = CapturePolicy()
    e.device = 'cpu'
    e.last_plane = {'n': None, 'c': None}
    return e


@pytest.mark.parametrize('mode', ['masked_rgbd', 'full_rgbd', 'masked_rgb', 'masked_rgb_nodepth'])
@pytest.mark.parametrize('float_depth', [True, False])
@pytest.mark.parametrize('nogrip', [True, False])
def test_metric_mask_plane_and_inspection_share_current_input(mode, float_depth, nogrip):
    msg = observation()
    original = {k: v.copy() for k, v in msg.items() if isinstance(v, np.ndarray)}
    e = engine(mode, float_depth=float_depth, nogrip=nogrip)
    with patch.object(metric, 'metric_plane_inputs', wraps=metric.metric_plane_inputs) as fit:
        chunk = e.infer(msg)
    assert fit.call_count == 1
    assert chunk.shape == (50, 7)
    result = e.last_metric_input
    keep = np.frombuffer(result['mask'], np.uint8).reshape(result['shape']) > 0
    removed = np.frombuffer(result['removed_mask'], np.uint8).reshape(result['shape']) > 0
    assert keep.any() and removed.any() and not (keep & removed).any()
    prepared = prepare_metric_inputs(msg)
    assert not prepared['rgb'][removed].any() and not prepared['depth_mm'][removed].any()
    for key, camera in [('wrist', 'observation.images.wrist'), ('aux', 'observation.images.mask')]:
        if result[key] is None:
            assert camera not in e.policy.batch
        else:
            inspected = np.frombuffer(result[key], np.float32).reshape(result['camera_shape'])
            actual = e.policy.batch[camera][0].permute(1, 2, 0).numpy()
            np.testing.assert_array_equal(inspected, actual)
    np.testing.assert_array_equal(e.policy.batch['observation.state'][0, 7:], e.last_plane['n'])
    for key, before in original.items():
        np.testing.assert_array_equal(msg[key], before)
    ensure_wire_safe({'chunk': chunk.tolist(), 'metric_inputs': result, 'plane': e.last_plane})


def test_no_plane_checkpoint_keeps_its_state_width_and_cropped_images():
    msg = observation()
    e = engine(state_dim=7)
    e.infer(msg)
    assert e.policy.batch['observation.state'].shape == (1, 7)
    np.testing.assert_array_equal(e.policy.batch['observation.state'][0], msg['state'][:7])
    assert e.last_metric_input and e.last_plane['n'] is not None


def test_legacy_query_has_no_metric_side_effects_or_stale_inspection():
    e = engine()
    e.infer(observation())
    msg = observation()
    msg['plane_top_cut_m'] = 0
    assert prepare_metric_inputs(msg) is msg
    with patch.object(metric, 'metric_plane_inputs', side_effect=AssertionError('metric called')):
        e.infer(msg)
    assert e.last_metric_input is None


def test_projected_fallback_preserves_visibility_and_removes_the_tip():
    msg = observation()
    mask, depth = msg['mask'], msg['depth_mm']
    empty = dict(mask=np.zeros_like(mask), removed_mask=np.zeros_like(mask),
                 plane=None, status='unreliable shaft axis')
    with patch.object(metric, '_metric_plane_inputs', return_value=empty):
        result = metric.metric_plane_inputs(mask, depth, .02)
    assert result['mask'].any() and result['removed_mask'].any()
    assert result['cut_mode'] == 'projected_depth_fallback'
    assert result['mask'][130, 320] == 0 and result['mask'][349, 320] == 255
    assert not ((result['mask'] > 0) & (result['removed_mask'] > 0)).any()
    assert result['plane']['c'][2] == pytest.approx(.25)
    with patch.object(metric, '_metric_plane_inputs', return_value=empty):
        absent = metric.metric_plane_inputs(np.zeros_like(mask), depth, .02)
    assert not absent['mask'].any()


@pytest.mark.parametrize('extra, fraction, length, expected', [
    ([], None, None, (0., .01)),
    ([], .1, None, (.1, 0.)),
    (['--plane-top-cut', '.1'], None, None, (.1, 0.)),
    (['--plane-top-cut-m', '0'], None, None, (0., 0.)),
    (['--plane-top-cut', '.1', '--plane-top-cut-m', '.02'], None, None, (.1, .02)),
    (['--plane-top-cut-m'], None, None, (0., .01)),
])
def test_cut_overrides_before_or_after_mode(extra, fraction, length, expected):
    preset = resolve_preset(load_real_config(), 'default')
    result = run_real.resolve_cut_options(preset, fraction, length, extra)
    assert (result['plane_top_cut'], result['plane_top_cut_m']) == expected


def test_launcher_rejects_stale_mask_or_metric_policy(tmp_path):
    ckpt = str(tmp_path / 'sam.pt')
    good = dict(ok=True, ckpt=ckpt, prompt='metal rake', conf=.2, protocol_version=2,
                warm_reset=True, locked_track_keepalive=True)
    assert run_real.mask_regime_ok(good, ckpt, 'metal rake', .2)
    for change in ({'warm_reset': False}, {'locked_track_keepalive': False},
                   {'ckpt': ckpt+'.old'}, {'protocol_version': 1}):
        assert not run_real.mask_regime_ok(dict(good, **change), ckpt, 'metal rake', .2)
    pp = dict(ok=True, ckpt=ckpt, eval_aug='0')
    assert not run_real.policy_regime_ok(pp, ckpt, metric_required=True)[0]
    assert run_real.policy_regime_ok(dict(pp, metric_plane_addon=1), ckpt, metric_required=True)[0]


def test_old_custom_preset_remains_legacy():
    cfg = load_real_config()
    cfg['presets']['old'] = dict(cfg['presets']['default'])
    del cfg['presets']['old']['plane_top_cut_m']
    cfg['presets']['old']['plane_top_cut'] = .1
    assert resolve_preset(cfg, 'old')['plane_top_cut_m'] == 0
