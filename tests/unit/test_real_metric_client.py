"""Client integration checks with synthetic observations and inert hardware fakes."""
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from simretract.real import client
from simretract.real.startup import StartupReadiness, StartupTrackingError


@pytest.mark.parametrize("flags,expected", [
    ([], .01), (["--plane-top-cut-m"], .01), (["--plane-top-cut-m", "0"], 0.),
    (["--plane-top-cut", ".2"], 0.), (["--plane-top-cut=0"], 0.),
    (["--plane-top-cut-m", ".02", "--plane-top-cut", ".1"], .02),
    (["--plane-top-cut", ".1", "--plane-top-cut-m", ".02"], .02),
])
def test_metric_default_and_explicit_legacy_selection(monkeypatch, flags, expected):
    for key in ("SIMRETRACT_PLANE_TOP_CUT_M", "PLANE_TOP_CUT_M"):
        monkeypatch.delenv(key, raising=False)
    assert client.build_parser().parse_args(flags).plane_top_cut_m == expected


@pytest.mark.parametrize("length", ["-1", "nan", "inf"])
def test_invalid_metric_length_rejected_by_parser(length):
    with pytest.raises(SystemExit):
        client.build_parser().parse_args(["--plane-top-cut-m", length])


def observation():
    rgb = np.full((480, 640, 3), 100, np.uint8)
    depth = np.full((480, 640), 250, np.uint16)
    mask = np.zeros((480, 640), np.uint8)
    mask[150:350, 280:340] = 255
    return rgb, depth, mask


def wire_inputs(mask, aux=True):
    effective = mask.copy()
    effective[150:165] = 0
    removed = np.where((mask > 0) & (effective == 0), 255, 0).astype(np.uint8)
    wrist = np.linspace(0, 1, 12 * 16 * 3, dtype=np.float32).reshape(12, 16, 3)
    extra = np.full_like(wrist, .25) if aux else None
    wire = dict(shape=mask.shape, mask=effective.tobytes(), removed_mask=removed.tobytes(),
                camera_shape=wrist.shape, wrist=wrist.tobytes(),
                aux=None if extra is None else extra.tobytes(), status="synthetic", length_m=.01,
                input_mode="masked_rgbd", nogrip=True)
    return wire, wrist, extra


@pytest.mark.parametrize("aux", [True, False])
def test_inspection_preserves_exact_server_cameras_and_raw_sam(tmp_path, aux):
    rgb, depth, mask = observation()
    wire, wrist, extra = wire_inputs(mask, aux)
    client.save_inspect(str(tmp_path), 0, rgb, depth, mask, np.zeros_like(mask), wire)
    with np.load(tmp_path / "q_000_raw.npz") as saved:
        np.testing.assert_array_equal(saved["mask"], mask)
        np.testing.assert_array_equal(saved["dep_mm"], depth)
        np.testing.assert_array_equal(saved["model_wrist"], wrist)
        np.testing.assert_array_equal(saved["model_aux"], np.empty((0,), np.float32) if extra is None else extra)
        assert np.count_nonzero(saved["effective_mask"]) < np.count_nonzero(mask)


def install_episode_fakes(monkeypatch, tmp_path, *, mask_capabilities=True, startup_error=False):
    rgb, depth, mask = observation()
    wire, _, _ = wire_inputs(mask)
    calls = {"camera": 0, "arm": 0, "query": [], "startup": 0, "home": [], "closed": []}
    lut_path = tmp_path / "lut.npz"
    grip = np.full_like(mask, 255)
    np.savez(lut_path, openings=[.044], masks=grip[None])

    class FakeClient:
        def __init__(self, host, port):
            self.port = port

        def request(self, request):
            if request.get("op") == "ping":
                if self.port == 8765:
                    return dict(ckpt="weights", nogrip=True, input_mode="masked_rgbd", metric_plane_addon=1)
                return dict(prompt="metal rake", warm_reset=mask_capabilities,
                            locked_track_keepalive=mask_capabilities)
            if request.get("op") == "reset":
                return {"ok": True}
            calls["query"].append(request)
            return dict(chunk=np.zeros((50, 7), np.float32), metric_inputs=wire)

        def close(self):
            calls["closed"].append(self.port)

    class FakeCamera:
        def __init__(self):
            calls["camera"] += 1

        def stop(self):
            calls["camera_stopped"] = True

    class FakeArm:
        def __init__(self, *args):
            calls["arm"] += 1
            self.present = np.array([0.] * 6 + [.044])

        def read(self):
            return self.present.copy()

        def enable_position_mode(self):
            pass

        def goto(self, pose, duration):
            self.present = pose.copy()

        def cleanup(self):
            calls["arm_cleaned"] = True

    class FakeStream:
        n_frame, frames_dir = 0, None

        def __init__(self, *args, **kwargs):
            calls["stream_options"] = kwargs

        def get_startup(self, after_ts):
            calls["startup"] += 1
            if startup_error:
                raise StartupTrackingError("synthetic startup failure")
            return rgb, depth, mask

        def get(self):
            raise AssertionError("first query must require startup readiness")

        def close(self):
            calls["stream_closed"] = True

    monkeypatch.setattr(client, "PolicyClient", FakeClient)
    monkeypatch.setattr(client, "D405", FakeCamera)
    monkeypatch.setattr(client, "Arm", FakeArm)
    monkeypatch.setattr(client, "MaskStream", FakeStream)
    monkeypatch.setattr(client, "ramp_to", lambda *args: 0)
    monkeypatch.setattr(client, "_return_home", lambda arm, aborted, speed: calls["home"].append(aborted))
    monkeypatch.setattr(client.time, "sleep", lambda duration: None)
    args = client.build_parser().parse_args(["--go", "--yes", "--follower-ip", "controller", "--max-ticks", "1",
                                            "--close-stop", "0", "--lut", str(lut_path),
                                            "--out-dir", str(tmp_path / "episode")])
    return calls, args


def test_capability_rejection_precedes_hardware_access(monkeypatch, tmp_path):
    calls, args = install_episode_fakes(monkeypatch, tmp_path, mask_capabilities=False)
    with pytest.raises(StartupTrackingError, match="missing capabilities"):
        client.run(args)
    assert calls["camera"] == calls["arm"] == 0
    assert sorted(calls["closed"]) == [8765, 8766]


def test_first_query_waits_for_startup_and_carries_metric_gripper_exclusion(monkeypatch, tmp_path):
    calls, args = install_episode_fakes(monkeypatch, tmp_path)
    client.run(args)
    assert calls["startup"] == 1
    assert calls["stream_options"]["top_cut_m"] == .01
    request, = calls["query"]
    assert request["plane_top_cut_m"] == .01
    assert not request["gmask"].any()
    assert request["metric_gripper_mask"].all()
    assert calls["home"] == [False]
    assert calls["arm_cleaned"] and calls["stream_closed"] and calls["camera_stopped"]


def test_startup_failure_cancels_queries_and_uses_original_abort_cleanup(monkeypatch, tmp_path):
    calls, args = install_episode_fakes(monkeypatch, tmp_path, startup_error=True)
    with pytest.raises(StartupTrackingError):
        client.run(args)
    assert not calls["query"]
    assert calls["home"] == [True]
    assert calls["arm_cleaned"] and calls["stream_closed"] and calls["camera_stopped"]


def test_metric_worker_clears_stale_plane_and_preserves_raw_mask(monkeypatch):
    rgb, depth, mask = observation()
    stream = client.MaskStream.__new__(client.MaskStream)
    stream.stop, stream.top_cut_m, stream.top_cut = False, .01, .1
    stream.lock = threading.Lock()
    stream.startup = StartupReadiness()
    stream.local_plane = {"n": [0, 0, -1], "c": [0, 0, .2]}
    stream.gmask_fn = lambda: np.zeros_like(mask)
    effective = mask.copy()
    effective[150:165] = 0
    result = dict(mask=effective, removed_mask=mask - effective, plane=None, status="no plane")
    monkeypatch.setattr(client, "metric_plane_inputs", lambda *args: result)

    def request(message):
        stream.stop = True
        return mask.tobytes()

    stream.masker = SimpleNamespace(request=request)
    stream._mask_worker({"frame": (rgb, depth, 10.)}, threading.Lock())
    snapshot = stream.tracking_snapshot()
    assert snapshot["plane"] == {}
    assert snapshot["source_time"] == 10.
    np.testing.assert_array_equal(snapshot["observation"][2], mask)
    assert snapshot["metric_inputs"] is result


@pytest.mark.parametrize("stage,error", [("confirmation", KeyboardInterrupt), ("home", RuntimeError),
                                         ("arm_read", RuntimeError), ("mask_connect", RuntimeError)])
def test_preflight_failure_releases_resources_without_adding_home(monkeypatch, tmp_path, stage, error):
    calls, args = install_episode_fakes(monkeypatch, tmp_path)

    def fail(*args, **kwargs):
        raise error("synthetic preflight interruption")

    if stage == "confirmation":
        args.yes = False
        monkeypatch.setattr("builtins.input", fail)
    elif stage == "home":
        monkeypatch.setattr(client, "ramp_to", fail)
    elif stage == "arm_read":
        monkeypatch.setattr(client.Arm, "read", fail)
    else:
        original = client.PolicyClient

        class FailingConnection(original):
            def __init__(self, host, port):
                if port == 8766:
                    fail()
                super().__init__(host, port)

        monkeypatch.setattr(client, "PolicyClient", FailingConnection)
    with pytest.raises(error):
        client.run(args)
    assert calls["home"] == [] and calls["query"] == []
    assert sorted(calls["closed"]) == ([8765] if stage == "mask_connect" else [8765, 8766])
    if stage != "mask_connect":
        assert calls["arm_cleaned"] and calls["camera_stopped"]
