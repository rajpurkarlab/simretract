"""Connection and preview startup regressions using inert driver modules."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from simretract.real import client, live_view
from simretract.real.startup import StartupTrackingError


def test_camera_constructor_stops_a_partially_started_pipeline(monkeypatch):
    events = []

    def fail_after_start():
        raise RuntimeError("synthetic profile failure")

    pipeline = SimpleNamespace(start=lambda cfg: SimpleNamespace(get_device=fail_after_start),
                               stop=lambda: events.append("stopped"))
    fake = SimpleNamespace(pipeline=lambda: pipeline, config=lambda: SimpleNamespace(enable_stream=lambda *args: None),
                           stream=SimpleNamespace(color=0, depth=1), format=SimpleNamespace(rgb8=0, z16=1),
                           align=lambda stream: object())
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake)
    with pytest.raises(RuntimeError, match="profile failure"):
        client.D405()
    assert events == ["stopped"]


@pytest.mark.parametrize("stage", ["configure", "read"])
def test_arm_constructor_releases_partial_connection(monkeypatch, stage):
    events = []

    def configure(**kwargs):
        events.append("configure")
        if stage == "configure":
            raise RuntimeError("synthetic configure failure")

    def read():
        raise RuntimeError("synthetic read failure")

    driver = SimpleNamespace(configure=configure, cleanup=lambda: events.append("cleanup"),
                             set_motor_parameters=lambda params: None, get_motor_parameters=lambda: [],
                             get_all_positions=read)
    fake = SimpleNamespace(TrossenArmDriver=lambda: driver, Model=SimpleNamespace(wxai_v0=0),
                           StandardEndEffector=SimpleNamespace(wxai_v0_follower=0),
                           StandardMotorParameters=SimpleNamespace(wxai_v0_latest=0))
    monkeypatch.setitem(sys.modules, "trossen_arm", fake)
    with pytest.raises(RuntimeError, match="synthetic"):
        client.Arm("controller")
    assert events == ["configure", "cleanup"]


@pytest.mark.parametrize("capabilities", [False, True])
def test_live_preview_verifies_and_resets_tracker_before_camera(monkeypatch, tmp_path, capabilities):
    events = []
    lut = tmp_path / "lut.npz"
    np.savez(lut, openings=[.044], masks=np.zeros((1, 480, 640), np.uint8))

    class MaskClient:
        def __init__(self, *args):
            events.append("connect")

        def request(self, message):
            events.append(message["op"])
            return dict(ok=True, warm_reset=capabilities, locked_track_keepalive=capabilities)

        def close(self):
            events.append("close")

    def camera(**kwargs):
        events.append("camera")
        raise RuntimeError("synthetic camera failure")

    monkeypatch.setenv("DISPLAY", ":test")
    monkeypatch.setattr(live_view, "PolicyClient", MaskClient)
    monkeypatch.setattr(live_view, "D405", camera)
    expected_error = RuntimeError if capabilities else StartupTrackingError
    with pytest.raises(expected_error):
        live_view.main(["--lut", str(lut)])
    assert events == (["connect", "ping", "reset", "camera", "close"] if capabilities
                      else ["connect", "ping", "close"])


def test_live_preview_window_failure_releases_camera_socket_and_window(monkeypatch, tmp_path):
    events = []
    lut = tmp_path / "lut.npz"
    np.savez(lut, openings=[.044], masks=np.zeros((1, 480, 640), np.uint8))
    masker = SimpleNamespace(request=lambda msg: dict(warm_reset=True, locked_track_keepalive=True),
                             close=lambda: events.append("socket"))
    camera = SimpleNamespace(stop=lambda: events.append("camera"))

    def fail_resize(*args):
        raise RuntimeError("synthetic window failure")

    monkeypatch.setenv("DISPLAY", ":test")
    monkeypatch.setattr(live_view, "PolicyClient", lambda *args: masker)
    monkeypatch.setattr(live_view, "D405", lambda **kwargs: camera)
    monkeypatch.setattr(live_view.cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(live_view.cv2, "resizeWindow", fail_resize)
    monkeypatch.setattr(live_view.cv2, "destroyAllWindows", lambda: events.append("window"))
    with pytest.raises(RuntimeError, match="window failure"):
        live_view.main(["--lut", str(lut)])
    assert events == ["window", "camera", "socket"]
