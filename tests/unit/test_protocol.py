"""Wire framing round trip and the reply-side numpy guard."""

import pickle
import socket
import threading

import numpy as np
import pytest

from simretract.serve import protocol
from simretract.serve.protocol import (PolicyClient, WireTypeError, ensure_wire_safe,
                                       recv_msg, send_msg)


def _query():
    return {"rgb": np.zeros((480, 640, 3), np.uint8),
            "depth_mm": np.full((480, 640), 500, np.uint16),
            "mask": np.zeros((480, 640), np.uint8),
            "gmask": np.zeros((480, 640), np.uint8),
            "state": np.zeros(14, np.float32),
            "plane_top_cut": 0.0}


def test_roundtrip_query_and_reply_over_socketpair():
    a, b = socket.socketpair()
    try:
        send_msg(a, _query())
        msg = recv_msg(b)
        assert msg["rgb"].shape == (480, 640, 3)
        assert msg["depth_mm"].dtype == np.uint16
        assert msg["plane_top_cut"] == 0.0
        chunk = np.arange(350, dtype=np.float32).reshape(50, 7)
        reply = chunk.tolist()
        ensure_wire_safe(reply)
        send_msg(b, reply)
        got = recv_msg(a)
        assert isinstance(got, list) and not any(hasattr(x, "__array__") for x in got)
        assert np.asarray(got, np.float32).shape == (50, 7)
        assert np.array_equal(np.asarray(got, np.float32), chunk)
    finally:
        a.close()
        b.close()


def test_recv_returns_none_on_clean_close_and_raises_on_oversize():
    a, b = socket.socketpair()
    a.close()
    assert recv_msg(b) is None
    b.close()
    a, b = socket.socketpair()
    try:
        a.sendall(protocol._HEADER.pack(protocol.MAX_MESSAGE_BYTES + 1))
        with pytest.raises(ValueError):
            recv_msg(b)
    finally:
        a.close()
        b.close()


def test_recv_handles_fragmented_frames():
    a, b = socket.socketpair()
    try:
        data = pickle.dumps({"chunk": [[0.0] * 7] * 50}, protocol=pickle.HIGHEST_PROTOCOL)
        frame = protocol._HEADER.pack(len(data)) + data
        # deliver in small pieces from a thread while the receiver blocks
        def feed():
            for i in range(0, len(frame), 97):
                a.sendall(frame[i:i + 97])
        t = threading.Thread(target=feed)
        t.start()
        got = recv_msg(b)
        t.join()
        assert len(got["chunk"]) == 50
    finally:
        a.close()
        b.close()


@pytest.mark.parametrize("bad", [
    np.zeros((50, 7), np.float32),
    {"chunk": np.zeros((50, 7)).tolist(), "plane_n": np.zeros(3)},
    [np.float32(1.0)],
    {"ok": True, "n": np.int64(3)},
    {"regime": {"half": np.bool_(True)}},
    object(),
    {1.5: "float keys are not allowed"},
])
def test_ensure_wire_safe_rejects_numpy_and_foreign_objects(bad):
    with pytest.raises(WireTypeError):
        ensure_wire_safe(bad)


@pytest.mark.parametrize("good", [
    [[0.0] * 7] * 50,
    {"ok": True, "ckpt": "x", "eval_aug": "0", "nogrip": False,
     "regime": {"half": True, "drop_px": 300, "plane_top_cut": 0.0, "chunk": 50}},
    {"chunk": [[1, 2]], "plane_n": None, "plane_c": [0.1, 0.2, 0.3]},
    {"error": "RuntimeError('x')"},
    b"raw bytes",
    ("tuple", 1, None),
])
def test_ensure_wire_safe_accepts_plain_python(good):
    ensure_wire_safe(good)


def test_policy_client_ping_and_error_paths():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def serve():
        conn, _ = srv.accept()
        with conn:
            while True:
                msg = recv_msg(conn)
                if msg is None:
                    break
                if msg.get("op") == "ping":
                    send_msg(conn, {"ok": True, "ckpt": "stub", "regime": {"stub": True}})
                elif msg.get("op") == "boom":
                    send_msg(conn, {"error": "ValueError('boom')"})
                else:
                    send_msg(conn, [[0.0] * 7] * 50)

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    with PolicyClient("127.0.0.1", port, timeout=5) as cl:
        assert cl.ping()["regime"] == {"stub": True}
        assert len(cl.request({"op": "query"})) == 50
        with pytest.raises(RuntimeError):
            cl.request({"op": "boom"})
    t.join(timeout=5)
    srv.close()
