"""Length-prefixed pickle framing shared by the policy server, the mask
server and their clients.

Wire format (both directions): an 8-byte big-endian unsigned length followed
by a pickle of the payload.  Requests may carry numpy arrays (the sender's
numpy version pickles them; every receiver runs an equal or newer numpy).
Replies must NOT carry numpy objects: the policy environment runs numpy 2.x
while the Isaac Sim environment runs numpy 1.x, and numpy-2 array pickles do
not load there.  ``ensure_wire_safe`` enforces the reply rule before a
message is sent.
"""

from __future__ import annotations

import pickle
import socket
import struct
from typing import Any

__all__ = [
    "MAX_MESSAGE_BYTES",
    "PolicyClient",
    "WireTypeError",
    "ensure_wire_safe",
    "recv_msg",
    "send_msg",
]

_HEADER = struct.Struct(">Q")
#: Upper bound on a single framed message; a 640x480 RGB-D query is ~2 MB.
MAX_MESSAGE_BYTES = 1 << 30
_WIRE_SCALARS = (bool, int, float, str, bytes, bytearray, type(None))


class WireTypeError(TypeError):
    """Raised when a reply payload contains an object that is not plain
    Python data (numpy arrays or scalars, torch tensors, custom classes)."""


def _is_array_like(obj: Any) -> bool:
    # numpy arrays and scalars, torch tensors and friends all expose
    # __array__ or __array_interface__; plain Python containers do not.
    return hasattr(obj, "__array__") or hasattr(obj, "__array_interface__")


def ensure_wire_safe(obj: Any, _path: str = "reply") -> None:
    """Raise ``WireTypeError`` unless ``obj`` is built only from Python
    scalars, ``bytes``, ``str``, ``None`` and (nested) lists/tuples/dicts.

    Array-like leaves (anything with ``__array__``) are rejected even when
    they are zero-dimensional, because numpy scalars pickle as numpy
    objects too.
    """
    if _is_array_like(obj):
        raise WireTypeError(
            f"{_path}: {type(obj).__name__} is not allowed on the wire; "
            "convert arrays with .tolist() and scalars with float()/int()")
    if isinstance(obj, _WIRE_SCALARS):
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, (str, int)):
                raise WireTypeError(f"{_path}: dict key {k!r} must be str or int")
            ensure_wire_safe(v, f"{_path}[{k!r}]")
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            ensure_wire_safe(v, f"{_path}[{i}]")
        return
    raise WireTypeError(f"{_path}: {type(obj).__name__} is not allowed on the wire")


def send_msg(conn: socket.socket, obj: Any) -> None:
    """Pickle ``obj`` and send it with the 8-byte length prefix."""
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(_HEADER.pack(len(data)) + data)


def _recv_exact(conn: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def recv_msg(conn: socket.socket) -> Any | None:
    """Receive one framed message; ``None`` when the peer closed cleanly."""
    hdr = _recv_exact(conn, _HEADER.size)
    if hdr is None:
        return None
    n = _HEADER.unpack(hdr)[0]
    if n > MAX_MESSAGE_BYTES:
        raise ValueError(f"framed message of {n} bytes exceeds MAX_MESSAGE_BYTES")
    body = _recv_exact(conn, n)
    if body is None:
        return None
    return pickle.loads(body)


class PolicyClient:
    """One persistent connection to a policy (or mask) server.

    ``request`` raises ``RuntimeError`` when the server answers with an
    ``{"error": ...}`` dict and ``ConnectionError`` when the socket closes,
    so a client loop can decide whether to retry or abort.
    """

    def __init__(self, host: str, port: int, timeout: float = 300.0) -> None:
        self.sock = socket.create_connection((host, port), timeout=timeout)

    def request(self, obj: Any) -> Any:
        send_msg(self.sock, obj)
        reply = recv_msg(self.sock)
        if reply is None:
            raise ConnectionError("server closed the connection")
        if isinstance(reply, dict) and "error" in reply:
            raise RuntimeError(f"server error: {reply['error']}")
        return reply

    def ping(self) -> dict:
        """Return the server's ping reply (``{"ok": True, ...}``)."""
        reply = self.request({"op": "ping"})
        if not (isinstance(reply, dict) and reply.get("ok")):
            raise RuntimeError(f"unexpected ping reply: {reply!r}")
        return reply

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

    def __enter__(self) -> "PolicyClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
