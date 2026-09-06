"""``simretract-build-cache``: build one frame-cache set (+ plane sidecar)
from raw episode directories. See ``simretract.data.cache`` for the layout
and rules; ``--set strict|recovery`` selects the production flag preset.
"""
from __future__ import annotations

from simretract.data.cache import main

if __name__ == "__main__":
    raise SystemExit(main())
