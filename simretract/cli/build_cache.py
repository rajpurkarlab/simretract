"""``simretract-build-cache``: build one frame-cache set (+ plane sidecar)
from raw episode directories. See ``simretract.data.cache`` for the layout
and rules; ``--set`` names the cache set (strict, recovery, strict_nosmooth,
recovery_nosmooth, strict_nostandoff, recovery_nostandoff) and selects its
production flag preset.
"""
from __future__ import annotations

from simretract.data.cache import main

if __name__ == "__main__":
    raise SystemExit(main())
