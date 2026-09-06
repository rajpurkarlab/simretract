"""``simretract-download``: fetch raw data shards, checkpoints or base
weights as configured in ``configs/hub.yaml``.

    simretract-download data --set strict --out data/raw/strict
    simretract-download data --set recovery --out data/raw/recovery
    simretract-download checkpoints baseline act
    simretract-download base-weights pi0_base
    simretract-download base-weights paligemma_tokenizer smolvla --hub-cache
"""
from __future__ import annotations

from simretract.data.hub import main

if __name__ == "__main__":
    raise SystemExit(main())
