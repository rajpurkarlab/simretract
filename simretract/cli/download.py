"""``simretract-download``: fetch raw data shards, checkpoints or base
weights as configured in ``configs/hub.yaml``.

    simretract-download data --set strict --out data/raw/strict_train
    simretract-download data --set recovery --out data/raw/recovery_far1000
    simretract-download data --set strict_nosmooth --out data/raw/strict_no_smoothing
    simretract-download checkpoints baseline no_smooth
    simretract-download base-weights pi0_base
    simretract-download base-weights paligemma_tokenizer smolvla --hub-cache
"""
from __future__ import annotations

from simretract.data.hub import main

if __name__ == "__main__":
    raise SystemExit(main())
