"""Training-data pipeline: raw episode dirs -> frame cache -> plane sidecar
-> optional half-data subset, plus hub downloads and the alignment gate.

Modules:
    cache          build one cache set from raw episode directories
    plane_sidecar  per-frame plane-normal sidecar of a cache set
    seg_aug        train-time mask / depth augmentation (shared with the trainer)
    select_half    deterministic half-data subset selection
    verify_align   bitwise cache-vs-trainer alignment gate
    hub            downloads of raw shards, checkpoints, base and gated weights
"""
