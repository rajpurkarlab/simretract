"""Training: one entry point (``simretract.train.train``) for every policy row.

``presets`` maps the named rows of ``configs/train/presets.yaml`` to trainer
flags; ``train`` holds the dataset, augmentation hooks, policy builders and
the training loop.
"""
