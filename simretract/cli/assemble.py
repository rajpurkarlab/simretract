"""``simretract-assemble``: build training selections from raw episodes.

Thin console entry point over :mod:`simretract.sim.assemble`::

    simretract-assemble strict --raw data/raw/strict --per-bin 125 \\
        --out data/selections/strict_500.json --train-dir data/raw/strict_train
    simretract-assemble recovery --raw data/raw/recovery \\
        --strict-json data/selections/strict_500.json --exact 4 --far 1000 \\
        --out-train data/selections/recovery_train.json \\
        --out-far data/selections/recovery_far1000.json
"""

from __future__ import annotations

from simretract.sim.assemble import main as _main


def main(argv: list[str] | None = None) -> int:
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
