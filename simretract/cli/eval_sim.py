"""``simretract-eval-sim``: the simulation benchmark for one checkpoint.

Thin console entry point over ``simretract.eval.driver`` (server boot,
one Isaac Sim process per seed, multi-pass retries, funnel scoring).  It is
environment-agnostic: the policy server and the evaluator are started with
the interpreters recorded by ``envs/setup.sh`` (``configs/envs.local.yaml``).
"""

from __future__ import annotations

import sys

from simretract.eval import driver


def main(argv: list[str] | None = None) -> int:
    return driver.main(argv)


if __name__ == "__main__":
    sys.exit(main())
