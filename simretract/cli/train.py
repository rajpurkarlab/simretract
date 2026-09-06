"""``simretract-train`` console entry point.

Thin wrapper around ``simretract.train.train.main``.  The wrapper is
environment-agnostic: when the current interpreter has no ``lerobot`` the
job is re-launched under the policy interpreter recorded by ``envs/setup.sh``
(``simretract.config.interpreter("policy")``; ``"fastwam"`` for the
fastwam policy).  ``--help`` always runs in-process.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import Sequence


def _needs_reexec(argv: Sequence[str]) -> bool:
    if any(x in ("-h", "--help") for x in argv):
        return False
    return importlib.util.find_spec("lerobot") is None


def _role(argv: Sequence[str]) -> str:
    """Interpreter role for the requested policy (fastwam has its own env)."""
    args = list(argv)
    for i, x in enumerate(args):
        for flag in ("--policy", "--preset"):
            if x == flag and i + 1 < len(args):
                val = args[i + 1]
            elif x.startswith(flag + "="):
                val = x.split("=", 1)[1]
            else:
                continue
            if val == "fastwam":
                return "fastwam"
    return "policy"


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if _needs_reexec(argv):
        from simretract import config as srconfig

        python = str(srconfig.interpreter(_role(argv)))
        os.execv(python, [python, "-m", "simretract.train.train", *argv])
    from simretract.train.train import main as train_main

    return train_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
