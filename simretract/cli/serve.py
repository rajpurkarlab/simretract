"""``simretract-serve``: start the policy server in the right environment.

The command itself is environment-agnostic.  It resolves the interpreter for
the checkpoint type (``policy`` for pi0/pi05/pi0_fast/smolvla/act,
``fastwam`` for fastwam) from the interpreter table written by
``envs/setup.sh`` (``configs/envs.local.yaml``) and re-executes
``python -m simretract.serve.policy_server`` there with every other argument
passed through unchanged.  ``--stub`` needs no weights and runs in-process.
"""

from __future__ import annotations

import argparse
import os
import sys

from simretract.serve import policy_server


def _parse(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        prog="simretract-serve",
        description="Serve a checkpoint (or a hold-pose stub) over TCP. Any "
                    "flag not listed here is forwarded to "
                    "simretract.serve.policy_server (see its --help).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", default=os.environ.get("CKPT", ""),
                   help="checkpoint directory (flat or with a final/ subdirectory)")
    p.add_argument("--stub", action="store_true",
                   help="serve hold-pose chunks in this interpreter, no weights")
    p.add_argument("--interpreter", default="",
                   help="python executable to run the server with "
                        "(default: the role interpreter recorded by envs/setup.sh, "
                        "chosen by checkpoint type)")
    return p.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    args, rest = _parse(sys.argv[1:] if argv is None else argv)
    if args.stub:
        return policy_server.main(["--stub", *rest])
    if not args.ckpt:
        print("simretract-serve: --ckpt is required unless --stub is given",
              file=sys.stderr)
        return 2
    from simretract import config

    ckpt = policy_server.resolve_checkpoint_dir(args.ckpt)
    role = "fastwam" if policy_server.checkpoint_type(ckpt) == "fastwam" else "policy"
    python = args.interpreter or str(config.interpreter(role))
    cmd = [python, "-m", "simretract.serve.policy_server", "--ckpt", str(ckpt), *rest]
    os.execv(python, cmd)
    return 0  # pragma: no cover - execv does not return


if __name__ == "__main__":
    sys.exit(main())
