"""``simretract-generate``: record strict or recovery demonstration sets.

Environment-agnostic: it only drives simulator subprocesses through
:mod:`simretract.sim.campaign` (the simulator interpreter is the
``sim_python`` recorded by ``envs/setup.sh``). Recorder knobs and campaign
defaults come from ``configs/generation/<mode>.yaml``; every knob is a flag
with an environment fallback (see ``--help``).

Examples::

    simretract-generate strict --out data/raw/strict --seed-start 200000 \\
        --quota 125 --bins 4 --z-range -0.05 0.05 --parallel 3
    simretract-generate strict --out data/raw/strict --seeds 200350,400851
    simretract-generate strict --out data/raw/strict --seed-start 601001 \\
        --n-seeds 40 --fill-bin 0 --z-strict
    simretract-generate recovery --out data/raw/recovery \\
        --strict-json data/selections/strict_500.json --winners data/raw/strict_train \\
        --draws 4 --range 0:2000
    simretract-generate recovery ... --topup --draws 8 --target-keeps 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from simretract.sim import assemble, campaign
from simretract.sim.recorder import RecorderConfig, parse_seed_list
from simretract.sim.recovery_recorder import RECOVERY_DEFAULTS

log = logging.getLogger(__name__)


def _campaign_defaults(kind: str) -> dict:
    """The ``campaign:`` block of ``configs/generation/<kind>.yaml`` (``{}`` when absent)."""
    from simretract import config

    try:
        block = config.generation_recipe(kind).get("campaign") or {}
    except config.ConfigError:
        return {}
    return dict(block) if isinstance(block, dict) else {}


def _common(ap: argparse.ArgumentParser, kind: str, defaults: dict | None = None) -> None:
    ap.add_argument("--out", required=True, help="raw episode root")
    defaults = {**(defaults or {}), **RecorderConfig.recipe_defaults(kind)}
    ap.add_argument("--accept-eula", action="store_true",
                    help="set OMNI_KIT_ACCEPT_EULA=YES for the simulator workers")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the planned chunks and worker commands, spawn nothing")
    campaign.CampaignConfig.add_args(ap, defaults=_campaign_defaults(kind))
    RecorderConfig.add_args(ap, defaults=defaults)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Generate demonstration episodes with parallel simulator workers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    s = sub.add_parser("strict", help="expert episodes (one grasp per seed)",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    s.add_argument("--seeds", default="", help="explicit comma-separated seed list")
    s.add_argument("--seed-start", type=int, default=200000, help="first fresh seed")
    s.add_argument("--n-seeds", type=int, default=0,
                   help="record this many consecutive seeds from --seed-start")
    s.add_argument("--quota", type=int, default=0,
                   help="quota mode: kept episodes wanted per grasp-height bin (0 = off)")
    s.add_argument("--bins", type=int, default=4, help="number of grasp-height bins")
    s.add_argument("--z-range", type=float, nargs=2, default=(-0.05, 0.05),
                   metavar=("Z_LO", "Z_HI"), help="grasp-height range along the tool axis (m)")
    s.add_argument("--max-seeds", type=int, default=0,
                   help="quota mode: stop after this many fresh seeds (0 = unlimited)")
    s.add_argument("--fill-bin", type=int, default=None,
                   help="gap fill: restrict the grasp height to this bin of --bins/--z-range")
    s.add_argument("--stored-winners", default=None,
                   help="episode root whose ep_<seed>/meta.npz grasps are reused")
    _common(s, "strict")

    r = sub.add_parser("recovery", help="perturb-and-recover episodes",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    r.add_argument("--strict-json", required=True, help="strict selection JSON (seeds)")
    r.add_argument("--winners", required=True, help="strict episode root with the stored grasps")
    r.add_argument("--draws", type=int, default=4, help="perturbation draws per seed")
    r.add_argument("--range", default=None, metavar="A:B",
                   help="virtual index span [A, B) (default: all seeds x draws)")
    r.add_argument("--virt", default="", help="explicit comma-separated virtual indices")
    r.add_argument("--topup", action="store_true",
                   help="plan rounds of fresh draws until every seed has --target-keeps keeps")
    r.add_argument("--target-keeps", type=int, default=4, help="top-up target keeps per seed")
    r.add_argument("--max-rounds", type=int, default=6, help="top-up rounds")
    _common(r, "recovery", defaults=RECOVERY_DEFAULTS)
    return ap


def _env_for(args: argparse.Namespace) -> dict[str, str]:
    env = {}
    if args.accept_eula:
        env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    return env


def _dry_run_strict(ccfg: campaign.CampaignConfig, rcfg: RecorderConfig, seeds: list[int],
                    stored_winners: str | None) -> None:
    for ch in campaign.chunked(seeds, ccfg.chunk):
        job = campaign.strict_job(ccfg.out, ch, rcfg.to_cli_args(), stored_winners, f"strict_{ch[0]}")
        print(" ".join(["<sim_python>"] + campaign.resolve_cmd(job)))


def run_strict(args: argparse.Namespace) -> int:
    rcfg = RecorderConfig.from_args(args)
    if args.fill_bin is not None:
        edges = assemble.bin_edges(args.z_range[0], args.z_range[1], args.bins)
        if not 0 <= args.fill_bin < args.bins:
            raise SystemExit(f"--fill-bin must be in [0, {args.bins})")
        rcfg.z_lo, rcfg.z_hi, rcfg.z_strict = edges[args.fill_bin], edges[args.fill_bin + 1], True
    ccfg = campaign.CampaignConfig.from_args(args, args.out, _env_for(args))
    os.makedirs(args.out, exist_ok=True)
    if args.quota > 0:
        if args.dry_run:
            print(f"quota mode: {args.quota} x {args.bins} bins over {args.z_range}, "
                  f"seeds from {args.seed_start}, recipe {rcfg.to_dict()}")
            return 0
        summary = campaign.run_strict_quota(ccfg, rcfg, args.seed_start, args.quota, args.bins,
                                            args.z_range[0], args.z_range[1], args.max_seeds)
    else:
        seeds = parse_seed_list(args.seeds) or [args.seed_start + i for i in range(args.n_seeds)]
        if not seeds:
            raise SystemExit("pass --seeds, --seed-start/--n-seeds, or --quota")
        if args.dry_run:
            _dry_run_strict(ccfg, rcfg, seeds, args.stored_winners)
            return 0
        summary = campaign.run_strict_list(ccfg, rcfg, seeds, args.stored_winners)
    print(json.dumps({"mode": "strict", **summary}))
    return 0


def run_recovery(args: argparse.Namespace) -> int:
    rcfg = RecorderConfig.from_args(args)
    ccfg = campaign.CampaignConfig.from_args(args, args.out, _env_for(args))
    os.makedirs(args.out, exist_ok=True)
    with open(args.strict_json) as f:
        seeds_sorted = sorted(int(s) for s in json.load(f)["seeds"])
    if args.topup:
        if args.dry_run:
            plan, stats = campaign.plan_topup(args.out, seeds_sorted, args.draws, args.target_keeps)
            print(json.dumps({"plan": plan[:50], "n": len(plan), **stats}))
            return 0
        summary = campaign.run_recovery_topup(ccfg, rcfg, args.strict_json, args.winners,
                                              args.draws, args.target_keeps, args.max_rounds)
    else:
        if args.virt:
            virts = parse_seed_list(args.virt)
        else:
            a, b = 0, len(seeds_sorted) * args.draws
            if args.range:
                a, b = (int(x) for x in args.range.split(":"))
            virts = list(range(a, min(b, len(seeds_sorted) * args.draws)))
        if args.dry_run:
            for ch in campaign.chunked(virts, ccfg.chunk):
                job = campaign.recovery_job(args.out, ch, rcfg.to_cli_args(), args.strict_json,
                                            args.winners, args.draws, seeds_sorted, f"rec_{ch[0]}")
                print(" ".join(["<sim_python>"] + campaign.resolve_cmd(job)))
            return 0
        summary = campaign.run_recovery_list(ccfg, rcfg, virts, args.strict_json, args.winners,
                                             args.draws)
    print(json.dumps({"mode": "recovery", **summary}))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    return run_strict(args) if args.mode == "strict" else run_recovery(args)


if __name__ == "__main__":
    raise SystemExit(main())
