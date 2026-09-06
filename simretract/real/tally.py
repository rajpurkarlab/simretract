"""Tally real-robot runs per served checkpoint.

Reads ``<root>/<model>/run_*/result.json`` written by the client and prints
runs, gripper-close rate, aborts and the median close tick per model. Note
that ``closed`` records the gripper-close event only; whether the tool was
actually held or taken over is an operator judgement outside this file.

    python -m simretract.real.tally [--root <eval>/real]
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def tally(root: str) -> list[dict]:
    """Per-model summary rows for every model directory under ``root`` with results."""
    rows = []
    for md in sorted(d for d in glob.glob(os.path.join(root, "*"))
                     if os.path.isdir(d) and not os.path.basename(d).startswith("run_")):
        rs = []
        for f in sorted(glob.glob(os.path.join(md, "run_*", "result.json"))):
            with open(f) as fh:
                rs.append(json.load(fh))
        if not rs:
            continue
        n = len(rs)
        c = sum(bool(r.get("closed")) for r in rs)
        ab = sum(bool(r.get("aborted")) for r in rs)
        ct = sorted(r["closed_tick"] for r in rs if r.get("closed") and r.get("closed_tick") is not None)
        rows.append({"model": os.path.basename(md), "runs": n, "closed": c, "aborted": ab,
                     "median_close_tick": ct[len(ct) // 2] if ct else None})
    return rows


def format_rows(rows: list[dict]) -> str:
    lines = []
    for r in rows:
        mct = r["median_close_tick"]
        lines.append(f"{r['model']:32s} runs {r['runs']:3d}  closed {r['closed']:3d} "
                     f"({100 * r['closed'] / r['runs']:.0f}%)  aborted {r['aborted']:3d}  "
                     f"median close tick {mct if mct is not None else '-'}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m simretract.real.tally",
                                 description="Per-model tally of real-run result.json files.")
    ap.add_argument("--root", default=None, help="run root (default <eval>/real from configs/paths.yaml)")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.root is None:
        from simretract.real import default_runs_root
        a.root = str(default_runs_root())
    rows = tally(a.root)
    print(format_rows(rows) if rows else f"no result.json under {a.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
