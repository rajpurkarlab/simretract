"""Funnel scorer for simulation-benchmark result files.

An evaluation directory holds one ``result_<seed>.json`` per seed (earlier
attempts are kept as ``result_<seed>.try<k>.json`` and ignored here).  The
funnel counts, over the seeds of a selection file:

* grabbed   -- ``outcome == "completed"`` (the gripper closed on the tool)
* secured   -- grabbed and ``drop_mm <= 20`` (the tool stayed in the gripper
               when the operator hand released; the evaluator marks a larger
               drop ``verdict.note == "dropped_on_release"``)
* held      -- ``held`` (secured and the load battery did not pull the tool
               out of the gripper)
* kept      -- held and ``verdict.grasp_t_mm < 10`` and
               ``verdict.grasp_r_deg < 10`` (pose preserved through the
               hand-over)
* timeouts  -- ``outcome == "timeout"``

Offset (``grasp_t_mm``) and twist (``grasp_twist_deg``) are summarised as
mean +- sd [median] over completed rows that carry the verdict field.

CLI: ``python -m simretract.eval.score <eval_dir> --seeds <selection.json>``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Iterable

__all__ = ["Funnel", "Stats", "funnel", "load_results", "load_seeds",
           "row_line", "table_md", "main"]

#: Largest release drop (mm) that still counts as "secured".
DROP_MM_SECURED = 20.0
#: Pose-preservation thresholds for "kept".
KEPT_T_MM = 10.0
KEPT_R_DEG = 10.0


@dataclass
class Stats:
    """mean +- sd [median] of a metric over ``n`` rows (sd is the sample
    standard deviation; NaN when fewer than two values)."""

    n: int = 0
    mean: float = float("nan")
    sd: float = float("nan")
    median: float = float("nan")

    @classmethod
    def of(cls, values: Iterable[float]) -> "Stats":
        v = [float(x) for x in values]
        if not v:
            return cls()
        return cls(n=len(v), mean=statistics.fmean(v),
                   sd=statistics.stdev(v) if len(v) > 1 else float("nan"),
                   median=statistics.median(v))

    def fmt(self, unit: str = "") -> str:
        if self.n == 0:
            return "n/a"
        return f"{self.mean:.1f} +- {self.sd:.1f} [{self.median:.1f}]{unit}"


@dataclass
class Funnel:
    """Funnel counts and metric summaries over one seed set."""

    n_seeds: int
    n_results: int
    grabbed: int
    secured: int
    held: int
    kept: int
    timeouts: int
    outcomes: dict[str, int]
    missing: list[int]
    offset_mm: Stats = field(default_factory=Stats)
    twist_deg: Stats = field(default_factory=Stats)
    rot_deg: Stats = field(default_factory=Stats)

    def to_dict(self) -> dict:
        return asdict(self)


def load_seeds(path: str | os.PathLike[str]) -> tuple[list[int], str]:
    """Return (seeds, label) from a selection json; the label is the file's
    ``name`` without the ``new_xz_`` prefix (``core50`` for the paper set)."""
    with open(path) as f:
        sel = json.load(f)
    seeds = [int(s) for s in sel["seeds"]]
    label = str(sel.get("name", os.path.splitext(os.path.basename(path))[0]))
    return seeds, label.replace("new_xz_", "")


def load_results(eval_dir: str | os.PathLike[str]) -> dict[int, dict]:
    """Current result per seed (``result_<seed>.json``; ``.try<k>`` attempts
    and perturbation-variant files are skipped)."""
    out: dict[int, dict] = {}
    for f in sorted(glob.glob(os.path.join(str(eval_dir), "result_*.json"))):
        base = os.path.basename(f)
        stem = base[len("result_"):-len(".json")]
        if ".try" in base or not stem.isdigit():
            continue
        try:
            with open(f) as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            continue
        if "seed" in d:
            out[int(d["seed"])] = d
    return out


def _verdict(d: dict, key: str):
    return (d.get("verdict") or {}).get(key)


def funnel(results: dict[int, dict], seeds: Iterable[int]) -> Funnel:
    """Compute the funnel over ``seeds`` (seeds without a result count as
    missing and as non-grabbed)."""
    seeds = [int(s) for s in seeds]
    rows = [results[s] for s in seeds if s in results]
    missing = [s for s in seeds if s not in results]
    grabbed = [d for d in rows if d.get("outcome") == "completed"]
    secured = [d for d in grabbed
               if d.get("drop_mm") is not None and float(d["drop_mm"]) <= DROP_MM_SECURED]
    held = [d for d in grabbed if d.get("held")]
    kept = [d for d in held
            if (_verdict(d, "grasp_t_mm") if _verdict(d, "grasp_t_mm") is not None else 99) < KEPT_T_MM
            and (_verdict(d, "grasp_r_deg") if _verdict(d, "grasp_r_deg") is not None else 99) < KEPT_R_DEG]
    t = [_verdict(d, "grasp_t_mm") for d in grabbed if _verdict(d, "grasp_t_mm") is not None]
    w = [_verdict(d, "grasp_twist_deg") for d in grabbed if _verdict(d, "grasp_twist_deg") is not None]
    r = [_verdict(d, "grasp_r_deg") for d in grabbed if _verdict(d, "grasp_r_deg") is not None]
    outcomes = Counter(str(d.get("outcome")) for d in rows)
    return Funnel(
        n_seeds=len(seeds), n_results=len(rows), grabbed=len(grabbed),
        secured=len(secured), held=len(held), kept=len(kept),
        timeouts=outcomes.get("timeout", 0), outcomes=dict(sorted(outcomes.items())),
        missing=missing, offset_mm=Stats.of(t), twist_deg=Stats.of(w),
        rot_deg=Stats.of(r))


def row_line(name: str, label: str, f: Funnel, exec_h: int | None = None,
             passes: int | None = None) -> str:
    """One-line summary in the driver's ROW format."""
    tag = f" {passes}-pass" if passes else ""
    eh = f" EXEC_H={exec_h}" if exec_h is not None else ""
    med = f"{f.offset_mm.median:.1f}" if f.offset_mm.n else "nan"
    tw = f"{f.twist_deg.median:.1f}" if f.twist_deg.n else "nan"
    return (f"ROW {name} {label}{tag}{eh}: grabbed {f.grabbed}/{f.n_seeds} | "
            f"secured {f.secured} | held {f.held} | kept {f.kept} | "
            f"med offset {med} mm | med twist {tw} deg | "
            f"outcomes {f.outcomes} | n={f.n_results}")


def table_md(name: str, label: str, f: Funnel, exec_h: int | None = None) -> str:
    """Markdown table: funnel counts and metric summaries."""
    lines = [
        f"# {name}",
        "",
        f"Seed set: {label} ({f.n_seeds} seeds, {f.n_results} results"
        + (f", missing {f.missing}" if f.missing else "") + ")"
        + (f"; exec-h {exec_h}" if exec_h is not None else ""),
        "",
        "| grabbed | secured | held | kept | timeouts | offset mm (mean +- sd [med]) | twist deg (mean +- sd [med]) |",
        "|---|---|---|---|---|---|---|",
        f"| {f.grabbed}/{f.n_seeds} | {f.secured} | {f.held} | {f.kept} | {f.timeouts} "
        f"| {f.offset_mm.fmt()} | {f.twist_deg.fmt()} |",
        "",
        "Outcomes: " + ", ".join(f"{k} {v}" for k, v in f.outcomes.items()) if f.outcomes else "Outcomes: none",
        "",
    ]
    return "\n".join(lines)


def score_dir(eval_dir: str | os.PathLike[str], seeds_json: str | os.PathLike[str]) -> tuple[Funnel, str]:
    """Convenience: results in ``eval_dir`` scored over ``seeds_json``."""
    seeds, label = load_seeds(seeds_json)
    return funnel(load_results(eval_dir), seeds), label


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="simretract.eval.score",
        description="Score result_<seed>.json files over a seed selection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("eval_dir", help="directory with result_<seed>.json files")
    p.add_argument("--seeds", required=True, help="selection json with a 'seeds' list")
    p.add_argument("--name", default=None, help="row name (default: eval_dir basename)")
    p.add_argument("--exec-h", type=int, default=None, help="annotate the ROW line")
    p.add_argument("--passes", type=int, default=None, help="annotate the ROW line")
    p.add_argument("--json", default=None, help="write the funnel as json here")
    p.add_argument("--md", default=None, help="write the markdown table here")
    a = p.parse_args(argv)
    name = a.name or os.path.basename(os.path.normpath(a.eval_dir))
    f, label = score_dir(a.eval_dir, a.seeds)
    print(row_line(name, label, f, a.exec_h, a.passes))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"name": name, "label": label, "exec_h": a.exec_h,
                       "passes": a.passes, "funnel": f.to_dict()}, fh, indent=1)
    if a.md:
        with open(a.md, "w") as fh:
            fh.write(table_md(name, label, f, a.exec_h))
    return 0


if __name__ == "__main__":
    sys.exit(main())
