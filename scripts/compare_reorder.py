#!/usr/bin/env python3
"""Score the follow-ups to the anatomy experiment against the Haiku matrix.

Three variants (see protocols/chopping_tight.py):

  chopping-min         merge only the pieces that carry the conflict
  chopping-reorder     chopping-min, after asking each agent for an order of
                       its steps that puts the contended ones together
  chopping-2pl-intent  chopping-2pl, taking the declared write lock at the
                       first touch

Baselines for the twelve benchmark cells come from results/haiku; the new
trials and the three reorder cells are in results/reorder.

Same scoring as the matrix: the anomaly_free predicate, medians per cell,
speedup against the cell's serial median.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coagent.runner import reference_states                    # noqa: E402
from coagent.workloads import CELLS as BENCH                    # noqa: E402
from coagent.workloads.reorder_cells import REORDER_CELLS       # noqa: E402

CELLS = {**BENCH, **REORDER_CELLS}
ORDER = ["serial", "2PL-saga", "2PL-prio", "MTPO", "chopping", "chopping-min",
         "chopping-reorder", "chopping-reorder-free", "chopping-2pl", "chopping-2pl-intent",
         "chopping-prio"]


def load(paths: List[Path]) -> List[Dict[str, Any]]:
    rows = []
    for p in paths:
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "results/haiku/llm_runs.jsonl"))
    ap.add_argument("--new", default=str(ROOT / "results/reorder"))
    ap.add_argument("--json", default=str(ROOT / "results/reorder/summary.json"))
    args = ap.parse_args()

    rows = load([Path(args.base)] + sorted(Path(args.new).glob("**/*.jsonl")))
    refs = {c: reference_states(CELLS[c]) for c in {r["cell"] for r in rows}}

    def af(r: Dict[str, Any]) -> bool:
        cell = CELLS[r["cell"]]
        return bool(cell.anomaly_free(r["final_state"], refs[r["cell"]]))

    by: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(r["cell"], r["protocol"])].append(r)

    def count(r: Dict[str, Any], needle: str) -> int:
        return sum(needle in t for t in r["trace"])

    summary: Dict[str, Dict[str, Any]] = {}
    for cell in CELLS:
        serial = by.get((cell, "serial"))
        if not serial:
            continue
        s_wall = statistics.median(r["wall"] for r in serial)
        s_usd = statistics.median(r["usd"] for r in serial)
        scored = all(af(r) for r in serial)
        summary[cell] = {"scored": scored, "protocols": {}}
        for proto in ORDER:
            rs = by.get((cell, proto))
            if not rs:
                continue
            neg = [r["negotiation"]["agents"] for r in rs
                   if r.get("negotiation")]
            asked = [a for n in neg for a in n.values()]
            summary[cell]["protocols"][proto] = {
                "n": len(rs),
                "af": sum(af(r) for r in rs),
                "speedup": s_wall / statistics.median(r["wall"] for r in rs),
                "cost": statistics.median(r["usd"] for r in rs) / s_usd,
                "deadlocks": sum(count(r, "DEADLOCK") for r in rs) / len(rs),
                "wounds": sum(count(r, "WOUND") for r in rs) / len(rs),
                "refused": sum(count(r, "REFUSED") for r in rs) / len(rs),
                "pieces": (statistics.mean(
                    sum(len(v) for v in r["plan"].values())
                    for r in rs if r.get("plan"))
                    if any(r.get("plan") for r in rs) else None),
                "asked": len(asked),
                "reordered": sum(a["reordered"] for a in asked),
                "useful_asks": sum(a["gap_inside_span"] for a in asked),
                "useful_reordered": sum(a["reordered"] for a in asked
                                        if a["gap_inside_span"]),
                "invalid": sum(not a["valid"] for a in asked),
                "plan_latency": (statistics.median(a["latency"] for a in asked)
                                 if asked else None),
            }

    Path(args.json).write_text(json.dumps(summary, indent=1))

    protos = [p for p in ORDER
              if any(p in c["protocols"] for c in summary.values())]
    for title, key, fmt in (
        ("anomaly-free trials", "af", lambda v, d: f"{v}/{d['n']}"),
        ("speedup over serial (median wall)", "speedup",
         lambda v, d: f"{v:.2f}"),
        ("cost relative to serial", "cost", lambda v, d: f"{v:.2f}"),
        ("deadlock aborts per trial", "deadlocks", lambda v, d: f"{v:.1f}"),
        ("pieces in the plan that ran", "pieces",
         lambda v, d: "-" if v is None else f"{v:.1f}"),
    ):
        print(f"\n== {title}")
        print(f"{'cell':24}" + "".join(f"{p[-11:]:>12}" for p in protos))
        for cell, c in summary.items():
            mark = "" if c["scored"] else " *"
            print(f"{cell + mark:24}" + "".join(
                f"{fmt(c['protocols'][p][key], c['protocols'][p]):>12}"
                if p in c["protocols"] else f"{'':>12}" for p in protos))
    print("\n* serial itself fails the predicate on this cell with this "
          "model, so its correctness column is not a protocol signal")

    print("\n== order negotiation (chopping-reorder)")
    print(f"{'cell':24}{'asked':>7}{'reordered':>11}{'could help':>12}"
          f"{'did, where it could':>21}{'invalid':>9}{'plan s':>8}")
    for cell, c in summary.items():
        d = c["protocols"].get("chopping-reorder")
        if not d:
            continue
        lat = "-" if d["plan_latency"] is None else f"{d['plan_latency']:.1f}"
        print(f"{cell:24}{d['asked']:>7}{d['reordered']:>11}"
              f"{d['useful_asks']:>12}{d['useful_reordered']:>21}"
              f"{d['invalid']:>9}{lat:>8}")
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
