#!/usr/bin/env python3
"""Section 7.3 / Figure 6: one contended cell under every protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coagent.protocols import PROTOCOL_ORDER                       # noqa: E402
from coagent.runner import TrialConfig, reference_states, run_trial  # noqa: E402
from coagent.workloads import CELLS                                # noqa: E402

PAPER = {"serial": 50.2, "naive": 31.8, "2PL-saga": 53.1,
         "OCC-saga": 63.7, "MTPO": 32.8}
PAPER_OK = {"serial": True, "naive": False, "2PL-saga": True,
            "OCC-saga": True, "MTPO": True}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="k8s-canary")
    ap.add_argument("--trace", action="store_true", help="print the MTPO trace")
    args = ap.parse_args()

    cell = CELLS[args.cell]
    refs = reference_states(cell)
    # deterministic: the case study is one run, not a distribution
    cfg = TrialConfig(p_misjudge=0.0, p_task_failure=0.0, jitter=0.0,
                      len_sigma=0.0)
    print(f"cell: {cell.name}  ({cell.suite})")
    print(f"anomaly: {cell.anomaly}\n{cell.description}\n")

    base = run_trial(cell, "serial", 1, cfg, refs)
    head = (f"{'protocol':12}{'wall':>8}{'vs serial':>11}{'tokens':>9}"
            f"{'vs serial':>11}  {'ok':<6}{'events':<26}{'paper':>8}")
    print(head)
    print("-" * len(head))
    for proto in PROTOCOL_ORDER:
        r = run_trial(cell, proto, 1, cfg, refs, keep_trace=args.trace)
        ev = (f"dl={r.deadlocks} ab={r.aborts} "
              f"notif={r.notifications} rep={r.repairs}")
        mark = "PASS" if r.invariant_ok else "FAIL"
        agree = "" if r.invariant_ok == PAPER_OK[proto] else "  <-- differs"
        print(f"{proto:12}{r.wall_clock:7.1f}s{base.wall_clock / r.wall_clock:11.2f}"
              f"{r.tokens:9d}{r.tokens / base.tokens:11.2f}  {mark:<6}{ev:<26}"
              f"{PAPER[proto]:7.1f}s{agree}")
        if args.trace and proto == "MTPO":
            print()
            for line in r.note.split("; "):
                print("      " + line)
            print()
    print("\npaper Figure 6: naive fails; 2PL pays a deadlock, OCC a full "
          "redo,\nMTPO one notification and a localized repair.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
