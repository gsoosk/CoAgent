#!/usr/bin/env python3
"""Where the time goes in the REALM runs: useful work, wasted work, overlap.

For each trial, from the per-call events and the protocol trace:

  work      seconds of model calls, all attempts included
  wasted    seconds of model calls in attempts that were later aborted
            (a deadlock victim or a wounded agent restarts from scratch)
  blocked   seconds agents spent waiting on a lock
  overlap   work / wall: 1.0 is one call at a time, 3.0 all three at once

    python scripts/realm_mechanism.py results/realm/parts
"""

from __future__ import annotations

import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ORDER = ["serial", "naive", "2PL-saga", "MTPO", "chopping-prio"]
STAMP = re.compile(r"\[\s*([\d.]+)s\] (\S+) (blocked on|ABORT)")


def attempts(r: Dict[str, Any]) -> Dict[str, List[float]]:
    """Each agent's abort times, in order."""
    out: Dict[str, List[float]] = defaultdict(list)
    for line in r["trace"]:
        m = STAMP.match(line)
        if m and m.group(3) == "ABORT":
            out[m.group(2)].append(float(m.group(1)))
    return out


def blocked_seconds(r: Dict[str, Any]) -> float:
    """A block lasts until the agent's next call starts."""
    starts: Dict[str, List[float]] = defaultdict(list)
    for e in r["events"]:
        starts[e["agent"]].append(e["t"])
    total = 0.0
    for line in r["trace"]:
        m = STAMP.match(line)
        if m and m.group(3) == "blocked on":
            t, who = float(m.group(1)), m.group(2)
            nxt = [s for s in starts[who] if s >= t]
            if nxt:
                total += min(nxt) - t
    return total


def breakdown(r: Dict[str, Any]) -> Dict[str, float]:
    aborts = attempts(r)
    work = wasted = 0.0
    for e in r["events"]:
        work += e["dur"]
        # a call is wasted if its agent aborted after it started
        if any(a >= e["t"] for a in aborts.get(e["agent"], [])):
            wasted += e["dur"]
    return {"wall": r["wall"], "work": work, "wasted": wasted,
            "blocked": blocked_seconds(r), "overlap": work / r["wall"],
            "aborts": sum(len(v) for v in aborts.values()),
            "calls": len(r["events"])}


def main() -> int:
    parts = Path(sys.argv[1] if len(sys.argv) > 1 else "results/realm/parts")
    rows = [json.loads(l) for p in sorted(parts.glob("*.jsonl"))
            for l in p.read_text().splitlines() if l.strip()]
    by = defaultdict(list)
    for r in rows:
        by[r["protocol"]].append(breakdown(r))
    print(f"{'protocol':15}{'wall s':>8}{'work s':>8}{'wasted':>8}"
          f"{'blocked':>9}{'overlap':>9}{'calls':>7}{'aborts':>8}")
    for p in ORDER:
        bs = by.get(p)
        if not bs:
            continue
        m = {k: st.mean(b[k] for b in bs) for k in bs[0]}
        print(f"{p:15}{m['wall']:8.0f}{m['work']:8.0f}"
              f"{100 * m['wasted'] / m['work']:7.0f}%{m['blocked']:9.0f}"
              f"{m['overlap']:9.2f}{m['calls']:7.1f}{m['aborts']:8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
