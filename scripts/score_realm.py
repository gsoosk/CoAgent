#!/usr/bin/env python3
"""Score the REALM-Bench P3 runs: isolation, speed, cost and REALM's metrics.

    python scripts/score_realm.py            # reads results/realm/parts/*.jsonl

Per instance and protocol: anomaly-free trials (``consistent_dispatch``),
speed and cost against the instance's serial median, aborts and
notifications, and REALM-Bench's own plan-quality metrics as recorded by the
runner (``realm_eval``). Mean per-call model latency is printed too: speed
ratios are only comparable between protocols whose calls were equally fast.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["serial", "naive", "2PL-saga", "MTPO", "chopping-prio"]
REALM_KEYS = ["goal_satisfaction", "constraint_satisfaction", "travel_min",
              "double_dispatch", "record_mismatch", "unclaimed_routed",
              "orphan_claims"]


def load(paths: List[Path]) -> List[Dict[str, Any]]:
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows


def count(r: Dict[str, Any], needle: str) -> int:
    return sum(needle in t for t in r["trace"])


def call_latency(r: Dict[str, Any]) -> float:
    durs = [e["dur"] for e in r.get("events", []) if e.get("dur")]
    return st.mean(durs) if durs else float("nan")


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by = defaultdict(list)
    for r in rows:
        by[(r["cell"], r["protocol"])].append(r)
    out: Dict[str, Any] = {}
    for cell in sorted({r["cell"] for r in rows}):
        serial = by.get((cell, "serial"))
        s_wall = st.median(r["wall"] for r in serial) if serial else None
        s_usd = st.median(r["usd"] for r in serial) if serial else None
        out[cell] = {}
        for proto in ORDER:
            rs = by.get((cell, proto))
            if not rs:
                continue
            wall = st.median(r["wall"] for r in rs)
            out[cell][proto] = {
                "n": len(rs),
                "af": sum(bool(r.get("anomaly_free")) for r in rs),
                "speedup": s_wall / wall if s_wall else None,
                "cost": (st.median(r["usd"] for r in rs) / s_usd
                         if s_usd else None),
                "usd": sum(r["usd"] for r in rs),
                "rounds": st.mean(r["rounds"] for r in rs),
                "deadlocks": st.mean(count(r, "DEADLOCK") for r in rs),
                "wounds": st.mean(count(r, "WOUND") for r in rs),
                "notifications": st.mean(r["notifications"] for r in rs),
                "latency": st.mean(call_latency(r) for r in rs),
                **{k: st.mean(r["realm"][k] for r in rs) for k in REALM_KEYS},
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default=str(ROOT / "results/realm/parts"))
    ap.add_argument("--json", default=str(ROOT / "results/realm/summary.json"))
    args = ap.parse_args()
    rows = load(sorted(Path(args.parts).glob("*.jsonl")))
    summary = summarise(rows)
    Path(args.json).write_text(json.dumps(summary, indent=1))

    cols = [("af", "AF", lambda v, d: f"{v}/{d['n']}"),
            ("speedup", "speed", lambda v, d: f"{v:.2f}"),
            ("cost", "cost", lambda v, d: f"{v:.2f}"),
            ("deadlocks", "dlock", lambda v, d: f"{v:.1f}"),
            ("wounds", "wound", lambda v, d: f"{v:.1f}"),
            ("notifications", "notif", lambda v, d: f"{v:.1f}"),
            ("latency", "s/call", lambda v, d: f"{v:.2f}"),
            ("goal_satisfaction", "served%", lambda v, d: f"{v:.0f}"),
            ("constraint_satisfaction", "constr%", lambda v, d: f"{v:.0f}"),
            ("double_dispatch", "dbl", lambda v, d: f"{v:.1f}"),
            ("record_mismatch", "mism", lambda v, d: f"{v:.1f}"),
            ("unclaimed_routed", "unclm", lambda v, d: f"{v:.1f}"),
            ("travel_min", "min", lambda v, d: f"{v:.0f}")]
    head = f"{'cell':14}{'protocol':15}" + "".join(f"{c[1]:>9}" for c in cols)
    print(head)
    for cell, ps in summary.items():
        for proto, d in ps.items():
            print(f"{cell:14}{proto:15}" + "".join(
                f"{(f(d[k], d) if d[k] is not None else '-'):>9}"
                for k, _, f in cols))
        print()

    # pooled over instances: mean of per-instance figures
    print("mean over instances")
    for proto in ORDER:
        ds = [ps[proto] for ps in summary.values() if proto in ps]
        if not ds:
            continue
        af = sum(d["af"] for d in ds)
        n = sum(d["n"] for d in ds)
        print(f"  {proto:15} AF {af:2}/{n:<3} speed "
              f"{st.mean(d['speedup'] for d in ds):.2f}  cost "
              f"{st.mean(d['cost'] for d in ds):.2f}  served "
              f"{st.mean(d['goal_satisfaction'] for d in ds):.0f}%  "
              f"constraints {st.mean(d['constraint_satisfaction'] for d in ds):.0f}%  "
              f"s/call {st.mean(d['latency'] for d in ds):.2f}")
    print(f"\ntotal spend ${sum(r['usd'] for r in rows):.2f} over "
          f"{len(rows)} trials -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
