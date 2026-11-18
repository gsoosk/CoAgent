#!/usr/bin/env python3
"""Score REALM-Bench runs: isolation, speed, cost and REALM's metrics.

    python scripts/score_realm.py                                  # P3
    python scripts/score_realm.py --parts results/realm_jssp/parts \
        --json results/realm_jssp/summary.json                      # J1

Per instance and protocol: anomaly-free trials (the cell's isolation
predicate), speed and cost against the instance's serial median, aborts and
notifications, the causes of each inconsistency, and REALM-Bench's own
plan-quality metrics as recorded by the runner (``Cell.quality``). Mean
per-call model latency is printed too: speed ratios are only comparable
between protocols whose calls were equally fast.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ORDER = ["serial", "naive", "2PL-saga", "MTPO", "chopping-prio"]


def load(paths: List[Path]) -> List[Dict[str, Any]]:
    """Trials, with REALM's metrics recomputed from each final state.

    Recomputing rather than trusting the recorded metrics keeps every trial
    scored by the same code, whatever version the runner had loaded.
    """
    from coagent.workloads.realm_jssp import JSSP_CELLS
    from coagent.workloads.realm_urs import REALM_CELLS
    from coagent.workloads.realm_wedding import WEDDING_CELLS
    from coagent.workloads.realm_tours import TOURS_CELLS
    from coagent.workloads.realm_thanksgiving import THANKS_CELLS
    cells = {**REALM_CELLS, **JSSP_CELLS, **WEDDING_CELLS, **TOURS_CELLS,
             **THANKS_CELLS}
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    for r in rows:
        cell = cells.get(r["cell"])
        if cell is not None and cell.quality is not None:
            r["realm"] = cell.quality(r["final_state"])
            r["anomaly_free"] = bool(cell.anomaly_free(r["final_state"], []))
    return rows


# most serious first: interference over agent error
CAUSES = ["double claim", "double booking", "failed heal", "wrong release",
          "ignored booking", "routing error"]


def _vehicles(cell: str) -> List[str]:
    from coagent.workloads.realm_urs import REALM_CELLS, load_instance
    inst = load_instance(REALM_CELLS[cell].llm_spec["instance"])
    return [v["vehicle_id"] for v in inst["vehicles"]]


def failure_causes(r: Dict[str, Any]) -> List[str]:
    """Why each inconsistent passenger is inconsistent, from the tool calls.

    routing error  a vehicle routes a passenger it never claimed
    wrong release  a vehicle that does not route the passenger set its claim
                   to null, freeing it for a second claimant. Serially that
                   is a model bug; under MTPO it is also the failed heal --
                   releasing where it had to hand the claim back
    double claim   two vehicles both claimed and routed the passenger

    Serial's rate of each cause is the agent-error baseline; what a protocol
    shows beyond it is interference. Claims made in an attempt a saga later
    undid still count as claims, so an aborted agent is never blamed for
    routing a passenger it claimed before the abort.
    """
    from coagent.workloads.realm_urs import REQ, ROUTE, route_passengers
    vids = _vehicles(r["cell"])
    claimed: Dict[str, set] = defaultdict(set)    # passenger -> claimants
    nulled: Dict[str, set] = defaultdict(set)     # passenger -> who nulled
    for i, turns in enumerate(r["transcript"]):
        me = vids[i] if i < len(vids) else f"agent-{i}"
        for t in turns:
            rep = t.get("reply") or {}
            if rep.get("tool") != "assign_request":
                continue
            args = rep.get("args") or {}
            pid = str(args.get("obj", "")).split("/")[-1]
            if args.get("vehicle") == me:
                claimed[pid].add(me)
            elif args.get("vehicle") is None:
                nulled[pid].add(me)
    final = r["final_state"]
    routed: Dict[str, List[str]] = defaultdict(list)
    for v in vids:
        for pid in route_passengers(final.get(ROUTE + v)):
            routed[pid].append(v)
    causes = []
    for pid, vs in routed.items():
        owner = (final.get(REQ + pid) or {}).get("assigned")
        if len(vs) < 2 and (owner is None or owner in vs):
            continue                                  # consistent
        if any(v not in claimed[pid] for v in vs):
            causes.append("routing error")
        elif nulled[pid] - set(vs):
            causes.append("wrong release")
        else:
            causes.append("double claim")
    return causes


def _granted(r: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Each agent's calls at the time they ran, not when they were issued.

    The runtime charges a call's latency once it is granted, so a call that
    blocked ran from (the agent's next issue time - its duration).
    """
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in r["events"]:
        by[e["agent"]].append(e)
    out = {}
    for a, evs in by.items():
        evs = sorted(evs, key=lambda e: e["t"])
        out[a] = [dict(e, t=max(e["t"], evs[i + 1]["t"] - e["dur"])
                       if i + 1 < len(evs) else e["t"])
                  for i, e in enumerate(evs)]
    return out


def _overlaps(r: Dict[str, Any]):
    """(object, agent, agent) for every overlapping pair of bookings."""
    from coagent.workloads.realm_jssp import MACH, N_MACHINES, _slots, _span
    from coagent.workloads.realm_wedding import CAR, CARS, _windows, errands
    final, out = r["final_state"], []
    if r["cell"].startswith("realm-p2-tours"):
        from coagent.workloads.realm_tours import TOURS_CELLS
        return [(o, f"agent-{a}", f"agent-{b}") for o, a, b in
                TOURS_CELLS[r["cell"]].llm_spec["conflicts"](final)]
    if r["cell"].endswith("thanksgiving"):
        from coagent.workloads.realm_thanksgiving import conflicts
        return [(o, f"agent-{a}", f"agent-{b}") for o, a, b in conflicts(final)]
    if r["cell"].startswith("realm-jssp"):
        for m in range(N_MACHINES):
            items = [(k, _span(v)) for k, v in _slots(final, m).items()]
            items = [(int(k[1:]), s) for k, s in items
                     if s and k[1:].isdigit()]
            out += [(f"{MACH}m{m}", f"agent-{a}", f"agent-{b}")
                    for (a, x), (b, y) in itertools.combinations(items, 2)
                    if x[0] < y[1] and y[0] < x[1]]
    else:
        index = {e: i for i, e in enumerate(errands_order())}
        for car in CARS:
            items = [(index[e], w) for e, w in _windows(final, car).items()
                     if e in index]
            out += [(CAR + car, f"agent-{a}", f"agent-{b}")
                    for (a, x), (b, y) in itertools.combinations(items, 2)
                    if x[0] < y[1] and y[0] < x[1]]
    return out


def errands_order() -> List[str]:
    from coagent.workloads.realm_wedding import errands, load
    return list(errands(load()))


def booking_causes(r: Dict[str, Any]) -> List[str]:
    """Why each pair of overlapping bookings overlaps (J1 machines, P5 cars,
    P2 guides, P6/P9 people).

    double booking   the later booker last read the machine or car before
                     the earlier booking was written: it acted on a stale
                     read, which is interference
    ignored booking  it read after that booking was written and booked over
                     it anyway: an agent error, which serial shows too
    failed heal      MTPO: the agent later in its order was notified of the
                     earlier booking and did not move (MTPO orders by sigma,
                     not by time, so the time rule above does not apply)
    """
    calls = _granted(r)
    causes = []
    for obj, a, b in _overlaps(r):
        if r["protocol"] == "MTPO":
            hi = max(a, b, key=lambda x: int(x.split("-")[1]))
            told = any(f"->{hi} on {obj}" in line for line in r["trace"])
            causes.append("failed heal" if told else "double booking")
            continue
        def last_write(x):
            ws = [e["t"] for e in calls.get(x, [])
                  if e["kind"] == "write" and obj in e["label"]]
            return max(ws) if ws else -1.0
        early, late = sorted((a, b), key=last_write)
        t_early, t_late = last_write(early), last_write(late)
        reads = [e["t"] for e in calls.get(late, [])
                 if e["kind"] == "read" and e["t"] < t_late
                 and (obj in e["label"] or obj.rsplit("/", 1)[0] + "/"
                      in e["label"])]
        saw = bool(reads) and max(reads) > t_early
        causes.append("ignored booking" if saw else "double booking")
    return causes


def causes_of(r: Dict[str, Any]) -> List[str]:
    """P3 classifies claims from the tool calls; J1 and P5 classify each
    overlapping pair of bookings from the timed calls."""
    if r["cell"].startswith("realm-p3"):
        return failure_causes(r)
    return booking_causes(r)


def primary(r: Dict[str, Any]) -> str:
    """The most serious cause in a trial: interference over agent error."""
    found = set(causes_of(r))
    return next((c for c in CAUSES if c in found), "")


def realm_means(rs: List[Dict[str, Any]]) -> Dict[str, float]:
    """Mean of every numeric REALM metric, over the trials that have it."""
    keys = {k for r in rs for k, v in (r.get("realm") or {}).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}
    out = {}
    for k in sorted(keys):
        vals = [r["realm"][k] for r in rs
                if isinstance(r["realm"].get(k), (int, float))]
        out[k] = st.mean(vals) if vals else None
    return out


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
                # trials whose inconsistency includes each cause
                **{f"cause:{c}": sum(c in causes_of(r) for r in rs)
                   for c in CAUSES},
                # each inconsistent trial once, by its most serious cause
                **{f"primary:{c}": sum(primary(r) == c for r in rs)
                   for c in CAUSES},
                **realm_means(rs),
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
            ("goal_satisfaction", "goal%", lambda v, d: f"{v:.0f}"),
            ("constraint_satisfaction", "constr%", lambda v, d: f"{v:.0f}"),
            ("makespan_optimality", "opt", lambda v, d: f"{v:.2f}"),
            ("travel_min", "min", lambda v, d: f"{v:.0f}")]
    head = f"{'cell':14}{'protocol':15}" + "".join(f"{c[1]:>9}" for c in cols)
    print(head)
    for cell, ps in summary.items():
        for proto, d in ps.items():
            print(f"{cell:14}{proto:15}" + "".join(
                f"{(f(d[k], d) if d.get(k) is not None else '-'):>9}"
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
              f"{st.mean(d['cost'] for d in ds):.2f}  goal "
              f"{st.mean(d['goal_satisfaction'] for d in ds):.0f}%  "
              f"constraints {st.mean(d['constraint_satisfaction'] for d in ds):.0f}%  "
              f"s/call {st.mean(d['latency'] for d in ds):.2f}")
    print("\ntrials with an inconsistency, by cause (pooled over instances)")
    seen = [c for c in CAUSES if any(d.get("cause:" + c)
                                     for ps in summary.values()
                                     for d in ps.values())]
    print(f"  {'protocol':15}{'trials':>7}" + "".join(f"{c:>17}" for c in seen))
    for proto in ORDER:
        ds = [ps[proto] for ps in summary.values() if proto in ps]
        if ds:
            print(f"  {proto:15}{sum(d['n'] for d in ds):>7}" + "".join(
                f"{sum(d['cause:' + c] for d in ds):>17}" for c in seen))

    print(f"\ntotal spend ${sum(r['usd'] for r in rows):.2f} over "
          f"{len(rows)} trials -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
