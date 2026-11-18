"""Score a finished REALM P3 run with REALM-Bench's own metric classes.

``consistent_dispatch`` asks the isolation question. This module asks
REALM's: did the vehicles' plans serve the passengers within the instance's
constraints? Each written route is driven through the instance's travel
times, and the resulting times and loads are handed to REALM-Bench's
``PlanningQualityMetrics``, ``ConstraintSatisfactionMetrics`` and
``PlanningOptimalityMetrics`` (third_party/REALM-Bench/evaluation/metrics.py),
in the input format those classes define.

The module is loaded by path, not as a package: REALM's
``evaluation/__init__.py`` pulls in pandas and matplotlib for its own runner,
which the metrics themselves do not need.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from typing import Any, Dict, List, Optional

from .realm_urs import (REALM_ROOT, REQ, ROUTE, START_HOUR, load_instance,
                        route_passengers, travel)

__all__ = ["simulate", "evaluate"]


@lru_cache(maxsize=1)
def _realm_metrics():
    path = REALM_ROOT / "evaluation" / "metrics.py"
    spec = importlib.util.spec_from_file_location("realm_metrics", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stops(route: Any) -> List[Dict[str, str]]:
    """The route's stops as (passenger, action) pairs, in order."""
    out: List[Dict[str, str]] = []
    raw = (route or {}).get("stops") if isinstance(route, dict) else None
    if raw is None:
        return out
    pids = set(route_passengers(route))       # tolerant parsing lives there
    if isinstance(raw, str):
        import json
        raw = json.loads(raw)
    for s in raw:
        if not isinstance(s, dict):
            continue
        pid = str(s.get("passenger", "")).split("/")[-1]
        act = str(s.get("action", "")).lower()
        if pid in pids and act in ("pickup", "dropoff"):
            out.append({"passenger": pid, "action": act})
    return out


def simulate(inst: Dict[str, Any], vehicle: Dict[str, Any],
             route: Any) -> Dict[str, Any]:
    """Drive one route: when each passenger is picked up and dropped off.

    The vehicle starts at its location at START_HOUR and may wait at a pickup
    for the window to open. A leg with no travel time in the instance, or a
    dropoff before its pickup, ends the drive there: what follows it never
    happens.
    """
    reqs = {r["passenger_id"]: r for r in inst["ride_requests"]}
    t, at, load, max_load, minutes = START_HOUR, vehicle["location"], 0, 0, 0
    picked: Dict[str, float] = {}
    rides: Dict[str, float] = {}
    for s in _stops(route):
        r = reqs.get(s["passenger"])
        if r is None:
            break
        where = r["pickup"] if s["action"] == "pickup" else r["dropoff"]
        leg = travel(inst, at, where)
        if leg is None:
            break
        t, at, minutes = t + leg / 60.0, where, minutes + leg
        if s["action"] == "pickup":
            if s["passenger"] in picked:
                continue
            t = max(t, r["time_window"][0])
            picked[s["passenger"]] = t
            load += 1
            max_load = max(max_load, load)
        else:
            if s["passenger"] not in picked or s["passenger"] in rides:
                break
            rides[s["passenger"]] = (t - picked[s["passenger"]]) * 60.0
            load -= 1
    return {"pickup_h": picked, "ride_min": rides, "max_load": max_load,
            "minutes": minutes, "end_h": t}


def evaluate(instance_id: str, final: Dict[str, Any]) -> Dict[str, Any]:
    """REALM's metrics, plus the dispatch-consistency counts, for one run."""
    m = _realm_metrics()
    inst = load_instance(instance_id)
    vehicles = {v["vehicle_id"]: v for v in inst["vehicles"]}
    reqs = {r["passenger_id"]: r for r in inst["ride_requests"]}

    sims = {vid: simulate(inst, v, final.get(ROUTE + vid))
            for vid, v in vehicles.items()}
    routed: Dict[str, List[str]] = {}
    for vid in vehicles:
        for pid in route_passengers(final.get(ROUTE + vid)):
            routed.setdefault(pid, []).append(vid)

    # A passenger is served by the vehicle whose claim the store records,
    # if that vehicle's route actually picks them up and drops them off.
    pickup: Dict[str, Optional[float]] = {}
    ride: Dict[str, Optional[float]] = {}
    served = []
    for pid in reqs:
        owner = (final.get(REQ + pid) or {}).get("assigned")
        sim = sims.get(owner)
        if sim is not None and pid in sim["ride_min"]:
            pickup[pid] = sim["pickup_h"][pid]
            ride[pid] = sim["ride_min"][pid]
            served.append(pid)

    constraints, completion = [], {}
    for pid, r in reqs.items():
        constraints.append({"type": "deadline", "id": f"{pid}-window",
                            "task_id": f"{pid}-window",
                            "deadline": r["time_window"][1]})
        constraints.append({"type": "deadline", "id": f"{pid}-ride",
                            "task_id": f"{pid}-ride", "deadline": r["deadline"]})
        if pid in pickup:
            completion[f"{pid}-window"] = pickup[pid]
            completion[f"{pid}-ride"] = ride[pid]
    for vid, v in vehicles.items():
        constraints.append({"type": "capacity", "id": f"{vid}-capacity",
                            "capacity_id": vid, "max_capacity": v["capacity"]})
    results = {"task_completion_times": completion,
               "capacity_usage": {vid: s["max_load"] for vid, s in sims.items()}}

    quality = m.PlanningQualityMetrics().evaluate_goal_satisfaction(
        [f"serve {p}" for p in reqs], [f"serve {p}" for p in served])
    csr = m.ConstraintSatisfactionMetrics().evaluate_constraint_satisfaction(
        constraints, results)
    schedule = [{"task_id": vid, "end_time": s["end_h"]}
                for vid, s in sims.items() if s["minutes"]]
    span = m.PlanningOptimalityMetrics().evaluate_makespan(schedule)

    return {
        "goal_satisfaction": quality.value,          # % of passengers served
        "constraint_satisfaction": csr.value,        # % of REALM constraints
        "makespan_h": span.value,
        "travel_min": sum(s["minutes"] for s in sims.values()),
        "double_dispatch": sum(len(vs) > 1 for vs in routed.values()),
        "record_mismatch": sum(
            1 for pid, vs in routed.items() for v in vs
            if (final.get(REQ + pid) or {}).get("assigned") not in (None, v)),
        "unclaimed_routed": sum(
            1 for pid in routed
            if (final.get(REQ + pid) or {}).get("assigned") is None),
        "orphan_claims": sum(
            1 for pid in reqs
            if (final.get(REQ + pid) or {}).get("assigned") in vehicles
            and (final.get(REQ + pid) or {}).get("assigned")
            not in routed.get(pid, [])),
    }
