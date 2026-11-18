"""REALM-Bench P3, urban ride-sharing, run by concurrent vehicle dispatchers.

REALM-Bench (Geng & Chang, arXiv 2502.18836) poses P3 to one planning crew
that runs its role agents one after another and emits a plan as text. There
is no shared state and nothing runs concurrently, so there is nothing for a
concurrency-control protocol to do. This module keeps REALM's instances and
constraints and changes only who plans: one agent per vehicle, all at once,
against a shared store of ride requests. That is how a dispatch system is
actually built, and it is where the isolation question appears.

Each vehicle's dispatcher
  1. lists the open ride requests             read  urs/request/*
  2. claims the ones it can serve best        write urs/request/<p>.assigned
  3. plans a route over the legs it needs     read  urs/map/<location>
  4. writes the route                         write urs/route/<vehicle>

Steps 1-2 are a read-check-write across a reasoning gap, so two dispatchers
can both see a request free and both claim it: a lost update that sends two
vehicles for one passenger. Steps 3-4 touch nothing another vehicle writes.
That is the claim-then-work shape certified chopping is for: the claim piece
must be isolated, the routing after it need not be.

What is reused from REALM-Bench, unchanged: the instance files
(datasets/P3/processed, fetched by scripts/fetch_realm_bench.sh), their
constraint set (time windows, ride deadlines, vehicle capacity), and the
metric classes in evaluation/metrics.py (see ``realm_eval``). The role text of
the tasks follows REALM's P3 Vehicle Dispatch and Route Planning agents.

Instance fields REALM leaves undefined are fixed here, and only here:
travel times are minutes, ``time_window`` is the pickup window in clock
hours, ``deadline`` is the longest allowed ride in minutes, and every vehicle
starts at its listed location at 08:00. REALM's fuel figures are not
enforced: in the same units they make most instances infeasible.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..agents.scripted import PlanStep, ScriptedAgent
from ..core import Tool, WriteClass
from ..runtime import CallKind
from .base import Cell

__all__ = ["REALM_CELLS", "REALM_ROOT", "load_instance", "urs_cell",
           "consistent_dispatch", "route_passengers"]

REALM_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "REALM-Bench"
P3_DIR = REALM_ROOT / "datasets" / "P3" / "processed"

REQ, MAP, ROUTE = "urs/request/", "urs/map/", "urs/route/"
START_HOUR = 8.0

REALM_CELLS: Dict[str, Cell] = {}


# -- tools -----------------------------------------------------------------
assign = Tool(
    name="assign_request", wclass=WriteClass.BLIND,
    exec_fn=lambda v, a: dict(v or {}, assigned=a.get("vehicle")),
    covers_fn=lambda a: frozenset({"assigned"}),
)
write_route = Tool(
    name="write_route", wclass=WriteClass.BLIND,
    exec_fn=lambda v, a: {"stops": a.get("stops")},
)

TOOLS = [
    {"name": "list_requests", "kind": "read", "prefix": REQ,
     "description": "list every ride request with its fields, including "
                    "'assigned' (null when free)"},
    {"name": "get_object", "kind": "read",
     "description": "read one object, e.g. a request 'urs/request/p001'"},
    {"name": "get_travel_times", "kind": "read",
     "description": "travel times in minutes from one location; obj is "
                    "'urs/map/<location>'"},
    {"name": "assign_request", "kind": "write", "tool": assign,
     "args": {"vehicle": "your vehicle id to claim it, or null to release "
                         "your claim"},
     "description": "set who serves a request; obj is 'urs/request/<id>'"},
    {"name": "write_route", "kind": "write", "tool": write_route,
     "args": {"stops": "ordered list of {\"passenger\": id, \"action\": "
                       "\"pickup\" or \"dropoff\"}"},
     "description": "write your vehicle's route; obj is "
                    "'urs/route/<your vehicle id>'"},
]


# -- instances -------------------------------------------------------------
def load_instance(instance_id: str) -> Dict[str, Any]:
    path = P3_DIR / f"{instance_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; run scripts/fetch_realm_bench.sh")
    return json.loads(path.read_text())


def travel(inst: Dict[str, Any], a: str, b: str) -> Optional[int]:
    if a == b:
        return 0
    return inst["city_map"]["distances"].get(f"{a}-{b}")


def quota(inst: Dict[str, Any]) -> int:
    """Requests each vehicle may claim: enough that all could be served."""
    return math.ceil(len(inst["ride_requests"]) / len(inst["vehicles"]))


def initial_state(inst: Dict[str, Any]) -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    for r in inst["ride_requests"]:
        state[REQ + r["passenger_id"]] = {
            "pickup": r["pickup"], "dropoff": r["dropoff"],
            "time_window": [round(x, 2) for x in r["time_window"]],
            "deadline": r["deadline"], "assigned": None,
        }
    locs = inst["city_map"]["locations"]
    for a in locs:
        state[MAP + a] = {b: travel(inst, a, b) for b in locs if b != a}
    return state


# -- scoring: is the dispatch consistent? -----------------------------------
def route_passengers(route: Any) -> List[str]:
    """Passenger ids a written route serves, tolerant of how a model wrote it."""
    stops = (route or {}).get("stops") if isinstance(route, dict) else None
    if isinstance(stops, str):
        try:
            stops = json.loads(stops)
        except json.JSONDecodeError:
            return []
    out = []
    for s in stops or []:
        pid = s.get("passenger") if isinstance(s, dict) else None
        if isinstance(pid, str):
            pid = pid.split("/")[-1]
            if pid not in out:
                out.append(pid)
    return out


def consistent_dispatch(final: Dict[str, Any], vehicles: List[str]) -> bool:
    """No passenger is routed by two vehicles, or against another's claim.

    This is the isolation property: a serial execution of dispatchers can
    never violate it, whatever each one chose to claim. A route that serves
    a passenger nobody claimed is not counted here: one dispatcher forgetting
    a claim happens serially too, so it is a task error, scored with the
    rest of plan quality in ``realm_eval``.
    """
    seen: Dict[str, str] = {}
    for v in vehicles:
        for pid in route_passengers(final.get(ROUTE + v)):
            if pid in seen:
                return False
            seen[pid] = v
            owner = (final.get(REQ + pid) or {}).get("assigned")
            if owner is not None and owner != v:
                return False
    return True


# -- tasks -----------------------------------------------------------------
def task_text(inst: Dict[str, Any], veh: Dict[str, Any]) -> str:
    vid, loc = veh["vehicle_id"], veh["location"]
    times = ", ".join(f"{b} {travel(inst, loc, b)}"
                      for b in inst["city_map"]["locations"] if b != loc)
    return (
        f"You are the dispatcher and route planner for vehicle '{vid}', now "
        f"at {loc}, carrying at most {veh['capacity']} passengers. Other "
        f"vehicles' dispatchers are working on the same ride requests at the "
        f"same time.\n\n"
        f"1. Call list_requests. A request is free only if its 'assigned' "
        f"is null.\n"
        f"2. Claim the free requests you can serve best, at most "
        f"{quota(inst)}, with assign_request (vehicle='{vid}'). Never claim "
        f"a request assigned to another vehicle. To choose, use these travel "
        f"times in minutes from {loc}: {times}. Make all your claims before "
        f"you call get_travel_times.\n"
        f"3. Plan a route that picks up and drops off exactly the requests "
        f"you claimed, using get_travel_times for the legs. You start at "
        f"{loc} at hour {START_HOUR:.1f}. Each pickup must fall inside the "
        f"request's time_window (clock hours; you may wait), and each ride "
        f"may take at most 'deadline' minutes.\n"
        f"4. Write it with write_route, obj 'urs/route/{vid}'.\n\n"
        f"Claims are first-come. If you learn that another vehicle had "
        f"already claimed a request you also claimed, it is theirs: set it "
        f"back to their vehicle id with assign_request and leave it out of "
        f"your route. If you cannot serve a request you claimed, release it "
        f"with assign_request (vehicle=null)."
    )


# -- scripted dispatchers ---------------------------------------------------
def _choose(inst, veh, reqs: Dict[str, Any]) -> List[str]:
    """Nearest free pickups first, up to the quota."""
    vid, loc = veh["vehicle_id"], veh["location"]
    free = [(travel(inst, loc, str((v or {}).get("pickup"))) or 99, k)
            for k, v in sorted((reqs or {}).items())
            if (v or {}).get("assigned") in (None, vid)]
    return [k for _, k in sorted(free)[:quota(inst)]]


def _route(chosen: List[str]) -> Dict[str, Any]:
    stops = []
    for k in chosen:
        pid = k.split("/")[-1]
        stops += [{"passenger": pid, "action": "pickup"},
                  {"passenger": pid, "action": "dropoff"}]
    return {"stops": stops}


def _vehicle(inst, veh, oracle, rng, kw) -> ScriptedAgent:
    vid = veh["vehicle_id"]

    def claims(view) -> List[PlanStep]:
        # What this run already claimed lives in the view, which a restart
        # clears -- after a saga undo the claims are gone and must be redone.
        reqs = view.get("reqs") or {}
        before = view.get("_claimed", [])
        chosen = _choose(inst, veh, reqs)
        view["_claimed"] = chosen
        # A dropped claim is released if the view still shows it as ours. If
        # it shows another vehicle, that vehicle claimed it first: hand it
        # back. Under MTPO this rewrite is the heal -- our own claim sits
        # above theirs in the trajectory, and only a later write of their id
        # makes the sigma-serial state agree with who routes the passenger.
        steps = []
        for k in before:
            owner = (reqs.get(k) or {}).get("assigned")
            if k in chosen or owner is None:
                continue
            steps.append(PlanStep(
                kind=CallKind.WRITE, obj=k, tool=assign,
                args={"vehicle": None if owner == vid else owner},
                writes=(k,), duration=3.0, repairable=False,
                label=f"{'release' if owner == vid else 'return'} "
                      f"{k.split('/')[-1]}"))
        steps += [PlanStep(kind=CallKind.WRITE, obj=k, tool=assign,
                           args={"vehicle": vid}, writes=(k,), duration=3.0,
                           label=f"claim {k.split('/')[-1]}",
                           repairable=False)
                  for k in chosen if k not in before]
        return steps

    route_obj = ROUTE + vid
    return ScriptedAgent(
        name=vid, oracle=oracle, rng=rng,
        p_task_failure=kw.get("p_task_failure", 0.0),
        p_redo_truncate=kw.get("p_redo_truncate", 0.0),
        plan=[
            PlanStep(kind=CallKind.READ, prefix=REQ, reads=(REQ,),
                     premise="reqs", duration=3.0, label="list requests"),
            PlanStep(kind=CallKind.THINK, duration=3.0, label="choose"),
            PlanStep(kind=CallKind.WRITE, expand=claims, label="claim",
                     depends_on=("reqs",)),
            PlanStep(kind=CallKind.READ, prefix=MAP, reads=(MAP,),
                     premise="map", duration=3.0, label="travel times"),
            PlanStep(kind=CallKind.THINK, duration=8.0, label="plan route"),
            PlanStep(kind=CallKind.WRITE, obj=route_obj, tool=write_route,
                     args=lambda view: _route(_choose(inst, veh,
                                                      view.get("reqs"))),
                     # a model reads more legs after the first one, and
                     # those reads belong with the route, not a revision
                     reads=(MAP,), writes=(route_obj,), duration=4.0,
                     label="write route",
                     depends_on=("reqs",)),
        ],
    )


def urs_cell(instance_id: str) -> Cell:
    inst = load_instance(instance_id)
    vehicles = [v["vehicle_id"] for v in inst["vehicles"]]

    def build(oracle, rng, **kw):
        return [_vehicle(inst, v, oracle, rng, kw) for v in inst["vehicles"]]

    return Cell(
        name=f"realm-p3-{instance_id.split('_')[-1]}",
        suite="realm",
        anomaly="double dispatch: read-check-write claim across a "
                "reasoning gap (lost update)",
        description=f"REALM-Bench P3 {instance_id}: {len(vehicles)} vehicle "
                    f"dispatchers, {len(inst['ride_requests'])} requests",
        initial=initial_state(inst),
        build_agents=build,
        anomaly_free=lambda final, refs: consistent_dispatch(final, vehicles),
        llm_spec={"tasks": [task_text(inst, v) for v in inst["vehicles"]],
                  "tools": TOOLS, "instance": instance_id},
    )


# The first three P3 instances with three vehicles and at least six
# requests: a fixed rule, so the choice is not tuned to any result.
def _selected(k: int = 3) -> List[str]:
    out = []
    for path in sorted(P3_DIR.glob("p3_instance_*.json")):
        inst = json.loads(path.read_text())
        if len(inst["vehicles"]) == 3 and len(inst["ride_requests"]) >= 6:
            out.append(path.stem)
        if len(out) == k:
            break
    return out


if P3_DIR.exists():
    for _iid in _selected():
        _c = urs_cell(_iid)
        REALM_CELLS[_c.name] = _c
