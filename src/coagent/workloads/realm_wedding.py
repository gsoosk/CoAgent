"""REALM-Bench P5, wedding logistics, run by concurrent errand planners.

REALM-Bench's P5 (the scenario SagaLLM evaluates on) has two guests landing
at Boston airport (Alex 11:00, Jamie 12:30), a gift to collect at the gift
shop (opens 12:00) and clothes at the tailor (closes 14:00), everyone at the
venue for photos at 15:00, and two shared cars at the venue: Pat's, from
12:00, and Chris's, from 13:30. REALM poses it to one planning crew. Here
each errand has its own planner and they share the cars:

    read  wedding/car/*                 both cars' availability and bookings
    write wedding/car/<car>.<errand>    book [start, end]
    read  wedding/map/<location>        travel times (nobody writes these)
    write wedding/plan/<errand>         the stops, in order

Two planners that read the cars before either books can book overlapping
time on one car. Booking is the only shared step; working out the stops is
private, so this has P3's claim-then-work shape.

Reused from REALM-Bench unchanged: the instance (datasets/P5; all 100 files
are this one scenario), its travel times, arrival times, opening hours and
deadline. The second car comes from REALM's own P5 statement ("Chris's
5-seater, available only after 1:30 PM"); the instance file's vehicle list
names locations that have no travel times. Cars start and end at the venue.
"""

from __future__ import annotations

import itertools
import json
from typing import Any, Dict, List, Optional, Tuple

from ..agents.scripted import PlanStep, ScriptedAgent
from ..core import Tool, WriteClass
from ..runtime import CallKind
from .base import Cell
from .realm_urs import REALM_ROOT

__all__ = ["WEDDING_CELLS", "double_bookings", "evaluate", "drive"]

P5_FILE = REALM_ROOT / "datasets" / "P5" / "custom" / "p5_instance_001.json"
CAR, MAP, PLAN = "wedding/car/", "wedding/map/", "wedding/plan/"
VENUE = "W"
CARS = {"pat": "12:00", "chris": "13:30"}

WEDDING_CELLS: Dict[str, Cell] = {}


def hm(t: Any) -> Optional[int]:
    """'13:05' or 785 -> minutes after midnight."""
    if isinstance(t, (int, float)):
        return int(t)
    try:
        h, m = str(t).strip().split(":")
        return int(h) * 60 + int(m)
    except ValueError:
        return None


def available(car: str) -> int:
    """When a car may first leave the venue, in minutes after midnight."""
    return hm(CARS[car]) or 0


def fmt(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


def load() -> Dict[str, Any]:
    if not P5_FILE.exists():
        raise FileNotFoundError(f"{P5_FILE} missing; run scripts/fetch_realm_bench.sh")
    return json.loads(P5_FILE.read_text())


def travel(inst, a: str, b: str) -> Optional[int]:
    if a == b:
        return 0
    tt = inst["travel_times"]
    return tt.get(f"{a}-{b}", tt.get(f"{b}-{a}"))


def errands(inst) -> Dict[str, Dict[str, Any]]:
    """What each planner must do, and the stops it must visit."""
    g = {x["name"]: x for x in inst["guests"]}
    c = inst["constraints"]
    return {
        "alex": {"stops": {"B"}, "ready": {"B": hm(g["Alex"]["arrival_time"])},
                 "what": f"pick up Alex at Boston Airport (B), landing "
                         f"{g['Alex']['arrival_time']}, and bring Alex to "
                         f"the venue"},
        "jamie": {"stops": {"B"}, "ready": {"B": hm(g["Jamie"]["arrival_time"])},
                  "what": f"pick up Jamie at Boston Airport (B), landing "
                          f"{g['Jamie']['arrival_time']}, and bring Jamie to "
                          f"the venue"},
        "errands": {"stops": {"G", "T"},
                    "ready": {"G": hm(c["gift_store_opens"])},
                    "closes": {"T": hm(c["tailor_closes"])},
                    "what": f"collect the gift at the Gift shop (G, opens "
                            f"{c['gift_store_opens']}) and the clothes at the "
                            f"Tailor shop (T, closes {c['tailor_closes']}), "
                            f"and bring them to the venue"},
    }


def drive(inst, job: Dict[str, Any], stops: List[str],
          start: int) -> Tuple[Optional[int], bool]:
    """Leave the venue at ``start``, visit ``stops``, return to the venue.

    Waits at a stop until it is ready (a guest landed, a shop opened).
    Returns (arrival back at the venue, every shop reached before closing).
    """
    t, at, ok = start, VENUE, True
    for s in list(stops) + [VENUE]:
        leg = travel(inst, at, s)
        if leg is None:
            return None, False
        t += leg
        t = max(t, job.get("ready", {}).get(s, 0))
        if s in job.get("closes", {}) and t > job["closes"][s]:
            ok = False
        at = s
    return t, ok


# -- tools -----------------------------------------------------------------
# A car is {"available_from": "HH:MM", <errand>: [start, end], ...}: each
# errand writes only its own field, so two planners never overwrite each
# other's booking -- they can only overlap it.
def _book(v, a):
    return dict(v or {}, **{str(a.get("errand")): [a.get("start"), a.get("end")]})


def _unbook(v, a):
    out = dict(v or {})
    out.pop(str(a.get("errand")), None)
    return out


_mine = lambda a: frozenset({str(a.get("errand"))})                 # noqa: E731
book = Tool(name="book_car", wclass=WriteClass.BLIND, exec_fn=_book,
            covers_fn=_mine)
unbook = Tool(name="cancel_car", wclass=WriteClass.BLIND, exec_fn=_unbook,
              covers_fn=_mine)
plan = Tool(name="write_plan", wclass=WriteClass.BLIND,
            exec_fn=lambda v, a: {"car": a.get("car"), "start": a.get("start"),
                                  "stops": a.get("stops")})

TOOLS = [
    {"name": "list_cars", "kind": "read", "prefix": CAR,
     "description": "both cars: 'available_from', and one "
                    "errand: [start, end] entry per booking"},
    {"name": "get_travel_times", "kind": "read",
     "description": "travel minutes from one location; obj is "
                    "'wedding/map/<B|G|T|W>'"},
    {"name": "book_car", "kind": "write", "tool": book,
     "args": {"errand": "your errand id", "start": "HH:MM",
              "end": "HH:MM"},
     "description": "book a car for [start, end] (replaces your booking on "
                    "it); obj is 'wedding/car/<pat|chris>'"},
    {"name": "cancel_car", "kind": "write", "tool": unbook,
     "args": {"errand": "your errand id"},
     "description": "remove your booking from a car"},
    {"name": "write_plan", "kind": "write", "tool": plan,
     "args": {"car": "pat or chris", "start": "HH:MM, leaving the venue",
              "stops": "locations in visiting order, e.g. [\"G\", \"T\"]"},
     "description": "write your plan; obj is 'wedding/plan/<your errand id>'"},
]


def task_text(inst, eid: str, job: Dict[str, Any]) -> str:
    c = inst["constraints"]
    times = ", ".join(f"{k} {v}" for k, v in inst["travel_times"].items())
    return (
        f"You plan the '{eid}' errand for a wedding: {job['what']}. Everyone "
        f"must be at the venue (W) by {c['photo_time']} for photos. Other "
        f"errands' planners are booking the same cars at the same time.\n\n"
        f"Two cars wait at the venue: Pat's from {CARS['pat']} and Chris's "
        f"from {CARS['chris']}. A car does one errand at a time, leaving from "
        f"and returning to the venue. Travel minutes: {times}.\n\n"
        f"1. Call list_cars to see availability and bookings.\n"
        f"2. Book one car with book_car (errand='{eid}') for a window that "
        f"does not overlap any booking, starts no earlier than the car is "
        f"available, and is long enough to drive your stops and return. "
        f"Book before you call get_travel_times.\n"
        f"3. Work out the order of your stops and write it with write_plan, "
        f"obj 'wedding/plan/{eid}'.\n\n"
        f"Bookings are first-come. If you learn that a car you booked has a "
        f"booking overlapping yours that was made before yours, move yours "
        f"to a free window or the other car."
    )


# -- scoring ---------------------------------------------------------------
def _windows(final, car) -> Dict[str, Tuple[int, int]]:
    out = {}
    for e, w in (final.get(CAR + car) or {}).items():
        if e == "available_from":
            continue
        try:
            s, t = hm(w[0]), hm(w[1])
        except (TypeError, IndexError):
            continue
        if s is not None and t is not None and t > s:
            out[e] = (s, t)
    return out


def double_bookings(final: Dict[str, Any]) -> int:
    n = 0
    for car in CARS:
        ws = list(_windows(final, car).values())
        n += sum(1 for i, a in enumerate(ws) for b in ws[i + 1:]
                 if a[0] < b[1] and b[0] < a[1])
    return n


def evaluate(final: Dict[str, Any]) -> Dict[str, Any]:
    """REALM's metrics: each errand's plan driven on the car it booked."""
    from .realm_eval import _realm_metrics
    mm = _realm_metrics()
    inst = load()
    jobs = errands(inst)
    photo = hm(inst["constraints"]["photo_time"]) or 0
    constraints: List[Dict[str, Any]] = []
    completion: Dict[str, int] = {}
    done: List[str] = []
    for eid, job in jobs.items():
        p = final.get(PLAN + eid) or {}
        car, stops = str(p.get("car")), p.get("stops") or []
        if isinstance(stops, str):
            stops = [s.strip() for s in stops.replace(",", " ").split()]
        stops = [str(s).strip().upper()[:1] for s in stops]
        window = _windows(final, car).get(eid) if car in CARS else None
        end: Optional[int] = None
        ok = False
        if window is not None and set(stops) >= job["stops"]:
            start = max(window[0], available(car))
            end, shops = drive(inst, job, stops, start)
            # a car cannot drive two errands at once: an overlapped
            # booking fails, however good its plan
            clash = any(w[0] < window[1] and window[0] < w[1]
                        for e, w in _windows(final, car).items() if e != eid)
            ok = (end is not None and shops and not clash
                  and end <= window[1] and window[0] >= available(car)
                  and end <= photo)
        if ok:
            done.append(eid)
        for kind, deadline in (("photo", photo), ("booking", window[1]
                                                   if window else None)):
            cid = f"{eid}-{kind}"
            constraints.append({"type": "deadline", "id": cid,
                                "task_id": cid, "deadline": deadline})
            if end is not None:
                completion[cid] = end
    quality = mm.PlanningQualityMetrics().evaluate_goal_satisfaction(
        list(jobs), done)
    csr = mm.ConstraintSatisfactionMetrics().evaluate_constraint_satisfaction(
        constraints, {"task_completion_times": completion})
    n = double_bookings(final)
    return {"goal_satisfaction": quality.value,
            "constraint_satisfaction": csr.value,
            "double_bookings": n, "causes": ["double booking"] * n}


# -- scripted planners ------------------------------------------------------
def _choose(inst, eid, job, cars: Dict[str, Any]):
    """Earliest-finishing free window on either car, over every stop order."""
    best = None
    for car, avail in CARS.items():
        taken = sorted(w for e, w in _windows(
            {CAR + car: (cars or {}).get(CAR + car)}, car).items() if e != eid)
        for stops in itertools.permutations(sorted(job["stops"])):
            start = hm(avail) or 0
            for a, b in taken + [(10 ** 6, 10 ** 6)]:
                end, ok = drive(inst, job, list(stops), start)
                if end is not None and ok and end <= a:
                    break
                start = max(start, b)
            if end is not None and ok and (best is None or end < best[2]):
                best = (car, start, end, list(stops))
    return best


def _planner(inst, eid, job, oracle, rng, kw) -> ScriptedAgent:
    def booking(view):
        car, s, e, _ = _choose(inst, eid, job, view.get("cars"))
        return car, {"errand": eid, "start": fmt(s), "end": fmt(e)}

    def book_step(view) -> List[PlanStep]:
        car, args = booking(view)
        prev = view.get("_car")
        steps = []
        if prev and prev != car:
            steps.append(PlanStep(kind=CallKind.WRITE, obj=CAR + prev,
                                  tool=unbook, args={"errand": eid},
                                  writes=(CAR + prev,), duration=3.0,
                                  label=f"cancel {prev}", repairable=False))
        view["_car"] = car
        steps.append(PlanStep(kind=CallKind.WRITE, obj=CAR + car, tool=book,
                              args=args, writes=(CAR + car,), duration=3.0,
                              label=f"book {car}", repairable=False))
        return steps

    def plan_args(view):
        car, s, _, stops = _choose(inst, eid, job, view.get("cars"))
        return {"car": car, "start": fmt(s), "stops": stops}

    obj = PLAN + eid
    return ScriptedAgent(
        name=eid, oracle=oracle, rng=rng,
        p_task_failure=kw.get("p_task_failure", 0.0),
        p_redo_truncate=kw.get("p_redo_truncate", 0.0),
        plan=[
            PlanStep(kind=CallKind.READ, prefix=CAR, reads=(CAR,),
                     premise="cars", duration=3.0, label="list cars"),
            PlanStep(kind=CallKind.THINK, duration=3.0, label="pick a window"),
            PlanStep(kind=CallKind.WRITE, expand=book_step, label="book",
                     depends_on=("cars",)),
            PlanStep(kind=CallKind.READ, prefix=MAP, reads=(MAP,),
                     premise="map", duration=3.0, label="travel times"),
            PlanStep(kind=CallKind.THINK, duration=8.0, label="order stops"),
            PlanStep(kind=CallKind.WRITE, obj=obj, tool=plan, args=plan_args,
                     reads=(MAP,), writes=(obj,), duration=4.0,
                     label="write plan", depends_on=("cars",)),
        ])


def wedding_cell() -> Cell:
    inst = load()
    jobs = errands(inst)
    initial: Dict[str, Any] = {CAR + c: {"available_from": a}
                               for c, a in CARS.items()}
    locs = sorted(inst["locations"])
    for a in locs:
        initial[MAP + a] = {b: travel(inst, a, b) for b in locs if b != a}

    def build(oracle, rng, **kw):
        return [_planner(inst, e, j, oracle, rng, kw) for e, j in jobs.items()]

    return Cell(
        name="realm-p5-wedding", suite="realm",
        anomaly="double booking: read-pick-book of a car window across a "
                "reasoning gap",
        description="REALM-Bench P5 wedding logistics: three errand "
                    "planners sharing two cars",
        initial=initial, build_agents=build,
        anomaly_free=lambda final, refs: double_bookings(final) == 0,
        llm_spec={"tasks": [task_text(inst, e, j) for e, j in jobs.items()],
                  "tools": TOOLS, "instance": "p5_instance_001"},
        quality=evaluate,
    )


if P5_FILE.exists():
    _c = wedding_cell()
    WEDDING_CELLS[_c.name] = _c
