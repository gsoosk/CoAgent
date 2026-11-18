"""REALM-Bench P2, multi-group campus tours, run by concurrent group planners.

REALM-Bench's P2 has visitor groups (size, arrival, preferred start) and
tour guides (availability, maximum group size); every tour lasts 90
minutes. REALM poses it to one planner. Here each group has its own
planner, and the planners share the guides:

    read  tours/guide/*                 every guide's availability, bookings
    write tours/guide/<g>.<group>       book [start, end]
    read  tours/map/<building>          walking minutes (nobody writes these)
    write tours/plan/<group>            the guide, start and buildings

Two planners that read the guides before either books can book one guide
for overlapping tours. Booking is the only shared step; choosing the
route is private and takes several map reads, so P2 has the longest
private stretch of the REALM problems here.

Reused from REALM-Bench unchanged: the P2 instances (datasets/P2/custom)
and, for the route, the campus of the P1 instance with the same number
(datasets/P1/custom: buildings, directional walking minutes, opening
hours), since P2 is P1 run for several groups. Not in the data and fixed
here: a building visit takes 15 minutes, and a tour starts and ends at the
entrance.
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

__all__ = ["TOURS_CELLS", "conflicts", "evaluate"]

P1_DIR = REALM_ROOT / "datasets" / "P1" / "custom"
P2_DIR = REALM_ROOT / "datasets" / "P2" / "custom"
GUIDE, MAP, PLAN = "tours/guide/", "tours/map/", "tours/plan/"
TOUR, VISIT, GATE = 90, 15, "entrance"
INSTANCES = ("001", "002", "003")

TOURS_CELLS: Dict[str, Cell] = {}


def hm(t: Any) -> Optional[int]:
    """'13:05', 13 (an hour) or 13.5 -> minutes after midnight."""
    if isinstance(t, (int, float)):
        return int(round(t * 60)) if t < 24 else int(t)
    try:
        h, m = str(t).strip().split(":")
        return int(h) * 60 + int(m)
    except ValueError:
        return None


def fmt(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


def load(num: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    p2, p1 = P2_DIR / f"p2_instance_{num}.json", P1_DIR / f"p1_instance_{num}.json"
    for p in (p2, p1):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing; run scripts/fetch_realm_bench.sh")
    return json.loads(p2.read_text()), json.loads(p1.read_text())


def walk(campus, a: str, b: str) -> Optional[int]:
    return 0 if a == b else campus["distances"].get(f"{a}-{b}")


def tour_end(campus, start: int, stops: List[str]) -> Tuple[Optional[int], int]:
    """Walk the stops from the entrance and back; (end, buildings visited
    while open). A stop reached while closed is walked past, not visited."""
    t, at, seen = start, GATE, 0
    for s in list(stops) + [GATE]:
        leg = walk(campus, at, s)
        if leg is None:
            return None, seen
        t, at = t + leg, s
        if s != GATE:
            lo, hi = campus["time_windows"].get(s, [0, 24])
            if lo * 60 <= t and t + VISIT <= hi * 60:
                seen += 1
            t += VISIT
    return t, seen


# -- tools -----------------------------------------------------------------
def _book(v, a):
    return dict(v or {}, **{str(a.get("group")): [a.get("start"), a.get("end")]})


def _unbook(v, a):
    out = dict(v or {})
    out.pop(str(a.get("group")), None)
    return out


_mine = lambda a: frozenset({str(a.get("group"))})                  # noqa: E731
book = Tool(name="book_guide", wclass=WriteClass.BLIND, exec_fn=_book,
            covers_fn=_mine)
unbook = Tool(name="cancel_guide", wclass=WriteClass.BLIND, exec_fn=_unbook,
              covers_fn=_mine)
plan = Tool(name="write_plan", wclass=WriteClass.BLIND,
            exec_fn=lambda v, a: {"guide": a.get("guide"),
                                  "start": a.get("start"),
                                  "stops": a.get("stops")})

TOOLS = [
    {"name": "list_guides", "kind": "read", "prefix": GUIDE,
     "description": "every guide: 'available' [from, to] in hours, "
                    "'max_group_size', and one group: [start, end] entry "
                    "per booking"},
    {"name": "get_travel_times", "kind": "read",
     "description": "walking minutes from one place (directional); obj is "
                    "'tours/map/<entrance|library|cafeteria|gym|"
                    "student_center>'"},
    {"name": "book_guide", "kind": "write", "tool": book,
     "args": {"group": "your group id", "start": "HH:MM", "end": "HH:MM"},
     "description": "book a guide for [start, end] (replaces your booking "
                    "on that guide); obj is 'tours/guide/<guide id>'"},
    {"name": "cancel_guide", "kind": "write", "tool": unbook,
     "args": {"group": "your group id"},
     "description": "remove your booking from a guide"},
    {"name": "write_plan", "kind": "write", "tool": plan,
     "args": {"guide": "guide id", "start": "HH:MM",
              "stops": "buildings in visiting order"},
     "description": "write your tour plan; obj is 'tours/plan/<group id>'"},
]


def task_text(group: Dict[str, Any], campus: Dict[str, Any]) -> str:
    gid = group["group_id"]
    hours = ", ".join(f"{b} {w[0]}:00-{w[1]}:00"
                      for b, w in campus["time_windows"].items())
    return (
        f"You plan the campus tour for '{gid}': {group['size']} visitors, "
        f"arriving at {group['arrival_time']}:00, who would like to start "
        f"around {group['preferred_time']}:00. Planners for other groups are "
        f"booking the same guides at the same time.\n\n"
        f"1. Call list_guides. Book one guide with book_guide "
        f"(group='{gid}') for a {TOUR}-minute tour that starts no earlier "
        f"than your arrival, as close to your preferred time as you can, "
        f"inside the guide's availability, for a guide whose "
        f"max_group_size fits your group, and not overlapping any booking. "
        f"Book before you call get_travel_times.\n"
        f"2. Plan the route: start and end at the entrance within the "
        f"{TOUR} minutes, visiting as many buildings as you can. A visit "
        f"takes {VISIT} minutes and counts only while the building is open "
        f"({hours}). Walking times differ by direction; read them with "
        f"get_travel_times.\n"
        f"3. Write it with write_plan, obj 'tours/plan/{gid}'.\n\n"
        f"Bookings are first-come. If you learn that a guide you booked has "
        f"a booking overlapping yours that was made before yours, move "
        f"yours to a free time or another guide."
    )


# -- scoring ---------------------------------------------------------------
def _windows(final, guide) -> Dict[str, Tuple[int, int]]:
    out = {}
    for g, w in (final.get(GUIDE + guide) or {}).items():
        if g in ("available", "max_group_size") or not isinstance(w, list):
            continue
        s, e = (hm(w[0]), hm(w[1])) if len(w) >= 2 else (None, None)
        if s is not None and e is not None and e > s:
            out[g] = (s, e)
    return out


def conflicts(final, groups: List[str], guides: List[str]):
    """(object, agent index, agent index) for overlapping guide bookings."""
    idx = {g: i for i, g in enumerate(groups)}
    out = []
    for guide in guides:
        items = [(idx[g], w) for g, w in _windows(final, guide).items()
                 if g in idx]
        out += [(GUIDE + guide, a, b)
                for (a, x), (b, y) in itertools.combinations(items, 2)
                if x[0] < y[1] and y[0] < x[1]]
    return out


def evaluate(num: str, final: Dict[str, Any]) -> Dict[str, Any]:
    """REALM's P2 goals: serve every group, wait little, use the campus."""
    from .realm_eval import _realm_metrics
    mm = _realm_metrics()
    inst, campus = load(num)
    guides = {g["guide_id"]: g for g in inst["tour_guides"]}
    groups = [g["group_id"] for g in inst["visitor_groups"]]
    served, waits, visits = [], [], []
    constraints: List[Dict[str, Any]] = []
    completion: Dict[str, float] = {}
    for grp in inst["visitor_groups"]:
        gid = grp["group_id"]
        p = final.get(PLAN + gid) or {}
        guide = str(p.get("guide"))
        w = _windows(final, guide).get(gid) if guide in guides else None
        stops = p.get("stops") or []
        if isinstance(stops, str):
            stops = [s.strip() for s in stops.replace(",", " ").split()]
        stops = [str(s).strip().lower() for s in stops]
        ok = False
        if w is not None:
            g = guides[guide]
            end, seen = tour_end(campus, w[0], stops)
            clash = any(o[0] < w[1] and w[0] < o[1]
                        for k, o in _windows(final, guide).items() if k != gid)
            ok = (end is not None and end <= w[0] + TOUR and not clash
                  and w[0] >= grp["arrival_time"] * 60
                  and g["availability"][0] * 60 <= w[0]
                  and w[0] + TOUR <= g["availability"][1] * 60
                  and grp["size"] <= g["max_group_size"])
            if ok:
                waits.append(abs(w[0] - grp["preferred_time"] * 60))
                visits.append(seen)
            completion[f"{gid}-tour"] = end if end is not None else 10 ** 6
        constraints.append({"type": "deadline", "id": f"{gid}-tour",
                            "task_id": f"{gid}-tour",
                            "deadline": (w[0] + TOUR) if w else None})
        if ok:
            served.append(gid)
    quality = mm.PlanningQualityMetrics().evaluate_goal_satisfaction(
        groups, served)
    csr = mm.ConstraintSatisfactionMetrics().evaluate_constraint_satisfaction(
        constraints, {"task_completion_times": completion})
    n = len(conflicts(final, groups, list(guides)))
    return {"goal_satisfaction": quality.value,
            "constraint_satisfaction": csr.value,
            "wait_min": sum(waits) / len(waits) if waits else None,
            "buildings": sum(visits) / len(visits) if visits else None,
            "double_bookings": n}


# -- scripted planners ------------------------------------------------------
def _choose(inst, grp, guides_view: Dict[str, Any]):
    """The fitting guide and start nearest the preferred time."""
    best = None
    for g in inst["tour_guides"]:
        if grp["size"] > g["max_group_size"]:
            continue
        lo = max(grp["arrival_time"], g["availability"][0]) * 60
        hi = g["availability"][1] * 60 - TOUR
        taken = list(_windows({GUIDE + g["guide_id"]:
                               (guides_view or {}).get(GUIDE + g["guide_id"])},
                              g["guide_id"]).items())
        for s in range(lo, hi + 1, 15):
            if any(w[0] < s + TOUR and s < w[1]
                   for k, w in taken if k != grp["group_id"]):
                continue
            cost = abs(s - grp["preferred_time"] * 60)
            if best is None or cost < best[2]:
                best = (g["guide_id"], s, cost)
    return best


def _route(campus, start: int) -> List[str]:
    """Greedy: the nearest building still worth walking to, while time lasts."""
    stops: List[str] = []
    left = [b for b in campus["locations"] if b != GATE]
    while left:
        cand = []
        for b in left:
            end, seen = tour_end(campus, start, stops + [b])
            if end is not None and end <= start + TOUR and seen == len(stops) + 1:
                cand.append((end, b))
        if not cand:
            break
        stops.append(min(cand)[1])
        left.remove(stops[-1])
    return stops


def _planner(inst, campus, grp, oracle, rng, kw) -> ScriptedAgent:
    gid = grp["group_id"]

    def book_steps(view) -> List[PlanStep]:
        choice = _choose(inst, grp, view.get("guides"))
        if choice is None:
            return []
        guide, s, _ = choice
        prev = view.get("_guide")
        steps = []
        if prev and prev != guide:
            steps.append(PlanStep(kind=CallKind.WRITE, obj=GUIDE + prev,
                                  tool=unbook, args={"group": gid},
                                  writes=(GUIDE + prev,), duration=3.0,
                                  label=f"cancel {prev}", repairable=False))
        view["_guide"] = guide
        steps.append(PlanStep(kind=CallKind.WRITE, obj=GUIDE + guide,
                              tool=book, writes=(GUIDE + guide,),
                              args={"group": gid, "start": fmt(s),
                                    "end": fmt(s + TOUR)},
                              duration=3.0, label=f"book {guide}",
                              repairable=False))
        return steps

    def plan_args(view):
        choice = _choose(inst, grp, view.get("guides"))
        if choice is None:
            return {"guide": None, "start": None, "stops": []}
        guide, s, _ = choice
        return {"guide": guide, "start": fmt(s), "stops": _route(campus, s)}

    obj = PLAN + gid
    return ScriptedAgent(
        name=gid, oracle=oracle, rng=rng,
        p_task_failure=kw.get("p_task_failure", 0.0),
        p_redo_truncate=kw.get("p_redo_truncate", 0.0),
        plan=[
            PlanStep(kind=CallKind.READ, prefix=GUIDE, reads=(GUIDE,),
                     premise="guides", duration=3.0, label="list guides"),
            PlanStep(kind=CallKind.THINK, duration=3.0, label="pick a slot"),
            PlanStep(kind=CallKind.WRITE, expand=book_steps, label="book",
                     depends_on=("guides",)),
            PlanStep(kind=CallKind.READ, prefix=MAP, reads=(MAP,),
                     premise="map", duration=3.0, label="travel times"),
            PlanStep(kind=CallKind.THINK, duration=10.0, label="plan route"),
            PlanStep(kind=CallKind.WRITE, obj=obj, tool=plan, args=plan_args,
                     reads=(MAP,), writes=(obj,), duration=4.0,
                     label="write plan", depends_on=("guides",)),
        ])


def tours_cell(num: str) -> Cell:
    inst, campus = load(num)
    groups = [g["group_id"] for g in inst["visitor_groups"]]
    guides = [g["guide_id"] for g in inst["tour_guides"]]
    initial: Dict[str, Any] = {
        GUIDE + g["guide_id"]: {"available": g["availability"],
                                "max_group_size": g["max_group_size"]}
        for g in inst["tour_guides"]}
    for a in campus["locations"]:
        initial[MAP + a] = {b: walk(campus, a, b)
                            for b in campus["locations"] if b != a}

    def build(oracle, rng, **kw):
        return [_planner(inst, campus, g, oracle, rng, kw)
                for g in inst["visitor_groups"]]

    return Cell(
        name=f"realm-p2-tours-{num}", suite="realm",
        anomaly="double booking: read-pick-book of a guide across a "
                "reasoning gap",
        description=f"REALM-Bench P2 instance {num} (campus of P1 {num}): "
                    f"three group planners sharing three guides",
        initial=initial, build_agents=build,
        anomaly_free=lambda final, refs: not conflicts(final, groups, guides),
        llm_spec={"tasks": [task_text(g, campus) for g in inst["visitor_groups"]],
                  "tools": TOOLS, "instance": f"p2_instance_{num}",
                  "conflicts": lambda final: conflicts(final, groups, guides)},
        quality=lambda final: evaluate(num, final),
    )


if P2_DIR.exists() and P1_DIR.exists():
    for _n in INSTANCES:
        _c = tours_cell(_n)
        TOURS_CELLS[_c.name] = _c
