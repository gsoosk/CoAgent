"""REALM-Bench P6/P9, Thanksgiving dinner, run by concurrent task planners.

REALM-Bench's P6 (with P9, its flight-delay variant, the scenario SagaLLM
evaluates on) has mom at home with a car, dad landing at the airport and
renting a car there, the sister landing later and needing a ride, a turkey
(4 hours) and side dishes (2 hours) that need someone home while they cook,
and dinner at 18:00. Here each task has its own planner, and the shared
resource is people's time:

    read  thanks/person/*                 who is where, from when; bookings
    write thanks/person/<p>.<task>        book [start, end, home|drive]
    write thanks/plan/<task>              the task's timeline

One person can supervise both dishes at once, so two "home" bookings on a
person are compatible; anything else that overlaps is a double booking
(mom cannot watch the oven and drive to the airport at once).

Reused from REALM-Bench unchanged: the P6 instance (datasets/P6; all 100
files are one scenario) and the P9 instances' delayed arrival times
(datasets/P9/disruptions), with the travel time, cooking times and the
supervision rule from REALM's P6 statement (home to airport 60 minutes,
turkey 4 hours, sides 2 hours, "someone must stay home during cooking").
P9's delay is announced before planning starts (REALM: the delay is known
when the flight should depart), so a P9 cell is P6 with the delayed times.
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

__all__ = ["THANKS_CELLS", "conflicts", "evaluate"]

P6_FILE = REALM_ROOT / "datasets" / "P6" / "custom" / "p6_instance_001.json"
P9_DIR = REALM_ROOT / "datasets" / "P9" / "disruptions"
PERSON, PLAN = "thanks/person/", "thanks/plan/"
DRIVE, DINNER, MOM_FROM = 60, 18 * 60, 8 * 60
COOK = {"turkey": 240, "sides": 120}
TASKS = ["turkey", "sides", "sister"]
VARIANTS = ("p6", "p9-001", "p9-002")

THANKS_CELLS: Dict[str, Cell] = {}


def hm(t: Any) -> Optional[int]:
    if isinstance(t, (int, float)):
        return int(t)
    try:
        h, m = str(t).strip().split(":")
        return int(h) * 60 + int(m)
    except ValueError:
        return None


def fmt(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


def load(variant: str) -> Dict[str, int]:
    """Landing times for the variant, in minutes after midnight."""
    if not P6_FILE.exists():
        raise FileNotFoundError(f"{P6_FILE} missing; run scripts/fetch_realm_bench.sh")
    base = json.loads(P6_FILE.read_text())
    land = {f: hm(v["arrival_time"]) or 0 for f, v in base["flights"].items()}
    if variant.startswith("p9"):
        d = json.loads((P9_DIR / f"p9_instance_{variant[3:]}.json").read_text())
        for x in d["disruptions"]:
            if x["type"] == "flight_delay":
                land[x["flight"]] = hm(x["new_arrival_time"]) or 0
    return {"dad": land["flight1"], "sister": land["flight2"]}


# -- tools -----------------------------------------------------------------
def _book(v, a):
    return dict(v or {}, **{str(a.get("task")): [a.get("start"), a.get("end"),
                                                 a.get("activity")]})


def _unbook(v, a):
    out = dict(v or {})
    out.pop(str(a.get("task")), None)
    return out


_mine = lambda a: frozenset({str(a.get("task"))})                   # noqa: E731
book = Tool(name="book_person", wclass=WriteClass.BLIND, exec_fn=_book,
            covers_fn=_mine)
unbook = Tool(name="cancel_person", wclass=WriteClass.BLIND, exec_fn=_unbook,
              covers_fn=_mine)
plan = Tool(name="write_plan", wclass=WriteClass.BLIND,
            exec_fn=lambda v, a: {"person": a.get("person"),
                                  "start": a.get("start"), "end": a.get("end"),
                                  "timeline": a.get("timeline")})

TOOLS = [
    {"name": "list_people", "kind": "read", "prefix": PERSON,
     "description": "everyone who can help: where they are from when, and "
                    "one task: [start, end, activity] entry per booking"},
    {"name": "book_person", "kind": "write", "tool": book,
     "args": {"task": "your task id", "start": "HH:MM", "end": "HH:MM",
              "activity": "home (supervise cooking) or drive"},
     "description": "book a person for your task (replaces your booking on "
                    "them); obj is 'thanks/person/<mom|dad>'"},
    {"name": "cancel_person", "kind": "write", "tool": unbook,
     "args": {"task": "your task id"},
     "description": "remove your booking from a person"},
    {"name": "write_plan", "kind": "write", "tool": plan,
     "args": {"person": "mom or dad", "start": "HH:MM", "end": "HH:MM",
              "timeline": "short list of timed steps"},
     "description": "write your task's plan; obj is 'thanks/plan/<task id>'"},
]


def task_text(task: str, land: Dict[str, int]) -> str:
    world = (
        f"Thanksgiving dinner is at {fmt(DINNER)}. Mom is home with a car "
        f"from {fmt(MOM_FROM)}. Dad lands at the airport at "
        f"{fmt(land['dad'])} and rents a car there; driving home takes "
        f"{DRIVE} minutes, so he can be home from {fmt(land['dad'] + DRIVE)} "
        f"if he drives straight home. The sister lands at "
        f"{fmt(land['sister'])} and has no car. Home to airport is {DRIVE} "
        f"minutes each way. One person at home can supervise any amount of "
        f"cooking, but a person who is driving is not home. Planners for "
        f"the other tasks are booking the same people at the same time.")
    what = {
        "turkey": "cook the turkey and gravy: it needs someone at home for "
                  "4 hours, finished by dinner. Book a person with "
                  "activity='home' for those 4 hours.",
        "sides": "cook the side dishes: they need someone at home for 2 "
                 "hours, finished by dinner. Book a person with "
                 "activity='home' for those 2 hours.",
        "sister": "get the sister home before dinner. Book a driver with "
                  "activity='drive' for the whole trip: mom would leave "
                  "home, wait at the airport if early, and drive back; dad "
                  "is already at the airport once he lands and can wait "
                  "for her there.",
    }[task]
    return (
        f"You plan the '{task}' task: {what}\n\n{world}\n\n"
        f"1. Call list_people to see where everyone is and what is booked.\n"
        f"2. Book one person with book_person (task='{task}') for a window "
        f"that does not overlap a booking of theirs (two 'home' bookings "
        f"may overlap) and that they can actually be there for.\n"
        f"3. Write your plan with write_plan, obj 'thanks/plan/{task}'.\n\n"
        f"Bookings are first-come. If you learn that a person you booked has "
        f"an incompatible booking made before yours, move yours to a free "
        f"window or another person."
    )


# -- scoring ---------------------------------------------------------------
def _bookings(final, who) -> Dict[str, Tuple[int, int, str]]:
    out = {}
    for t, w in (final.get(PERSON + who) or {}).items():
        if not isinstance(w, list) or len(w) < 3:
            continue
        s, e = hm(w[0]), hm(w[1])
        if s is not None and e is not None and e > s:
            out[t] = (s, e, str(w[2]).lower())
    return out


def _clash(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1] and not (a[2] == b[2] == "home")


def conflicts(final):
    """(object, agent index, agent index) for incompatible bookings."""
    idx = {t: i for i, t in enumerate(TASKS)}
    out = []
    for who in ("mom", "dad"):
        items = [(idx[t], w) for t, w in _bookings(final, who).items()
                 if t in idx]
        out += [(PERSON + who, a, b)
                for (a, x), (b, y) in itertools.combinations(items, 2)
                if _clash(x, y)]
    return out


def _feasible(task, who, w, land) -> bool:
    """Can this person actually do this task in this window?"""
    s, e, act = w
    if e > DINNER:
        return False
    home_from = MOM_FROM if who == "mom" else land["dad"] + DRIVE
    if task in COOK:
        return act == "home" and s >= home_from and e - s >= COOK[task]
    if act != "drive":
        return False
    if who == "mom":
        return s >= MOM_FROM and e >= max(s + DRIVE, land["sister"]) + DRIVE
    return s >= land["dad"] and e >= max(s, land["sister"]) + DRIVE


def evaluate(variant: str, final: Dict[str, Any]) -> Dict[str, Any]:
    from .realm_eval import _realm_metrics
    mm = _realm_metrics()
    land = load(variant)
    done, constraints, completion = [], [], {}
    for task in TASKS:
        mine = [(who, w) for who in ("mom", "dad")
                for t, w in _bookings(final, who).items() if t == task]
        ok = False
        if len(mine) == 1:
            who, w = mine[0]
            others = [x for t, x in _bookings(final, who).items() if t != task]
            ok = (_feasible(task, who, w, land)
                  and not any(_clash(w, o) for o in others))
            completion[task] = w[1]
        if ok:
            done.append(task)
        constraints.append({"type": "deadline", "id": f"{task}-dinner",
                            "task_id": task, "deadline": DINNER})
    quality = mm.PlanningQualityMetrics().evaluate_goal_satisfaction(TASKS, done)
    csr = mm.ConstraintSatisfactionMetrics().evaluate_constraint_satisfaction(
        constraints, {"task_completion_times": completion})
    return {"goal_satisfaction": quality.value,
            "constraint_satisfaction": csr.value,
            "double_bookings": len(conflicts(final))}


# -- scripted planners ------------------------------------------------------
def _choose(task, land, people: Dict[str, Any]):
    """First feasible person and window that clashes with nothing booked."""
    if task in COOK:
        windows = [(who, (DINNER - COOK[task], DINNER, "home"))
                   for who in ("mom", "dad")]
    else:
        trip_dad = (land["dad"], max(land["dad"], land["sister"]) + DRIVE,
                    "drive")
        windows = [("dad", trip_dad)] + [
            ("mom", (s, max(s + DRIVE, land["sister"]) + DRIVE, "drive"))
            for s in range(MOM_FROM, DINNER, 15)]
    for who, w in windows:
        taken = [x for t, x in _bookings(
            {PERSON + who: (people or {}).get(PERSON + who)}, who).items()
            if t != task]
        if _feasible(task, who, w, land) and not any(_clash(w, o) for o in taken):
            return who, w
    return None


def _planner(task, land, oracle, rng, kw) -> ScriptedAgent:
    def book_steps(view) -> List[PlanStep]:
        choice = _choose(task, land, view.get("people"))
        if choice is None:
            return []
        who, (s, e, act) = choice
        prev = view.get("_who")
        steps = []
        if prev and prev != who:
            steps.append(PlanStep(kind=CallKind.WRITE, obj=PERSON + prev,
                                  tool=unbook, args={"task": task},
                                  writes=(PERSON + prev,), duration=3.0,
                                  label=f"cancel {prev}", repairable=False))
        view["_who"] = who
        steps.append(PlanStep(kind=CallKind.WRITE, obj=PERSON + who, tool=book,
                              args={"task": task, "start": fmt(s),
                                    "end": fmt(e), "activity": act},
                              writes=(PERSON + who,), duration=3.0,
                              label=f"book {who}", repairable=False))
        return steps

    def plan_args(view):
        choice = _choose(task, land, view.get("people"))
        if choice is None:
            return {"person": None}
        who, (s, e, _) = choice
        return {"person": who, "start": fmt(s), "end": fmt(e),
                "timeline": [f"{fmt(s)} start", f"{fmt(e)} done"]}

    obj = PLAN + task
    return ScriptedAgent(
        name=task, oracle=oracle, rng=rng,
        p_task_failure=kw.get("p_task_failure", 0.0),
        p_redo_truncate=kw.get("p_redo_truncate", 0.0),
        plan=[
            PlanStep(kind=CallKind.READ, prefix=PERSON, reads=(PERSON,),
                     premise="people", duration=3.0, label="list people"),
            PlanStep(kind=CallKind.THINK, duration=3.0, label="pick a window"),
            PlanStep(kind=CallKind.WRITE, expand=book_steps, label="book",
                     depends_on=("people",)),
            PlanStep(kind=CallKind.THINK, duration=6.0, label="plan timeline"),
            PlanStep(kind=CallKind.WRITE, obj=obj, tool=plan, args=plan_args,
                     writes=(obj,), duration=4.0, label="write plan",
                     depends_on=("people",)),
        ])


def thanks_cell(variant: str) -> Cell:
    land = load(variant)
    initial: Dict[str, Any] = {
        PERSON + "mom": {"where": "home", "from": fmt(MOM_FROM), "car": "yes"},
        PERSON + "dad": {"where": "airport", "from": fmt(land["dad"]),
                         "car": "rental, at the airport"},
    }

    def build(oracle, rng, **kw):
        return [_planner(t, land, oracle, rng, kw) for t in TASKS]

    return Cell(
        name=f"realm-{variant}-thanksgiving", suite="realm",
        anomaly="double booking: read-pick-book of a person across a "
                "reasoning gap",
        description=f"REALM-Bench {variant.upper()} Thanksgiving: three task "
                    f"planners sharing mom's and dad's time",
        initial=initial, build_agents=build,
        anomaly_free=lambda final, refs: not conflicts(final),
        llm_spec={"tasks": [task_text(t, land) for t in TASKS],
                  "tools": TOOLS, "instance": variant},
        quality=lambda final: evaluate(variant, final),
    )


if P6_FILE.exists() and P9_DIR.exists():
    for _v in VARIANTS:
        _c = thanks_cell(_v)
        THANKS_CELLS[_c.name] = _c
