"""REALM-Bench J1, job-shop scheduling, run by concurrent job schedulers.

REALM-Bench's JSSP problems (J1-J4, its "P11" leaderboard) ask one planner
for a complete schedule. Here each job has its own scheduler, and all of
them book time on the same machines at once -- the way a shop floor with
independent work-order planners operates. For each operation, in the job's
order, a scheduler reads the machine's bookings, picks the earliest start
that follows its previous operation and fits a free gap, and books it:

    read  jssp/machine/<m>       which slots are taken
    write jssp/machine/<m>.<job> book [start, end]
    ...   (one pair per operation)
    write jssp/job/<job>         the job's schedule

Two schedulers that read a machine before either books will pick the same
gap: a double booking, which makes the shop schedule infeasible. Unlike
P3, every job here needs every machine, so there is no private stretch for
certified chopping to release: certification should merge each job back
into one piece. That makes this the control for the P3 result.

Reused from REALM-Bench unchanged: the instance files (ABZ, Taillard and
DMU sets in datasets/J1), the job-shop constraints, and the metric classes.
The instances are cut to the first three jobs on machines 0-2, keeping each
job's machine order, so an LLM scheduler has three operations rather than
fifteen; the cut's optimal makespan is found by enumeration.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..agents.scripted import PlanStep, ScriptedAgent
from ..core import Tool, WriteClass
from ..runtime import CallKind
from .base import Cell
from .realm_urs import REALM_ROOT

__all__ = ["JSSP_CELLS", "load_cut", "optimal_makespan", "double_bookings",
           "evaluate"]

J1 = REALM_ROOT / "datasets" / "J1"
MACH, JOB = "jssp/machine/", "jssp/job/"
SOURCES = {"abz07": ("abzswvyn/abz07.txt", "pairs"),
           "ta01": ("TA/TA01.txt", "taillard"),
           "dmu-cscmax-20-15-1": ("DMU/cscmax_20_15_1.txt", "pairs")}
N_JOBS, N_MACHINES = 3, 3

JSSP_CELLS: Dict[str, Cell] = {}
Ops = List[Tuple[int, int]]                  # (machine, duration) in order


# -- instances -------------------------------------------------------------
def _parse(path: Path, fmt: str) -> List[Ops]:
    lines = [l.split() for l in path.read_text().splitlines() if l.strip()]
    if fmt == "pairs":                      # "n m", then machine/time pairs
        n = int(lines[0][0])
        return [[(int(r[i]), int(r[i + 1])) for i in range(0, len(r), 2)]
                for r in lines[1:1 + n]]
    n = int(lines[1][0])                    # Taillard: Times, then Machines
    t0 = next(i for i, l in enumerate(lines) if l[0] == "Times") + 1
    m0 = next(i for i, l in enumerate(lines) if l[0] == "Machines") + 1
    return [[(int(m) - 1, int(t)) for m, t in zip(lines[m0 + j], lines[t0 + j])]
            for j in range(n)]


def load_cut(name: str) -> List[Ops]:
    """First N_JOBS jobs, restricted to machines < N_MACHINES, order kept."""
    rel, fmt = SOURCES[name]
    path = J1 / rel
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/fetch_realm_bench.sh")
    return [[(m, d) for m, d in job if m < N_MACHINES]
            for job in _parse(path, fmt)[:N_JOBS]]


def _schedule(jobs: List[Ops], orders) -> Optional[Dict[Tuple[int, int], int]]:
    """Start times for given machine orders, or None if they deadlock."""
    start: Dict[Tuple[int, int], int] = {}
    pos = [0] * len(jobs)
    nxt = {m: 0 for m in range(N_MACHINES)}
    free = {m: 0 for m in range(N_MACHINES)}
    ready = [0] * len(jobs)
    total = sum(len(j) for j in jobs)
    while len(start) < total:
        moved = False
        for j, ops in enumerate(jobs):
            if pos[j] == len(ops):
                continue
            m, d = ops[pos[j]]
            if orders[m][nxt[m]] != j:
                continue
            s = max(ready[j], free[m])
            start[(j, pos[j])] = s
            ready[j] = free[m] = s + d
            pos[j] += 1
            nxt[m] += 1
            moved = True
        if not moved:
            return None
    return start


def optimal_makespan(jobs: List[Ops]) -> int:
    best = None
    perms = list(itertools.permutations(range(len(jobs))))
    for orders in itertools.product(perms, repeat=N_MACHINES):
        st = _schedule(jobs, orders)
        if st is None:
            continue
        span = max(st[(j, i)] + jobs[j][i][1] for j, i in st)
        best = span if best is None else min(best, span)
    return best or 0


# -- tools -----------------------------------------------------------------
def _set_slot(v, a):
    out = dict(v or {})
    out[a["job"]] = [a.get("start"), a.get("end")]
    return out


def _clear_slot(v, a):
    out = dict(v or {})
    out.pop(a.get("job"), None)
    return out


book = Tool(name="book_slot", wclass=WriteClass.BLIND, exec_fn=_set_slot,
            covers_fn=lambda a: frozenset({str(a.get("job"))}))
cancel = Tool(name="cancel_slot", wclass=WriteClass.BLIND,
              exec_fn=_clear_slot,
              covers_fn=lambda a: frozenset({str(a.get("job"))}))
record = Tool(name="write_schedule", wclass=WriteClass.BLIND,
              exec_fn=lambda v, a: {"ops": a.get("ops")})

TOOLS = [
    {"name": "get_machine", "kind": "read",
     "description": "read a machine's booked slots {job: [start, end]}; "
                    "obj is 'jssp/machine/<m>'"},
    {"name": "book_slot", "kind": "write", "tool": book,
     "args": {"job": "your job id", "start": "start time",
              "end": "end time"},
     "description": "book [start, end] on a machine for your job (replaces "
                    "your job's slot there); obj is 'jssp/machine/<m>'"},
    {"name": "cancel_slot", "kind": "write", "tool": cancel,
     "args": {"job": "your job id"},
     "description": "remove your job's slot from a machine"},
    {"name": "write_schedule", "kind": "write", "tool": record,
     "args": {"ops": "list of [machine, start, end] in your job's order"},
     "description": "write your job's final schedule; obj is "
                    "'jssp/job/<your job id>'"},
]


def task_text(jid: str, ops: Ops) -> str:
    steps = "; ".join(f"{i + 1}) machine m{m} for {d}"
                      for i, (m, d) in enumerate(ops))
    return (
        f"You schedule job '{jid}' in a job shop. Its operations must run in "
        f"this order: {steps}. Other jobs' schedulers are booking the same "
        f"machines at the same time.\n\n"
        f"A machine runs one operation at a time, and an operation may start "
        f"only when your previous operation has ended (time starts at 0). "
        f"For each operation, in order: read the machine with get_machine, "
        f"pick the earliest start that is at or after your previous "
        f"operation's end and does not overlap any booked slot, and book it "
        f"with book_slot (job='{jid}'). When all are booked, write your "
        f"schedule with write_schedule, obj 'jssp/job/{jid}'.\n\n"
        f"Bookings are first-come. If you learn that a machine you booked "
        f"has a slot overlapping yours that was booked before yours, move "
        f"yours: book the next free start, and move your later operations "
        f"too if they no longer start after it ends."
    )


# -- scoring ---------------------------------------------------------------
def _slots(final: Dict[str, Any], m: int) -> Dict[str, Any]:
    v = final.get(f"{MACH}m{m}")
    return v if isinstance(v, dict) else {}


def _span(s) -> Optional[Tuple[float, float]]:
    try:
        a, b = float(s[0]), float(s[1])
    except (TypeError, ValueError, IndexError):
        return None
    return (a, b) if b > a else None


def double_bookings(final: Dict[str, Any]) -> int:
    """Pairs of jobs whose slots overlap on some machine."""
    n = 0
    for m in range(N_MACHINES):
        spans = [sp for sp in map(_span, _slots(final, m).values())
                 if sp is not None]
        for a, b in itertools.combinations(spans, 2):
            if a[0] < b[1] and b[0] < a[1]:
                n += 1
    return n


def evaluate(name: str, jobs: List[Ops], final: Dict[str, Any]) -> Dict[str, Any]:
    from .realm_eval import _realm_metrics
    mm = _realm_metrics()
    jids = [f"j{j}" for j in range(len(jobs))]
    constraints: List[Dict[str, Any]] = []
    completion: Dict[str, float] = {}
    capacity: Dict[str, int] = {}
    complete = []
    ends = []
    for j, ops in enumerate(jobs):
        spans: List[Any] = [_span(_slots(final, m).get(jids[j])) for m, _ in ops]
        ok = all(spans) and all(
            sp[1] - sp[0] == d for sp, (_, d) in zip(spans, ops))
        for i in range(len(ops) - 1):
            pid, sid = f"{jids[j]}-op{i}-end", f"{jids[j]}-op{i + 1}-start"
            constraints.append({"type": "dependency", "id": f"{pid}<{sid}",
                                "predecessor": pid, "successor": sid})
            if spans[i] and spans[i + 1]:
                completion[pid], completion[sid] = spans[i][1], spans[i + 1][0]
                ok = ok and spans[i][1] <= spans[i + 1][0]
        # a machine cannot run two operations at once: an overlapped job is
        # not scheduled, however good the rest of it is
        for (m, _), sp in zip(ops, spans):
            others = [_span(o) for k, o in _slots(final, m).items()
                      if k != jids[j]]
            if sp and any(o and o[0] < sp[1] and sp[0] < o[1] for o in others):
                ok = False
        if ok:
            complete.append(jids[j])
        ends += [{"task_id": f"{jids[j]}-{i}", "end_time": sp[1]}
                 for i, sp in enumerate(spans) if sp]
    for m in range(N_MACHINES):
        spans = [s for s in map(_span, _slots(final, m).values()) if s]
        peak: int = max((sum(a[0] <= t < a[1] for a in spans)
                    for t in {s[0] for s in spans}), default=0)
        capacity[f"m{m}"] = peak
        constraints.append({"type": "capacity", "id": f"m{m}-capacity",
                            "capacity_id": f"m{m}", "max_capacity": 1})
    results = {"task_completion_times": completion,
               "capacity_usage": capacity}
    quality = mm.PlanningQualityMetrics().evaluate_goal_satisfaction(
        [f"schedule {j}" for j in jids], [f"schedule {j}" for j in complete])
    csr = mm.ConstraintSatisfactionMetrics().evaluate_constraint_satisfaction(
        constraints, results)
    opt = optimal_makespan(jobs)
    feasible = len(complete) == len(jobs) and double_bookings(final) == 0
    span = mm.PlanningOptimalityMetrics().evaluate_makespan(
        ends, optimal_makespan=opt) if feasible else None
    return {
        "goal_satisfaction": quality.value,
        "constraint_satisfaction": csr.value,
        "makespan_optimality": span.value if span else None,
        "optimal_makespan": opt,
        "double_bookings": double_bookings(final),
        "causes": ["double booking"] * double_bookings(final),
    }


# -- scripted schedulers ----------------------------------------------------
def _times(ops: Ops, jid: str, view: Dict[str, Any], upto: int):
    """Earliest feasible slots for ops[0..upto], from the machines as read."""
    t, out = 0, []
    for i in range(upto + 1):
        m, d = ops[i]
        taken = sorted(s for k, s in ((view.get(f"op{i}") or {}).items())
                       if k != jid and _span(s))
        start = t
        for a, b in taken:
            if start < b and a < start + d:
                start = max(start, b)
        out.append((start, start + d))
        t = start + d
    return out


def _job(jid: str, ops: Ops, oracle, rng, kw) -> ScriptedAgent:
    plan: List[PlanStep] = []
    for i, (m, d) in enumerate(ops):
        obj = f"{MACH}m{m}"
        prem = tuple(f"op{k}" for k in range(i + 1))

        def args(view, i=i):
            s, e = _times(ops, jid, view, i)[i]
            return {"job": jid, "start": s, "end": e}

        plan += [
            PlanStep(kind=CallKind.READ, obj=obj, reads=(obj,),
                     premise=f"op{i}", duration=3.0, label=f"read m{m}"),
            PlanStep(kind=CallKind.THINK, duration=3.0, label="pick a gap"),
            PlanStep(kind=CallKind.WRITE, obj=obj, tool=book, args=args,
                     writes=(obj,), duration=3.0, label=f"book m{m}",
                     depends_on=prem),
        ]
    sched = JOB + jid
    plan.append(PlanStep(
        kind=CallKind.WRITE, obj=sched, tool=record, writes=(sched,),
        args=lambda view: {"ops": [[m, s, e] for (m, _), (s, e) in zip(
            ops, _times(ops, jid, view, len(ops) - 1))]},
        duration=3.0, label="write schedule",
        depends_on=tuple(f"op{k}" for k in range(len(ops)))))
    return ScriptedAgent(name=jid, oracle=oracle, rng=rng, plan=plan,
                         p_task_failure=kw.get("p_task_failure", 0.0),
                         p_redo_truncate=kw.get("p_redo_truncate", 0.0))


def jssp_cell(name: str) -> Cell:
    jobs = load_cut(name)
    jids = [f"j{j}" for j in range(len(jobs))]
    initial: Dict[str, Any] = {f"{MACH}m{m}": {} for m in range(N_MACHINES)}

    def build(oracle, rng, **kw):
        return [_job(jid, ops, oracle, rng, kw) for jid, ops in zip(jids, jobs)]

    return Cell(
        name=f"realm-jssp-{name}", suite="realm",
        anomaly="double booking: read-pick-book of a machine gap across a "
                "reasoning gap",
        description=f"REALM-Bench J1 {name}, first {N_JOBS} jobs on machines "
                    f"0-{N_MACHINES - 1}: one scheduler per job",
        initial=initial, build_agents=build,
        anomaly_free=lambda final, refs: double_bookings(final) == 0,
        llm_spec={"tasks": [task_text(j, o) for j, o in zip(jids, jobs)],
                  "tools": TOOLS, "instance": name},
        quality=lambda final: evaluate(name, jobs, final),
    )


if J1.exists():
    for _n in SOURCES:
        _c = jssp_cell(_n)
        JSSP_CELLS[_c.name] = _c
