"""Workloads from idea.pdf: where chopping pays, and where it cannot.

The ten contended cells all have a read-modify-write or scan-then-insert
straddling the reasoning gap, so certification correctly merges them back to
whole-workflow locking and chopping buys nothing. These two cells are the
structures the approach is actually about.

``cc-claim`` is choppable: the shared dependency is short and the private
interval is long, so releasing the lock across reasoning is the whole win.

``cc-escalate`` is the impossibility witness of Corollary 1 made executable:
an agent commits a read, a peer commits a conflicting piece, and only then
does the agent discover it must write what the peer read. No regrouping of
the future can make that safe, and waiting does not help.
"""

from __future__ import annotations

from ..agents.scripted import PlanStep, ScriptedAgent
from ..core import Tool, WriteClass
from ..runtime import CallKind
from .base import Cell, register

POOL = "pool/count"
X, Y = "shared/x", "shared/y"


def _set(field: str) -> Tool:
    def _exec(value, args):
        out = dict(value or {})
        out[field] = args[field]
        return out

    def _rev(cur, cap, args):
        out = dict(cur or {})
        if isinstance(cap, dict) and field in cap:
            out[field] = cap[field]
        else:
            out.pop(field, None)
        return out

    return Tool(name=f"set_{field}", wclass=WriteClass.BLIND, exec_fn=_exec,
                reverse_fn=_rev, covers_fn=lambda a: frozenset({field}))


set_v = _set("v")


# -- cc-claim: the shape chopping exists for -------------------------------
def _claim_agents(oracle, rng, **kw):
    pf = kw.get("p_task_failure", 0.0)
    rtr = kw.get("p_redo_truncate", 0.0)

    def mk(name, out_key, bump, touch_pool):
        """``touch_pool`` is the short shared piece: a read or a write.

        One agent must write the pool, or there is no conflict at all and
        every policy looks identical.
        """
        return ScriptedAgent(
            name=name, oracle=oracle, rng=rng, p_task_failure=pf,
            p_redo_truncate=rtr,
            plan=[
                touch_pool,
                # the long private interval: this is what chopping releases
                PlanStep(kind=CallKind.THINK, duration=30.0, tokens=400,
                         label=f"{name} reason"),
                # private write, conflicts with nobody
                PlanStep(kind=CallKind.WRITE, obj=out_key, tool=set_v,
                         args=lambda view, b=bump: {
                             "v": str((view.get("pool") or {}).get("v", 0))
                                  + f"+{b}"},
                         writes=(out_key,), duration=3.0, tokens=300,
                         label=f"{name} write claim", depends_on=("pool",)),
            ],
        )

    a_touch = PlanStep(kind=CallKind.READ, obj=POOL, reads=(POOL,),
                       premise="pool", duration=2.0, tokens=300,
                       label="A read pool")
    b_touch = PlanStep(kind=CallKind.WRITE, obj=POOL, tool=set_v,
                       args={"v": 99}, writes=(POOL,), duration=2.0,
                       tokens=300, label="B write pool")
    return [mk("A-claim", "claim/a", 1, a_touch),
            mk("B-claim", "claim/b", 2, b_touch)]


register(Cell(
    name="cc-claim",
    suite="chopping",
    anomaly="none — short shared read, long private reasoning (choppable)",
    description=("Both agents read a shared pool, reason privately for a long "
                 "time, then write a private result. The shared dependency is "
                 "short and the private interval is long, so a certified "
                 "chopping releases the lock across reasoning."),
    initial={POOL: {"v": 10}, "claim/a": None, "claim/b": None},
    build_agents=_claim_agents,
    anomaly_free=lambda final, refs: any(
        all(final.get(k) == r.get(k) for k in ("claim/a", "claim/b"))
        for r in refs),
    llm_spec={
        "tasks": [
            f"Read the shared pool '{POOL}' to get its 'v'. Then write your "
            f"claim to 'claim/a', setting v to the string \"<the pool v you "
            f"read>+1\" (for example if the pool v is 10, write \"10+1\").",
            f"Write v=99 to the shared pool '{POOL}'. Then write your claim "
            f"to 'claim/b', setting v to the string \"0+2\".",
        ],
        "tools": [
            {"name": "get_object", "kind": "read",
             "description": "read one object's fields"},
            {"name": "set_v", "kind": "write", "tool": set_v,
             "args": {"v": "the new v value"},
             "description": "set an object's v (reversible)"},
        ],
    },
))


# -- cc-escalate: Corollary 1, executable ----------------------------------
def _escalate_agents(oracle, rng, **kw):
    pf = kw.get("p_task_failure", 0.0)
    rtr = kw.get("p_redo_truncate", 0.0)

    # A commits a read of x, then discovers mid-flight that it must write y.
    agent_a = ScriptedAgent(
        name="A-escalate", oracle=oracle, rng=rng, p_task_failure=pf,
        p_redo_truncate=rtr,
        plan=[
            PlanStep(kind=CallKind.READ, obj=X, reads=(X,), premise="x",
                     duration=3.0, tokens=320, label="read x"),
            PlanStep(kind=CallKind.THINK, duration=12.0, tokens=300,
                     label="discover the escalation"),
            # Undeclared on purpose: the footprint names x, not y. Reaching y
            # is the mid-flight escalation the certifier must rule on.
            # writes=() on purpose: the certified footprint names x, not
            # y. Reaching y is the mid-flight escalation, and the runtime
            # must discover it the way a real agent would -- at the access.
            PlanStep(kind=CallKind.WRITE, obj=Y, tool=set_v,
                     args=lambda view: {"v": "from-A"}, writes=(),
                     duration=4.0, tokens=320, label="escalate: write y",
                     depends_on=("x",)),
        ],
    )
    # B writes x and reads y in one piece, closing the cycle through A's
    # committed read.
    agent_b = ScriptedAgent(
        name="B-peer", oracle=oracle, rng=rng, p_task_failure=pf,
        p_redo_truncate=rtr,
        plan=[
            PlanStep(kind=CallKind.WRITE, obj=X, tool=set_v,
                     args={"v": "from-B"}, writes=(X,), duration=3.0,
                     tokens=300, label="write x"),
            PlanStep(kind=CallKind.READ, obj=Y, reads=(Y,), premise="y",
                     duration=3.0, tokens=300, label="read y"),
        ],
    )
    return [agent_a, agent_b]


register(Cell(
    name="cc-escalate",
    suite="chopping",
    anomaly="mid-flight escalation into a peer's committed footprint",
    description=("A commits a read of x; B commits a piece writing x and "
                 "reading y; only then does A discover it must write y. The "
                 "residual cycle runs through A's committed read, so no "
                 "future-only repair exists (Corollary 1)."),
    initial={X: {"v": "init"}, Y: {"v": "init"}},
    build_agents=_escalate_agents,
    # Safety, not task completion. A refused escalation leaves y untouched:
    # the state is serializable and A's intent is simply unmet, which the
    # paper treats as an application-level obligation (compensate or
    # replan), not a concurrency-control failure.
    anomaly_free=lambda final, refs: (
        (final.get(Y) or {}).get("v") == "init"
        or any(final == r for r in refs)),
    llm_spec={
        "tasks": [
            f"Read '{X}'. Then, having seen it, you must also write "
            f"v='from-A' to '{Y}'. If an access is refused, say so and stop "
            f"— do not try to work around it.",
            f"Write v='from-B' to '{X}'. Then read '{Y}'.",
        ],
        "tools": [
            {"name": "get_object", "kind": "read",
             "description": "read one object's fields"},
            {"name": "set_v", "kind": "write", "tool": set_v,
             "args": {"v": "the new v value"},
             "description": "set an object's v (reversible)"},
        ],
    },
))
