"""A non-contended control pair.

Every cost the classical protocols pay in the matrix should come from real
contention, not from the middleware. Two agents whose footprints are disjoint
must therefore run at full concurrency under 2PL, OCC and MTPO alike, with no
deadlock, no abort and no token overhead. If that is not true, the baselines
are strawmanned and the contended numbers mean nothing.
"""

from __future__ import annotations

from ..agents.scripted import PlanStep, ScriptedAgent
from ..runtime import CallKind
from .base import Cell
from .patterns import _blind

_set = _blind("v")


def disjoint_pair() -> Cell:
    def build(oracle, rng, **kw):
        def mk(name, obj):
            return ScriptedAgent(
                name=name, oracle=oracle, rng=rng,
                p_task_failure=kw.get("p_task_failure", 0.0),
                plan=[
                    PlanStep(kind=CallKind.READ, obj=obj, reads=(obj,),
                             premise="c", duration=4.0, tokens=400,
                             label=f"read {obj}"),
                    PlanStep(kind=CallKind.THINK, duration=3.0, tokens=260,
                             label="plan"),
                    PlanStep(kind=CallKind.WRITE, obj=obj, tool=_set,
                             args={"v": name}, writes=(obj,), duration=6.0,
                             tokens=320, label=f"write {obj}",
                             depends_on=("c",)),
                ],
            )

        return [mk("A", "ns/a"), mk("B", "ns/b")]

    return Cell(
        name="control-disjoint",
        suite="control",
        anomaly="none (disjoint footprints)",
        description="Two agents that never touch the same object.",
        initial={"ns/a": {"v": 0}, "ns/b": {"v": 0}},
        build_agents=build,
    )
