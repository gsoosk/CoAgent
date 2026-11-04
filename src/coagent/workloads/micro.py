"""Micro-benchmark for section 5.2: notification alone is not enough.

Task pair  x <- y/2  ||  y <- x/2.  Either serial order terminates, at
(0.5, 0.25) or (0.25, 0.5). Under unconditional broadcast every write
invalidates the other agent's premise and every heal issues a new write, so
x and y walk to zero and the run never terminates. MTPO's pre-order orients
the notifications and the pair converges in one extra step.
"""

from __future__ import annotations

from typing import Any, Dict

from ..agents.scripted import PlanStep, ScriptedAgent
from ..core import Tool, WriteClass
from ..runtime import CallKind
from .base import Cell

set_v = Tool(
    name="set_v",
    wclass=WriteClass.BLIND,
    exec_fn=lambda value, args: dict(value or {}, v=args["v"]),
)


def halving_pair() -> Cell:
    def build(oracle, rng, **kw):
        def mk(name, src, dst):
            return ScriptedAgent(
                name=name, oracle=oracle, rng=rng,
                p_task_failure=kw.get("p_task_failure", 0.0),
                p_redo_truncate=kw.get("p_redo_truncate", 0.0),
                plan=[
                    PlanStep(kind=CallKind.READ, obj=src, reads=(src,),
                             premise="src", duration=1.0, tokens=200,
                             label=f"read {src}"),
                    PlanStep(
                        kind=CallKind.WRITE, obj=dst, tool=set_v,
                        args=lambda view: {
                            "v": (view.get("src") or {}).get("v", 1.0) / 2.0
                        },
                        writes=(dst,), duration=1.0, tokens=200,
                        label=f"{dst} <- {src}/2", depends_on=("src",),
                    ),
                ],
            )

        return [mk("A1", "y", "x"), mk("A2", "x", "y")]

    return Cell(
        name="micro-halving",
        suite="micro",
        anomaly="symmetric antidependency cycle (livelocks under broadcast)",
        description="x <- y/2 || y <- x/2",
        initial={"x": {"v": 1.0}, "y": {"v": 1.0}},
        build_agents=build,
    )
