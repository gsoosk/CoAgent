"""Launch-time order negotiation for model-driven agents.

The certifier decides how a plan may be cut, but it takes the order of each
agent's steps as given. When a plan fails certification, the steps of an
agent that conflict with another agent must end up in one piece, together
with everything between them. Whether anything is between them is a property
of the order, and the order is the agent's to choose.

So before launch, an agent whose plan failed certification is shown its own
steps, told which of them are contended, and asked for an order that gives
the same final result with the contended steps next to each other.

Trust boundary. The agent chooses an order; it does not choose footprints
(those still come from the workload) and it does not decide what is safe.
The certifier checks the reordered plan exactly as it checks any other, and
enforcement stays fail-closed, so a wrong answer cannot break isolation. It
can break the agent's own task: an order that moves a step ahead of
something it needs is the agent's mistake, and shows up as task failure.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Set

from ..chopping import Piece, Plan, certify, sc_edges
from ..runtime import AgentState, Call, CallKind, Runtime
from .llm import LLMAgent

__all__ = ["PlanningLLMAgent", "negotiate_order"]

PLAN_SYSTEM = """You are one of several agents working concurrently on the \
same shared system. Before anyone starts, the runtime is deciding how long \
each agent must hold locks.

You will be shown the steps of your own task. Steps marked SHARED touch \
objects another agent also uses. The runtime must hold your locks from your \
first SHARED step to your last SHARED step, including every step in between. \
Steps that are not marked run without blocking anyone.

Choose the order in which you will do your steps. Keep the final result of \
your task exactly the same. If you can, put the SHARED steps next to each \
other and move the other steps before or after them. Never move a step \
ahead of a step whose result it needs. If no other order is safe, keep the \
order you were given.

Reply with exactly one JSON object and nothing else:
  {"thought": "...", "order": [<step numbers, each exactly once>]}"""


def _fp_text(fp) -> str:
    parts = []
    for a in sorted(fp, key=lambda a: (a.lo, a.is_write)):
        key = a.lo if a.hi == a.lo + "\x00" else a.lo + "*"
        parts.append(("writes " if a.is_write else "reads ") + key)
    return ", ".join(parts) or "touches nothing"


class PlanningLLMAgent(LLMAgent):
    """An LLM Worker that can be asked for the order of its declared steps."""

    def __init__(self, *args: Any, **kw: Any):
        super().__init__(*args, **kw)
        self.order: List[int] = list(range(len(self._pieces)))
        self.plan_asked = False
        self.plan_valid = True
        self.plan_latency = 0.0
        self.plan_thought = ""
        self._plan_charged = False

    # -- negotiation -------------------------------------------------------
    def propose_order(self, shared: Set[int]) -> None:
        """One model call: choose an order for the declared steps."""
        n = len(self._pieces)
        lines = [
            f"{i + 1}. {label} [{_fp_text(fp)}]"
            + ("  SHARED" if i in shared else "")
            for i, (fp, label) in enumerate(self._pieces)
        ]
        prompt = (f"Your task:\n{self.task}\n\nYour steps, in the order "
                  f"given:\n" + "\n".join(lines))
        out = self.backend.complete(PLAN_SYSTEM,
                                    [{"role": "user", "content": prompt}])
        self.prompt_tokens += out.prompt_tokens
        self.completion_tokens += out.completion_tokens
        self.plan_asked = True
        self.plan_latency = out.latency
        self.plan_thought = str(out.parsed.get("thought", ""))
        self.transcript.append({"agent": self.name, "reply": out.parsed,
                                "latency": out.latency, "phase": "plan"})
        raw = out.parsed.get("order")
        try:
            order = [int(x) - 1 for x in raw]
        except (TypeError, ValueError):
            order = []
        if sorted(order) == list(range(n)):
            self.order = order
        else:
            # not a permutation of the steps: keep the given order
            self.plan_valid = False

    @property
    def reordered(self) -> bool:
        return self.order != list(range(len(self._pieces)))

    def declare_pieces(self):
        return [self._pieces[i] for i in self.order]

    # -- execution ---------------------------------------------------------
    def _reset(self) -> None:
        task = self.task
        if self.reordered:
            steps = "\n".join(f"{k + 1}. {self._pieces[i][1]}"
                              for k, i in enumerate(self.order))
            task += ("\n\nBefore starting you agreed with the runtime to do "
                     "your steps in this order, and locks were arranged for "
                     f"it:\n{steps}\nFollow this order.")
        self.messages = [{"role": "user", "content": task}]

    def run(self, rt: Runtime, st: AgentState):
        self._reset()
        if self.plan_latency and not self._plan_charged:
            # the planning call is real time the agent spent before acting
            self._plan_charged = True
            yield Call(kind=CallKind.THINK, duration=self.plan_latency,
                       tokens=0, label="plan order")
        yield from self._loop(rt, st, self.max_rounds)


def negotiate_order(agents: Sequence[Any]) -> Dict[str, Any]:
    """Ask each agent with contended steps for an order, if the plan fails.

    A plan that certifies as declared is left alone: there is nothing to
    gain, and asking would only add a model call.
    """
    plan = Plan({
        a.name: [Piece(a.name, i, frozenset(fp), label)
                 for i, (fp, label) in enumerate(a.declare_pieces())]
        or [Piece(a.name, 0, frozenset(), "whole")]
        for a in agents
    })
    ok, _ = certify(plan)
    report: Dict[str, Any] = {"certified_as_declared": ok, "agents": {}}
    if ok:
        return report
    shared: Dict[str, Set[int]] = {a.name: set() for a in agents}
    for u, v, lab in sc_edges(plan):
        if lab == "C":
            shared[u[0]].add(u[1])
            shared[v[0]].add(v[1])
    for a in agents:
        if not isinstance(a, PlanningLLMAgent) or len(a._pieces) < 2:
            continue
        idx = shared[a.name]
        if not idx:
            continue
        a.propose_order(idx)
        report["agents"][a.name] = {
            "shared": sorted(idx),
            # could any order shorten the span that must be merged?
            "gap_inside_span": any(i not in idx
                                   for i in range(min(idx), max(idx) + 1)),
            "order": list(a.order), "reordered": a.reordered,
            "valid": a.plan_valid, "latency": a.plan_latency,
            "thought": a.plan_thought,
        }
    return report
