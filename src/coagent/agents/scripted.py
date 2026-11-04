"""Deterministic scripted agents with an explicit premise graph.

A real Worker's plan lives in an LLM's context; here it is a list of steps
plus the premises each step rests on. That makes the paper's assumptions
testable rather than implicit:

  A1 individual success   -> ``p_task_failure`` (why serial is 98%, not 100%)
  A2 well-formedness      -> every step declares its footprint, and the inbox
                             is drained before the next action
  A3 self-healing         -> ``JudgmentOracle`` decides whether a notification
                             is a real premise conflict, and gets it wrong
                             with probability ``p_misjudge``
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..core import Footprint, Tool, covered_by, is_descendant
from ..runtime import AgentState, Call, CallKind, Runtime

# tokens a delivered notification adds to the receiver's context
NOTE_TOKENS = 180


class JudgmentOracle:
    """Stands in for the LLM's judgement on a delivered notification."""

    def __init__(self, rng, p_misjudge: float = 0.05):
        self.rng = rng
        self.p_misjudge = p_misjudge
        self.misjudgments = 0

    def is_relevant(self, truly_relevant: bool) -> bool:
        if self.rng.random() < self.p_misjudge:
            self.misjudgments += 1
            return not truly_relevant
        return truly_relevant


@dataclass
class PlanStep:
    """One step of an agent's plan."""

    kind: CallKind
    obj: str = ""
    prefix: Optional[str] = None
    tool: Optional[Tool] = None
    # args may be a dict, or a callable of the agent's current view
    args: Any = field(default_factory=dict)
    reads: Sequence[str] = ()
    writes: Sequence[str] = ()
    duration: float = 2.0
    tokens: int = 320
    label: str = ""
    # name under which a read's value enters the agent's view
    premise: Optional[str] = None
    # name under which this call's resolved args enter the view, so a later
    # repair can retract exactly what this step wrote
    record_as: Optional[str] = None
    # premises this step's correctness rests on
    depends_on: Sequence[str] = ()
    # steps re-issued when a premise is invalidated; defaults to this step
    repairable: bool = True
    # a plan whose shape depends on what an earlier read returned:
    # expand(view) -> list of concrete PlanSteps
    expand: Optional[Callable[[Dict[str, Any]], List["PlanStep"]]] = None
    # what to re-issue when a premise is invalidated. The paper's point: a
    # notified agent patches the affected operation, it does not redo itself,
    # so B's stale canary is fixed by one set_image, not by a re-create.
    repair_with: Optional[List["PlanStep"]] = None

    def resolve_args(self, view: Dict[str, Any]) -> Dict[str, Any]:
        if callable(self.args):
            return self.args(view)
        return dict(self.args)


class ScriptedAgent:
    """An agent that executes a fixed plan and repairs on notification."""

    def __init__(
        self,
        name: str,
        plan: List[PlanStep],
        oracle: JudgmentOracle,
        premise_sources: Optional[Dict[str, str]] = None,
        p_task_failure: float = 0.0,
        rng: Any = None,
    ):
        self.name = name
        self.plan = plan
        self.oracle = oracle
        self.rng = rng
        self.p_task_failure = p_task_failure
        # premise key -> object it was read from
        self.premise_sources: Dict[str, str] = premise_sources or {}
        for s in plan:
            if s.premise:
                self.premise_sources.setdefault(s.premise, s.prefix or s.obj)
        self.view: Dict[str, Any] = {}
        self.executed: set = set()
        # Rolled once per agent. Rolling it inside run() re-rolled it on every
        # abort, so the abort-heavy protocols absorbed strictly more A1
        # failures than MTPO at the same nominal rate.
        self.failed_alone = bool(
            rng is not None and p_task_failure > 0
            and rng.random() < p_task_failure
        )

    # -- call construction -------------------------------------------------
    def _call(self, step: PlanStep) -> Call:
        return Call(
            kind=step.kind,
            obj=step.obj,
            tool=step.tool,
            args=step.resolve_args(self.view),
            footprint=Footprint.of(step.reads, step.writes),
            duration=step.duration,
            tokens=step.tokens,
            label=step.label or step.obj,
            prefix=step.prefix,
        )

    # -- notification handling --------------------------------------------
    def _affected_premises(self, obj: str) -> List[str]:
        out = []
        for key, src in self.premise_sources.items():
            if key not in self.view:
                continue
            if covered_by(obj, src) or covered_by(src, obj):
                out.append(key)
        return out

    def _premise_actually_changed(self, st: AgentState, note, key: str) -> bool:
        """Ground truth the oracle is asked to reproduce."""
        src = self.premise_sources.get(key, note.obj)
        old = self.view.get(key)
        if src == note.obj:
            return old != note.new_value
        # range premise: the refreshed subtree differs from what we recorded
        return True

    def _refresh_premise(self, key: str, note) -> None:
        """Fold a notification into the view.

        A premise read from a range holds a map of children, so a notification
        naming one child refreshes that entry rather than replacing the map.
        """
        src = self.premise_sources.get(key)
        current = self.view.get(key)
        # Only a *strict* descendant refreshes one entry of a range premise;
        # a notification naming the premise object itself replaces it.
        if (
            src is not None
            and is_descendant(note.obj, src)
            and isinstance(current, dict)
        ):
            updated = dict(current)
            updated[note.obj] = note.new_value
            self.view[key] = updated
        else:
            self.view[key] = note.new_value

    def _handle_notifications(self, rt: Runtime, st: AgentState):
        """Consume the inbox before the next action (A2), repairing if needed."""
        steps: List[PlanStep] = []
        while st.inbox:
            note = st.inbox.pop(0)
            st.notes_seen += 1
            # The notification is injected into the agent's context, so the
            # next inference pays for it. This is where MTPO's token overhead
            # over serial comes from.
            st.context_tokens += NOTE_TOKENS
            keys = self._affected_premises(note.obj)
            truly = any(self._premise_actually_changed(st, note, k) for k in keys)
            if not self.oracle.is_relevant(truly):
                rt.trace(f"{self.name} judges notification on {note.obj} irrelevant")
                continue
            if not keys:
                continue
            rt.trace(f"{self.name} repairs after notification on {note.obj}")
            st.repairs += 1
            for k in keys:
                self._refresh_premise(k, note)
            # Re-issue exactly the steps that depended on the changed
            # premise -- but only those that already ran. A step still pending
            # needs no repair: it will simply use the refreshed premise when
            # it executes. (Section 5.3: keep, rewrite, or retract *pending*
            # operations.)
            for s in self.plan:
                if not s.repairable or id(s) not in self.executed:
                    continue
                if s.tool is not None and not s.tool.reversible:
                    continue      # re-firing an unrecoverable write is worse
                if any(k in s.depends_on for k in keys):
                    steps.extend(s.repair_with if s.repair_with else [s])
        return steps

    # -- execution ---------------------------------------------------------
    def run(self, rt: Runtime, st: AgentState):
        """Generator of calls; ends only when the plan is done and inbox empty."""
        self.view = {}
        self.executed = set()
        i = 0
        while i < len(self.plan):
            for rep in self._handle_notifications(rt, st):
                yield from self._exec(rt, st, rep)
            step = self.plan[i]
            i += 1
            if self.failed_alone and step.kind is CallKind.WRITE:
                # A1 violation: the agent fails on its own, writes nothing more.
                rt.trace(f"{self.name} individual-success failure (A1)")
                return
            yield from self._exec(rt, st, step)
        # drain anything that landed during the final steps
        for rep in self._handle_notifications(rt, st):
            yield from self._exec(rt, st, rep)

    def repair_run(self, rt: Runtime, st: AgentState):
        """Re-opened after quiescence by a late notification (section 5.3)."""
        for rep in self._handle_notifications(rt, st):
            yield from self._exec(rt, st, rep)

    def _exec(self, rt: Runtime, st: AgentState, step: PlanStep):
        if step.expand is not None:
            # Record the parent, not the generated children: a repair re-runs
            # the expansion against the refreshed view, which is what makes a
            # sweep agent able to heal at all.
            self.executed.add(id(step))
            for sub in step.expand(self.view):
                yield from self._exec(rt, st, sub)
            return
        self.executed.add(id(step))
        call = self._call(step)
        if step.record_as:
            self.view[step.record_as] = call.args
        result = yield call
        if step.premise:
            self.view[step.premise] = result
        elif step.kind is CallKind.READ:
            self.view[step.obj or step.prefix or "last"] = result
