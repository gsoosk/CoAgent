"""Protocol policy interface, plus the two reference protocols.

``serial`` is the correctness and wall-clock-upper-bound reference; ``naive``
is the "probably correct" lower bound that ignores every dependency.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core import TargetSystem, WriteOp
from ..runtime import AgentState, Call, CallKind, ReadRecord, Runtime, Status


@dataclass
class Outcome:
    value: Any = None
    blocked: bool = False
    blocked_on: Optional[str] = None
    aborted: bool = False
    reason: str = ""


class Protocol:
    """Policy consulted by the middleware on every toolcall."""

    name = "base"
    detects_deadlock = False

    # -- dispatch ----------------------------------------------------------
    def dispatch(self, rt: Runtime, st: AgentState, call: Call) -> Outcome:
        if call.kind is CallKind.READ:
            return self.on_read(rt, st, call)
        return self.on_write(rt, st, call)

    def on_read(self, rt: Runtime, st: AgentState, call: Call) -> Outcome:
        raise NotImplementedError

    def on_write(self, rt: Runtime, st: AgentState, call: Call) -> Outcome:
        raise NotImplementedError

    def on_commit(self, rt: Runtime, st: AgentState) -> None:
        pass

    def on_abort(self, rt: Runtime, st: AgentState) -> None:
        pass

    def on_quiescence(self, rt: Runtime) -> bool:
        """Return True if the protocol re-opened an agent (run continues)."""
        return False

    # -- shared mechanics --------------------------------------------------
    @staticmethod
    def live_read(rt: Runtime, st: AgentState, call: Call) -> Any:
        if call.prefix is not None:
            names = rt.target.children(call.prefix)
            value = {n: rt.target.read_live(n) for n in names}
        else:
            value = rt.target.read_live(call.obj)
        key = call.prefix if call.prefix is not None else call.obj
        st.read_set[key] = ReadRecord(copy.deepcopy(value), st.step_idx)
        return value

    @staticmethod
    def make_write(rt: Runtime, st: AgentState, call: Call) -> WriteOp:
        return WriteOp(
            obj=call.obj,
            tool=call.tool,
            args=dict(call.args),
            agent=st.name,
            sigma=st.sigma,
            step=st.step_idx,
            t=rt.now,
            label=call.label,
        )

    @staticmethod
    def apply_and_record(rt: Runtime, st: AgentState, w: WriteOp) -> None:
        rt.target.apply_live(w)
        rt.target.traj(w.obj).insert(w)
        st.writes.append(w)


class NaiveProtocol(Protocol):
    """No coordination at all: read live, write live, hope for the best."""

    name = "naive"

    def on_read(self, rt, st, call):
        return Outcome(value=self.live_read(rt, st, call))

    def on_write(self, rt, st, call):
        w = self.make_write(rt, st, call)
        self.apply_and_record(rt, st, w)
        return Outcome(value=True)


class SerialProtocol(Protocol):
    """One agent at a time, in sigma order."""

    name = "serial"

    def __init__(self):
        self.active = 0

    def _gate(self, rt: Runtime, st: AgentState) -> Optional[Outcome]:
        if st.sigma != self.active:
            return Outcome(blocked=True, blocked_on=f"serial-turn-{self.active}")
        return None

    def on_read(self, rt, st, call):
        gate = self._gate(rt, st)
        if gate:
            return gate
        return Outcome(value=self.live_read(rt, st, call))

    def on_write(self, rt, st, call):
        gate = self._gate(rt, st)
        if gate:
            return gate
        w = self.make_write(rt, st, call)
        self.apply_and_record(rt, st, w)
        return Outcome(value=True)

    def on_commit(self, rt, st):
        if st.sigma == self.active:
            self.active += 1
            rt.wake_blocked()

    def on_quiescence(self, rt):
        pending = [s for s in rt.states.values() if s.status is Status.BLOCKED]
        if pending:
            rt.wake_blocked()
            return True
        return False
