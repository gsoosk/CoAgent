"""The toolcall middleware and its discrete-event scheduler (paper section 6).

Every Worker toolcall passes through ``Runtime``: the middleware injects the
caller's pending notifications, hands the call to the active protocol policy,
dispatches it, and returns the result. Time is virtual, so a whole protocol
matrix runs deterministically in seconds.
"""

from __future__ import annotations

import copy
import heapq
import itertools
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Tuple

from .core import Footprint, TargetSystem, Tool, WriteOp

logger = logging.getLogger(__name__)


class CallKind(Enum):
    READ = "read"
    WRITE = "write"
    THINK = "think"


@dataclass
class Call:
    """One tool call emitted by an agent."""

    kind: CallKind
    obj: str = ""
    tool: Optional[Tool] = None
    args: Dict[str, Any] = field(default_factory=dict)
    footprint: Footprint = field(default_factory=Footprint)
    duration: float = 1.0      # seconds of inference + tool latency
    tokens: int = 300          # tokens this step appends to the context
    label: str = ""
    # Range reads (`list deployments`) read a subtree rather than one object.
    prefix: Optional[str] = None

    def describe(self) -> str:
        return self.label or f"{self.kind.value}:{self.obj}"


@dataclass
class Notification:
    """nu_{i->j}(o): a one-way, read-only push carrying the sigma-legal value."""

    obj: str
    new_value: Any
    from_agent: str
    to_agent: str
    t: float
    reason: str = "write"


class Status(Enum):
    RUNNING = "running"
    BLOCKED = "blocked"
    DONE = "done"
    FAILED = "failed"


@dataclass
class ReadRecord:
    """What a read returned, and where in the agent's program order it sat."""

    value: Any
    step: int


@dataclass
class AgentState:
    """Per-agent bookkeeping: rank, context, read set, pending notifications."""

    name: str
    sigma: int
    agent: Any                      # the Agent object (makes fresh generators)
    gen: Optional[Generator] = None
    ready_at: float = 0.0
    status: Status = Status.RUNNING
    step_idx: int = 0
    # context model: append-only, so each inference re-reads everything before it
    context_tokens: int = 500
    billed_tokens: int = 0
    # object/prefix -> ReadRecord(value, step): the step matters because a
    # notification must refresh the premise at the position it was read
    read_set: Dict[str, Any] = field(default_factory=dict)
    writes: List[WriteOp] = field(default_factory=list)
    inbox: List[Notification] = field(default_factory=list)
    notes_seen: int = 0
    repairs: int = 0
    restarts: int = 0
    pending_send: Any = None        # value handed back into the generator
    pending_call: Any = None        # a blocked call, held for re-dispatch
    blocked_on: Optional[str] = None
    speed: float = 1.0              # per-agent task-length multiplier
    finished_at: float = 0.0
    locks: set = field(default_factory=set)

    def bill(self, step_tokens: int) -> None:
        """One inference pass bills the whole context, then appends to it."""
        self.billed_tokens += self.context_tokens
        self.context_tokens += step_tokens

    def reset_context(self) -> None:
        """Abort / deadlock victim: the context is cleared before restart."""
        self.context_tokens = 500
        self.read_set.clear()
        self.inbox.clear()
        self.step_idx = 0
        self.pending_send = None
        self.pending_call = None
        self.restarts += 1


@dataclass
class TrialResult:
    protocol: str
    cell: str
    seed: int
    wall_clock: float
    tokens: int
    invariant_ok: bool
    deadlocks: int = 0
    aborts: int = 0
    notifications: int = 0
    repairs: int = 0
    undos: int = 0
    livelocked: bool = False
    # live copy == materialization of the full trajectory (MTPO's invariant)
    trajectory_match: bool = True
    final_state: Dict[str, Any] = field(default_factory=dict)
    note: str = ""


class Runtime:
    """Scheduler + middleware. The protocol supplies the policy."""

    MAX_EVENTS = 20000

    def __init__(
        self,
        target: TargetSystem,
        agents: List[Any],
        protocol: Any,
        rng: Any,
        retry_cap: int = 5,
    ):
        self.target = target
        self.protocol = protocol
        self.rng = rng
        self.retry_cap = retry_cap
        self.now = 0.0
        self.jitter_sigma = 0.12
        self.time_scale = 1.0
        self.states: Dict[str, AgentState] = {}
        for sigma, a in enumerate(agents):
            self.states[a.name] = AgentState(name=a.name, sigma=sigma, agent=a)
        self.order = [a.name for a in agents]
        self.deadlocks = 0
        self.aborts = 0
        self.notifications = 0
        self.livelocked = False
        self.events: List[Tuple[float, int, str]] = []
        self._trace: List[str] = []

    # -- helpers -----------------------------------------------------------
    def state(self, name: str) -> AgentState:
        return self.states[name]

    def others(self, name: str) -> List[AgentState]:
        return [s for n, s in self.states.items() if n != name]

    def trace(self, msg: str) -> None:
        self._trace.append(f"[{self.now:7.2f}s] {msg}")

    @property
    def trace_lines(self) -> List[str]:
        return list(self._trace)

    def notify(self, note: Notification) -> None:
        tgt = self.states[note.to_agent]
        if tgt.status is Status.DONE:
            # A quiescent receiver is re-opened (section 5.3).
            tgt.status = Status.RUNNING
            tgt.ready_at = max(tgt.ready_at, self.now)
            # a fresh generator must not be resumed with a stale sent value
            tgt.pending_send = None
            tgt.pending_call = None
            tgt.gen = tgt.agent.repair_run(self, tgt)
        tgt.inbox.append(note)
        self.notifications += 1
        self.trace(f"notify {note.from_agent}->{note.to_agent} on {note.obj}")

    # -- main loop ---------------------------------------------------------
    def run(self) -> None:
        for st in self.states.values():
            st.gen = st.agent.run(self, st)
            st.pending_send = None
            st.pending_call = None
        guard = 0
        while True:
            guard += 1
            if guard > self.MAX_EVENTS:
                self.livelocked = True
                self.trace("ABORT: event cap hit (livelock)")
                return
            runnable = [
                s for s in self.states.values()
                if s.status is Status.RUNNING
            ]
            if not runnable:
                if self.protocol.on_quiescence(self):
                    continue          # protocol re-opened somebody
                return
            st = min(runnable, key=lambda s: (s.ready_at, s.sigma))
            self.now = max(self.now, st.ready_at)
            self._advance(st)

    def _advance(self, st: AgentState) -> None:
        """Pull one call out of an agent and run it through the middleware.

        A call that the protocol blocked is held on the agent and re-dispatched
        when it wakes; it must not be skipped, and it must not be billed twice.
        """
        if st.pending_call is not None and not st.inbox:
            call = st.pending_call
        else:
            if st.pending_call is not None:
                # A notification arrived while this call was blocked. Its
                # arguments were resolved against a view the runtime now knows
                # is stale, so drop it and let the agent re-resolve them.
                st.pending_call = None
                st.step_idx -= 1
            try:
                call = st.gen.send(st.pending_send)
            except StopIteration:
                st.status = Status.DONE
                st.finished_at = self.now
                self.protocol.on_commit(self, st)
                return
            st.pending_send = None
            st.step_idx += 1
            st.bill(call.tokens)


        dur = call.duration * self.time_scale * st.speed * self._jitter()
        if call.kind is CallKind.THINK:
            st.pending_call = None
            st.ready_at = self.now + dur
            return

        outcome = self.protocol.dispatch(self, st, call)
        if outcome.blocked:
            st.pending_call = call
            st.status = Status.BLOCKED
            st.blocked_on = outcome.blocked_on
            self.trace(f"{st.name} blocked on {outcome.blocked_on}")
            self._check_deadlock()
            return
        if outcome.aborted:
            self._abort(st, outcome.reason)
            return
        st.pending_call = None
        st.pending_send = outcome.value
        st.ready_at = self.now + dur

    def _abort(self, st: AgentState, reason: str) -> None:
        self.aborts += 1
        self.trace(f"{st.name} ABORT ({reason}); undoing {len(st.writes)} writes")
        for w in sorted(st.writes, key=lambda w: w.key, reverse=True):
            self.target.undo_live(w)
            traj = self.target.traj(w.obj)
            traj.entries = [e for e in traj.entries if e is not w]
        st.writes.clear()
        st.blocked_on = None
        st.finished_at = 0.0
        self.protocol.on_abort(self, st)
        if st.restarts + 1 >= self.retry_cap:
            st.status = Status.FAILED
            self.livelocked = True
            self.trace(f"{st.name} hit retry cap {self.retry_cap}")
            return
        st.reset_context()
        st.gen = st.agent.run(self, st)
        st.pending_send = None
        st.pending_call = None
        st.status = Status.RUNNING
        st.ready_at = self.now

    def _check_deadlock(self) -> None:
        """Wait-for cycle detection; the victim is the higher-sigma agent."""
        blocked = [s for s in self.states.values() if s.status is Status.BLOCKED]
        if len(blocked) < 2:
            return
        if not self.protocol.detects_deadlock:
            return
        # Build the wait-for graph from "<owner>:<scope>" and look for a
        # cycle. Requiring *every* agent to be blocked would miss a genuine
        # two-agent deadlock whenever a third agent has already finished.
        waits: Dict[str, str] = {}
        for s in blocked:
            if s.blocked_on and ":" in s.blocked_on:
                owner = s.blocked_on.split(":", 1)[0]
                if owner in self.states:
                    waits[s.name] = owner
        cycle = self._find_cycle(waits)
        if cycle:
            victim = max((self.states[n] for n in cycle), key=lambda s: s.sigma)
            self.deadlocks += 1
            self.trace(f"DEADLOCK detected; victim={victim.name}")
            for s in blocked:
                if s is not victim:
                    s.status = Status.RUNNING
                    s.blocked_on = None
            self._abort(victim, "deadlock victim")

    @staticmethod
    def _find_cycle(waits: Dict[str, str]) -> Optional[List[str]]:
        """Return the members of any cycle in the wait-for graph."""
        for start in waits:
            seen, node = [], start
            while node in waits:
                if node in seen:
                    return seen[seen.index(node):]
                seen.append(node)
                node = waits[node]
        return None

    def wake_blocked(self) -> None:
        for s in self.states.values():
            if s.status is Status.BLOCKED:
                s.status = Status.RUNNING
                s.blocked_on = None
                s.ready_at = max(s.ready_at, self.now)

    def _jitter(self) -> float:
        if self.jitter_sigma <= 0:
            return 1.0
        return max(0.4, self.rng.gauss(1.0, self.jitter_sigma))

    # -- results -----------------------------------------------------------
    def totals(self) -> Tuple[float, int]:
        wall = max((s.finished_at for s in self.states.values()), default=self.now)
        tokens = sum(s.billed_tokens for s in self.states.values())
        return wall, tokens
