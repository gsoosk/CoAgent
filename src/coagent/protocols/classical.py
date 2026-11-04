"""The two classical baselines, adapted to live state with saga undo.

Both are what the paper calls 2PL-saga and OCC-saga: since the target system
offers no fork and no buffer (section 3.4), writes take effect in place and
rollback goes through the inverse each tool registered (section 6.3).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core import covered_by
from ..runtime import AgentState, Call, Runtime, Status
from .base import Outcome, Protocol


class Mode(Enum):
    SHARED = "S"
    EXCLUSIVE = "X"


@dataclass(frozen=True)
class Scope:
    """A lock scope: one object, or a whole subtree (a range read lock)."""

    kind: str   # "obj" | "prefix"
    path: str

    def overlaps(self, other: "Scope") -> bool:
        if self.kind == "obj" and other.kind == "obj":
            return self.path == other.path
        if self.kind == "obj" and other.kind == "prefix":
            return self.path.startswith(other.path)
        if self.kind == "prefix" and other.kind == "obj":
            return other.path.startswith(self.path)
        return self.path.startswith(other.path) or other.path.startswith(self.path)


@dataclass
class Lock:
    scope: Scope
    mode: Mode
    owner: str


class TwoPhaseLocking(Protocol):
    """Strict 2PL: locks held to commit, deadlock victim undone by saga."""

    name = "2PL-saga"
    detects_deadlock = True

    def __init__(self):
        self.locks: List[Lock] = []

    # -- lock table --------------------------------------------------------
    def _conflicting(self, scope: Scope, mode: Mode, owner: str) -> Optional[Lock]:
        for lk in self.locks:
            if lk.owner == owner or not lk.scope.overlaps(scope):
                continue
            if mode is Mode.SHARED and lk.mode is Mode.SHARED:
                continue
            return lk
        return None

    def _holds(self, scope: Scope, mode: Mode, owner: str) -> bool:
        for lk in self.locks:
            if lk.owner != owner:
                continue
            if lk.mode is mode or lk.mode is Mode.EXCLUSIVE:
                if lk.scope == scope or (
                    lk.scope.kind == "prefix" and scope.overlaps(lk.scope)
                ):
                    return True
        return False

    def _acquire(self, rt, st, scope: Scope, mode: Mode) -> Optional[Outcome]:
        if self._holds(scope, mode, st.name):
            return None
        clash = self._conflicting(scope, mode, st.name)
        if clash is not None:
            return Outcome(blocked=True, blocked_on=f"{clash.owner}:{scope.path}")
        self.locks.append(Lock(scope, mode, st.name))
        st.locks.add((scope.kind, scope.path, mode.value))
        return None

    def _release_all(self, st: AgentState) -> None:
        self.locks = [lk for lk in self.locks if lk.owner != st.name]
        st.locks.clear()

    # -- policy ------------------------------------------------------------
    def on_read(self, rt, st, call):
        scope = (
            Scope("prefix", call.prefix) if call.prefix is not None
            else Scope("obj", call.obj)
        )
        blocked = self._acquire(rt, st, scope, Mode.SHARED)
        if blocked:
            return blocked
        return Outcome(value=self.live_read(rt, st, call))

    def on_write(self, rt, st, call):
        blocked = self._acquire(rt, st, Scope("obj", call.obj), Mode.EXCLUSIVE)
        if blocked:
            return blocked
        w = self.make_write(rt, st, call)
        self.apply_and_record(rt, st, w)
        return Outcome(value=True)

    def on_commit(self, rt, st):
        self._release_all(st)
        rt.wake_blocked()

    def on_abort(self, rt, st):
        self._release_all(st)
        rt.wake_blocked()


class OptimisticCC(Protocol):
    """OCC with backward validation at commit, and saga undo instead of a buffer.

    Writes cannot be staged on live state (section 3.4), so they go in place
    and a failed validation unwinds them through the registered inverses. A
    transaction fails only if something committed during its lifetime wrote an
    object it read -- so the first committer wins and the loser redoes in full.
    """

    name = "OCC-saga"

    def __init__(self):
        self.clock = 0
        self.commits: List[Tuple[int, str, Set[str]]] = []
        self.start_stamp: Dict[str, int] = {}
        self.cascaded: Set[str] = set()
        self.aborting_writes: List[Any] = []

    def _stamp(self, st: AgentState) -> None:
        if st.name not in self.start_stamp:
            self.start_stamp[st.name] = self.clock

    @staticmethod
    def _covers(read_key: str, obj: str) -> bool:
        """A recorded read covers ``obj`` if it was that object or its subtree.

        The subtree case is the phantom: a range read conflicts with an insert
        under it, which is how A's namespace audit collides with B's canary.
        """
        return covered_by(obj, read_key)

    def on_read(self, rt, st, call):
        self._stamp(st)
        return Outcome(value=self.live_read(rt, st, call))

    def on_write(self, rt, st, call):
        self._stamp(st)
        w = self.make_write(rt, st, call)
        self.apply_and_record(rt, st, w)
        return Outcome(value=True)

    def on_commit(self, rt, st):
        self._stamp(st)
        if st.name in self.cascaded:
            self.cascaded.discard(st.name)
            self.aborting_writes = list(st.writes)
            rt.trace(f"{st.name} aborts: its premise came from an undone write")
            rt._abort(st, "cascaded abort")
            return
        began = self.start_stamp[st.name]
        my_writes = {w.obj for w in st.writes}
        for stamp, who, writes in self.commits:
            if who == st.name or stamp < began:
                continue
            # Reads overwritten by a concurrent committer (rw), and -- because
            # writes are installed in place rather than buffered -- writes to
            # an object a concurrent committer also wrote (ww). Omitting the
            # ww half lets two installed writes on one object both "validate".
            clash = {
                o for o in writes
                if any(self._covers(k, o) for k in st.read_set)
            } | (writes & my_writes)
            if clash:
                self.aborting_writes = list(st.writes)
                rt.trace(
                    f"{st.name} fails validation at commit "
                    f"(read {sorted(clash)[0]} overwritten by {who}); full redo"
                )
                rt._abort(st, "validation failed")
                return
        self.commits.append((self.clock, st.name, my_writes))
        self.clock += 1

    def on_abort(self, rt, st):
        self.start_stamp.pop(st.name, None)
        # Cascade. Writes install in place, so a peer may already have read an
        # uncommitted value from the agent now unwinding. Without this, that
        # peer commits on a premise that never existed in any serial run.
        dirty = {w.obj for w in self.aborting_writes}
        for other in rt.others(st.name):
            if other.status in (Status.DONE, Status.FAILED):
                continue
            if any(self._covers(k, o) for k in other.read_set for o in dirty):
                rt.trace(f"{other.name} read dirty state from {st.name}; cascade")
                self.cascaded.add(other.name)
