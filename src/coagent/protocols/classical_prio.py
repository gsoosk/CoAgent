"""Whole-workflow 2PL with priority-ordered locking instead of detection.

``chopping-prio`` changes two things relative to ``2PL-saga`` at once: it
runs over a certified chopping, and it resolves lock conflicts by sigma
(wound-wait) rather than by detecting a wait-for cycle. On the ten contended
cells the certifier merges every workflow to one piece, so the first change
is inert there and the gain has to come from the second. This protocol is
the control that says so directly: plain strict 2PL, locks held to commit,
no pieces, no certificate, with only the conflict rule swapped.

The rule is the one ``chopping-prio`` uses:

  requester has higher sigma than the holder  -> the requester waits
  requester has lower sigma than the holder   -> the holder is wounded:
      aborted, its writes undone by saga, its locks released

A workflow only ever waits for one of strictly lower sigma, so the wait-for
graph cannot hold a cycle and there is nothing for a detector to find.
"""

from __future__ import annotations

from typing import Dict, Optional

from ..runtime import AgentState, Runtime, Status
from .base import Outcome
from .classical import Lock, Mode, Scope, TwoPhaseLocking

__all__ = ["TwoPhaseLockingPriority"]


class TwoPhaseLockingPriority(TwoPhaseLocking):
    """Strict 2PL whose lock conflicts are resolved by sigma (wound-wait)."""

    name = "2PL-prio"
    detects_deadlock = False

    def __init__(self) -> None:
        super().__init__()
        self.wounds = 0
        self.waits = 0
        self._wounding = False

    def _acquire(self, rt: Runtime, st: AgentState, scope: Scope,
                 mode: Mode) -> Optional[Outcome]:
        if self._holds(scope, mode, st.name):
            return None
        # Wounding one holder can expose another on the same scope (several
        # readers), so resolve until the scope is clear or we must wait.
        while True:
            clash = self._conflicting(scope, mode, st.name)
            if clash is None:
                break
            hst = rt.states.get(clash.owner)
            if hst is None or hst.status is Status.DONE:
                # a finished or unknown owner cannot be waited on or wounded
                self.locks = [lk for lk in self.locks
                              if lk.owner != clash.owner]
                continue
            if hst.sigma < st.sigma:
                self.waits += 1
                return Outcome(blocked=True,
                               blocked_on=f"{clash.owner}:{scope.path}")
            self.wounds += 1
            rt.trace(f"WOUND {clash.owner} (sigma {hst.sigma}) for {st.name} "
                     f"(sigma {st.sigma})")
            self._wounding = True
            try:
                rt._abort(hst, f"wounded by {st.name}")
            finally:
                self._wounding = False
        self.locks.append(Lock(scope, mode, st.name))
        st.locks.add((scope.kind, scope.path, mode.value))
        return None

    def on_abort(self, rt: Runtime, st: AgentState) -> None:
        self._release_all(st)
        # the wounder is mid-acquire and takes the lock itself; waking
        # others here would let them race it for the scope it just cleared
        if not self._wounding:
            rt.wake_blocked()

    def on_quiescence(self, rt: Runtime) -> bool:
        """Safe to wake: a blocked workflow waits on a lower sigma only."""
        blocked = [s for s in rt.states.values() if s.status is Status.BLOCKED]
        if blocked:
            rt.wake_blocked()
            return True
        return False

    def stats(self) -> Dict[str, int]:
        return {"wounds": self.wounds, "lock_waits": self.waits}
