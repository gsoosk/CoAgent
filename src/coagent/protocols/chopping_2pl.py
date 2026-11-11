"""Certified chopping with two-phase locking inside each piece.

``CertifiedChopping`` takes a piece's entire declared footprint atomically in
one wait. That makes the lock manager structurally deadlock-free -- nothing is
ever acquired while holding something -- but it over-reserves in time: an
object the piece will not touch until its last step is held from its first
step, for the whole piece.

This variant keeps the certificate untouched and changes only the discipline
*inside* a piece. Locks are taken per access, at the moment the access
happens (the growing phase), and every lock the piece took is released
together at the piece boundary (the shrinking phase). That is strict 2PL
scoped to a piece.

Why the certificate survives. The chop certification reasons about pieces as
atomic units: it admits a chop when no mixed simple cycle exists over the SC
graph whose vertices are pieces. Strict two-phase locking makes a piece
atomic with respect to every *conflicting* piece -- a conflicting piece
cannot interleave, because the lock on the conflicting access is held to the
piece boundary -- so the hypothesis the certificate needs still holds. What
one-shot acquisition buys is not correctness; it is deadlock-freedom.

So this trades that away and must handle what comes back. Hold-and-wait
returns, a wait-for cycle is now reachable, and it is resolved exactly as
2PL-saga resolves one: the runtime aborts the highest-sigma member of the
cycle and the registered inverses unwind its writes. The expected shape of
the result is therefore shorter holds (more overlap) bought with a deadlock
tail -- the same trade 2PL-saga makes against whole-workflow locking, applied
one level down.

Two details that differ from the parent on purpose:

* Enforcement runs *before* acquisition. The parent can check the manifest
  after taking the footprint because the footprint is the manifest. Here a
  refused access must not reserve anything, or a call the protocol is about
  to deny could block a peer and even manufacture a deadlock.
* ``on_abort`` releases locks without advancing the frontier. The parent can
  route abort through ``_commit_piece`` because its aborts never happen; a
  deadlock victim here has *not* completed its piece and must redo it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from ..chopping import Access
from ..runtime import AgentState, Call, CallKind, Runtime
from .base import Outcome
from .chopping import CertifiedChopping, Held, accesses_for

__all__ = ["ChoppingTwoPhase"]


class ChoppingTwoPhase(CertifiedChopping):
    """Certified chopping whose pieces lock incrementally, 2PL-style."""

    name = "chopping-2pl"

    # Incremental acquisition reintroduces hold-and-wait, so the wait-for
    # graph can hold a cycle and the runtime has to be allowed to break it.
    # The parent leaves this False because one-shot acquisition cannot
    # deadlock.
    detects_deadlock = True

    def __init__(self) -> None:
        super().__init__()
        self.deadlock_aborts = 0
        self.incremental_acquires = 0

    # -- lock table: keyed by access, not by whole piece -------------------
    def _blocker_for(self, accs: Sequence[Access],
                     owner: str) -> Optional[str]:
        for h in self.locks:
            if h.owner == owner:
                continue
            for a in accs:
                if a.conflicts(h.access):
                    return h.owner
        return None

    def _already_held(self, acc: Access, owner: str) -> bool:
        return any(h.owner == owner and h.access == acc for h in self.locks)

    def _acquire_accesses(self, accs: Sequence[Access], owner: str) -> None:
        for a in accs:
            if not self._already_held(a, owner):
                self.locks.append(Held(a, owner))
                self.incremental_acquires += 1

    # -- dispatch ----------------------------------------------------------
    def dispatch(self, rt: Runtime, st: AgentState, call: Call) -> Outcome:
        self._ensure_plan(rt)
        piece = self._piece_of(st, call)

        # Shrinking phase: crossing a piece boundary drops everything the
        # previous piece held, which is the whole point of chopping.
        held = self.active.get(st.name)
        if held is not None and held != piece.index:
            self._commit_piece(rt, st)

        # Fail-closed enforcement first, so a denied access reserves nothing.
        for a in accesses_for(call):
            if not piece.permits(a.mode, a.lo):
                out = self._revise(rt, st, call, piece)
                if out is not None:
                    return out
                # The revision was admitted, so the plan changed under us and
                # the call may now belong to a different piece.
                piece = self._piece_of(st, call)
                break

        # Growing phase: take only what this call touches, when it touches it.
        need = accesses_for(call)
        blocker = self._blocker_for(need, st.name)
        if blocker is not None:
            key = (call.prefix if call.prefix is not None
                   else call.obj or piece.label or str(piece.index))
            return Outcome(blocked=True, blocked_on=f"{blocker}:{key}")
        self._acquire_accesses(need, st.name)
        self.active[st.name] = piece.index

        if call.kind is CallKind.READ:
            out = Outcome(value=self.live_read(rt, st, call))
        else:
            w = self.make_write(rt, st, call)
            self.apply_and_record(rt, st, w)
            out = Outcome(value=True)

        # Same rule as the parent: an unmerged piece closes when its call
        # returns, while a piece merged across a reasoning gap is held
        # through it -- releasing that one would undo the repair that made
        # the plan safe.
        if call.piece is None and piece.span == 1:
            self._commit_piece(rt, st)
        return out

    # -- abort -------------------------------------------------------------
    def on_abort(self, rt: Runtime, st: AgentState) -> None:
        """Release the victim's locks, but leave the frontier where it was.

        Routing this through ``_commit_piece`` would mark the piece done and
        advance the frontier past work the abort just undid.
        """
        self.deadlock_aborts += 1
        self.active.pop(st.name, None)
        self._release(st.name)
        rt.wake_blocked()

    def on_quiescence(self, rt: Runtime) -> bool:
        """Do not blanket-wake blocked agents; let the cycle be detected.

        The parent wakes everything that is blocked, which is the right
        recovery when a block can only ever be transient. Here a block can
        be a genuine wait-for cycle -- two pieces each holding a read lock
        and each wanting to upgrade it -- and waking them both just makes
        them re-block, forever. Returning False lets the runtime's detector
        run and abort the higher-sigma victim.
        """
        return False

    def stats(self) -> Dict[str, int]:
        s = super().stats()
        s["deadlock_aborts"] = self.deadlock_aborts
        s["incremental_acquires"] = self.incremental_acquires
        return s
