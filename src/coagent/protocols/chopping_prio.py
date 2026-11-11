"""Certified chopping with priority-ordered locking (wound-wait over sigma).

``ChoppingTwoPhase`` showed that relaxing one-shot acquisition buys shorter
holds and pays for them with 2PL's deadlock tail -- on the contended cells it
reproduces 2PL exactly, deadlocks included. The deadlock is the only thing
standing between incremental locking and a strict improvement, and it does
not have to be *detected*. It can be made unreachable.

The observation is MTPO's own. MTPO's ``rw`` rule pushes notifications
strictly from lower sigma to higher sigma, and that one-way flow is what
makes its trajectory a pre-order rather than a tangle. Apply the same
constraint to lock waits and the wait-for graph is acyclic by construction:

    requester sigma <  holder sigma  ->  WOUND the holder (abort it), proceed
    requester sigma >  holder sigma  ->  WAIT for the holder

Every wait points from high sigma to low sigma. A cycle needs one edge going
the other way, and there is none, so no cycle exists and no detector is
needed. This is textbook wound-wait, with sigma -- the pre-order MTPO already
assigns at launch -- as the timestamp.

The semantics line up with MTPO too. Under MTPO the low-sigma agent writes
speculatively as though it ran first and the high-sigma agent absorbs the
change through a notification. Here the low-sigma agent takes the lock as
though it ran first and the high-sigma agent absorbs it by aborting and
redoing. Same pre-order, same direction of yielding, different mechanism for
the yield -- premise refresh there, saga undo here.

Wound-wait rather than wait-die on purpose: wait-die aborts the high-sigma
agent the moment it meets any held lock, even one about to be released, so it
aborts strictly more often for the same guarantee.

What this is expected to cost. Deadlocks go to zero, but the aborts do not
vanish -- they move, and they land entirely on high-sigma agents. An agent
wounded repeatedly can exhaust ``retry_cap`` and be marked FAILED, which the
runner scores as a livelock. That starvation risk is the price of the
guarantee and is the thing to measure, not to assume away: a protocol that
never deadlocks but starves its low-priority agent has moved the failure, not
removed it.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

from ..chopping import Access
from ..runtime import AgentState, Call, CallKind, Runtime, Status
from .base import Outcome
from .chopping import accesses_for
from .chopping_2pl import ChoppingTwoPhase

__all__ = ["ChoppingPriority"]


class ChoppingPriority(ChoppingTwoPhase):
    """Chopping whose pieces lock incrementally, ordered by sigma."""

    name = "chopping-prio"

    # Prevention, not detection: the runtime's cycle detector has nothing to
    # find, because a cycle cannot form.
    detects_deadlock = False

    def __init__(self) -> None:
        super().__init__()
        self.wounds = 0
        self.waits = 0

    # -- conflict resolution ----------------------------------------------
    def _conflict_owner(self, accs: Sequence[Access],
                        owner: str) -> Optional[str]:
        """The first other owner holding something that conflicts."""
        for h in self.locks:
            if h.owner == owner:
                continue
            for a in accs:
                if a.conflicts(h.access):
                    return h.owner
        return None

    def _resolve(self, rt: Runtime, st: AgentState,
                 accs: Sequence[Access]) -> Optional[str]:
        """Clear every conflicting holder, or name the one to wait for.

        Returns ``None`` when the caller may proceed, or the name of the
        holder it must block on. Wounding is applied repeatedly because
        clearing one holder can expose another on a different object in the
        same footprint.
        """
        while True:
            holder = self._conflict_owner(accs, st.name)
            if holder is None:
                return None
            hst = rt.states.get(holder)
            if hst is None:                      # stale entry; drop it
                self._release(holder)
                continue
            if hst.sigma < st.sigma:
                # The holder outranks us. Wait -- and this is the only
                # direction a wait ever points, which is what kills cycles.
                self.waits += 1
                return holder
            if hst.status is Status.DONE:
                # It finished without releasing; nothing to abort.
                self._release(holder)
                continue
            # We outrank the holder: wound it.
            self.wounds += 1
            rt.trace(f"WOUND {holder} (sigma {hst.sigma}) for {st.name} "
                     f"(sigma {st.sigma})")
            self._wounding = True
            try:
                rt._abort(hst, f"wounded by {st.name}")
            finally:
                self._wounding = False

    # -- dispatch ----------------------------------------------------------
    def dispatch(self, rt: Runtime, st: AgentState, call: Call) -> Outcome:
        self._ensure_plan(rt)
        piece = self._piece_of(st, call)

        held = self.active.get(st.name)
        if held is not None and held != piece.index:
            self._commit_piece(rt, st)

        for a in accesses_for(call):
            if not piece.permits(a.mode, a.lo):
                out = self._revise(rt, st, call, piece)
                if out is not None:
                    return out
                piece = self._piece_of(st, call)
                break

        need = accesses_for(call)
        blocker = self._resolve(rt, st, need)
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

        if call.piece is None and piece.span == 1:
            self._commit_piece(rt, st)
        return out

    # -- abort -------------------------------------------------------------
    def on_abort(self, rt: Runtime, st: AgentState) -> None:
        """Release the victim's locks without advancing its frontier.

        Counted separately from the parent's deadlock aborts: here an abort
        is a wound, which is the mechanism working, not a cycle being broken
        after the fact.
        """
        self.active.pop(st.name, None)
        self._release(st.name)
        if not getattr(self, "_wounding", False):
            rt.wake_blocked()

    def on_quiescence(self, rt: Runtime) -> bool:
        """Waking a blocked agent is safe here: it waits on a lower sigma.

        A blocked agent is always waiting for a strictly higher-priority
        holder, so re-running it either succeeds now or re-blocks on the
        same holder. There is no cycle for the wake to paper over, which is
        why this can wake where ``ChoppingTwoPhase`` must not.
        """
        blocked = [s for s in rt.states.values() if s.status is Status.BLOCKED]
        if blocked:
            rt.wake_blocked()
            return True
        return False

    def stats(self) -> Dict[str, int]:
        s = super().stats()
        s["wounds"] = self.wounds
        s["lock_waits"] = self.waits
        return s
