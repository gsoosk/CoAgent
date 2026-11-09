"""Certified chopping as a protocol over the existing middleware.

The workflow is the agent, a piece is a contiguous run of its steps, and the
runtime makes the certificate's hypotheses true rather than assuming them:

* **Mediation and fail-closed enforcement.** Every call is checked against
  the manifest of the piece currently executing. An access outside it is
  refused before it reaches the store, because a model-generated description
  of what a tool does is not evidence about what it does.
* **Piece isolation.** A piece acquires locks for its *entire* declared
  footprint atomically, in one wait, before executing. Nothing is acquired
  incrementally while holding locks, so the lock manager is deadlock-free by
  construction -- unlike 2PL, which deadlocks here 0.79 times per trial.
* **Serialized admission (Theorem 3).** Revisions are evaluated and installed
  against a single plan snapshot, with the frontier fixed, because two
  individually-admissible revisions can be jointly unsafe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..chopping import (
    Access, Admission, Decision, Mode, Piece, Plan, admit, certify,
    cycle_through_committed, guided_coarsen, seal,
)
from ..runtime import AgentState, Call, CallKind, ReadRecord, Runtime, Status
from .base import Outcome, Protocol


def accesses_for(call: Call) -> List[Access]:
    """The accesses a call actually performs, for enforcement."""
    out = []
    if call.prefix is not None:
        out.append(Access.prefix(call.prefix, Mode.READ))
    elif call.kind is CallKind.READ and call.obj:
        out.append(Access.key(call.obj, Mode.READ))
    if call.kind is CallKind.WRITE and call.obj:
        out.append(Access.key(call.obj, Mode.WRITE))
    return out


@dataclass
class Held:
    access: Access
    owner: str


class CertifiedChopping(Protocol):
    """Static certification at launch, prefix-aware admission mid-flight."""

    name = "chopping"

    def __init__(self):
        self.plan: Optional[Plan] = None
        self.frontier: Dict[str, int] = {}
        self.locks: List[Held] = []
        self.active: Dict[str, int] = {}        # agent -> piece currently held
        self.certified = False
        self.initial_merges = 0
        self.revisions = 0
        self.admitted = 0
        self.admitted_coarsened = 0
        self.refused = 0
        self.refused_through_committed = 0
        self.enforcement_refusals = 0

    # -- plan setup --------------------------------------------------------
    def _ensure_plan(self, rt: Runtime) -> None:
        if self.certified:
            return
        pieces: Dict[str, List[Piece]] = {}
        for name, st in rt.states.items():
            declared = getattr(st.agent, "declare_pieces", lambda: [])()
            pieces[name] = [
                Piece(name, i, frozenset(fp), label)
                for i, (fp, label) in enumerate(declared)
            ] or [Piece(name, 0, frozenset(), "whole")]
            self.frontier[name] = 0
        plan = Plan(pieces)
        ok, _ = certify(plan)
        if not ok:
            # Before execution the repair always exists: in the limit each
            # workflow is one piece. This is the paper's "correctly collapses
            # to whole-workflow locking where the structure does not permit
            # chopping" -- it buys nothing, and that is the point.
            plan, self.initial_merges = guided_coarsen(plan, self.frontier)
            rt.trace(f"certification merged {self.initial_merges} piece(s)")
        self.plan = plan
        self.certified = True

    def _piece_of(self, st: AgentState, call: Call) -> Piece:
        """Which declared piece does this call belong to?

        A scripted agent says so directly. A model-driven agent does not
        follow a script, so the runtime advances from the piece currently
        held to the first later piece whose manifest permits the access --
        and if none does, the access stays with the current piece and
        becomes a revision proposal.
        """
        ps = self.plan.pieces[st.name]
        if call.piece is not None:
            return ps[min(call.piece, len(ps) - 1)]
        held = self.active.get(st.name)
        start = held if held is not None else self.frontier.get(st.name, 0)
        accs = accesses_for(call)
        for i in range(start, len(ps)):
            if all(ps[i].permits(a.mode, a.lo) for a in accs):
                return ps[i]
        return ps[min(start, len(ps) - 1)]

    # -- lock table --------------------------------------------------------
    def _blocker(self, piece: Piece, owner: str) -> Optional[str]:
        for h in self.locks:
            if h.owner == owner:
                continue
            for a in piece.footprint:
                if a.conflicts(h.access):
                    return h.owner
        return None

    def _acquire(self, piece: Piece, owner: str) -> None:
        for a in piece.footprint:
            self.locks.append(Held(a, owner))

    def _release(self, owner: str) -> None:
        self.locks = [h for h in self.locks if h.owner != owner]

    # -- dispatch ----------------------------------------------------------
    def dispatch(self, rt: Runtime, st: AgentState, call: Call) -> Outcome:
        self._ensure_plan(rt)
        piece = self._piece_of(st, call)

        # a new piece: commit the previous one, then take the whole footprint
        held = self.active.get(st.name)
        if held is not None and held != piece.index:
            self._commit_piece(rt, st)
        if self.active.get(st.name) != piece.index:
            blocker = self._blocker(piece, st.name)
            if blocker is not None:
                return Outcome(blocked=True,
                               blocked_on=f"{blocker}:{piece.label or piece.index}")
            self._acquire(piece, st.name)
            self.active[st.name] = piece.index

        # fail-closed enforcement against the manifest
        for a in accesses_for(call):
            key = call.prefix if call.prefix is not None else call.obj
            if not piece.permits(a.mode, a.lo):
                out = self._revise(rt, st, call, piece)
                if out is not None:
                    return out
                break

        if call.kind is CallKind.READ:
            out = Outcome(value=self.live_read(rt, st, call))
        else:
            w = self.make_write(rt, st, call)
            self.apply_and_record(rt, st, w)
            out = Outcome(value=True)

        if call.piece is None and piece.span == 1:
            # A model-driven agent declares no step boundaries, but every gap
            # between its tool calls *is* a reasoning interval. So an
            # unmerged piece closes when its call returns, which gives the
            # frontier something to advance and makes prefix-aware admission
            # meaningful at all.
            #
            # A piece with span > 1 is one certification merged across a
            # reasoning gap. Releasing it per call would undo exactly the
            # repair that made the plan safe -- which is how the three
            # lost-update cells lost their updates.
            self._commit_piece(rt, st)
        return out

    # -- mid-flight revision ----------------------------------------------
    def _revise(self, rt: Runtime, st: AgentState, call: Call,
                piece: Piece) -> Optional[Outcome]:
        """An access outside the manifest is a plan revision, not a bug.

        The agent discovered after reading that it must touch something it
        never declared. Admission is evaluated against a single snapshot with
        the frontier fixed (Theorem 3).
        """
        self.revisions += 1
        proposed = self.plan.copy()
        extra = frozenset(set(piece.footprint) | set(accesses_for(call)))
        ps = proposed.pieces[st.name]
        ps[piece.index] = Piece(st.name, piece.index, extra, piece.label)
        res = admit(proposed, self.frontier)

        if res.decision is Decision.REFUSED:
            self.refused += 1
            through = cycle_through_committed(res.witness, self.frontier)
            if through:
                self.refused_through_committed += 1
            self.enforcement_refusals += 1
            rt.trace(
                f"{st.name} revision REFUSED on {call.obj or call.prefix}"
                + (" (cycle through a committed piece)" if through else "")
            )
            # Fail closed: the access never reaches the store. The agent is
            # told, and may replan -- which is what the real-agent study
            # exercises.
            st.inbox.append(_refusal(call, rt.now, st.name))
            return Outcome(value={"refused": True,
                                  "reason": "revision not admissible",
                                  "object": call.obj or call.prefix})
        if res.decision is Decision.ADMITTED:
            self.admitted += 1
        else:
            self.admitted_coarsened += 1
        self.plan = res.plan
        # a merge may have widened the live piece; take any new locks now
        newp = self._piece_of(st, call)
        self._release(st.name)
        self._acquire(newp, st.name)
        self.active[st.name] = newp.index
        rt.trace(f"{st.name} revision {res.decision.value} "
                 f"on {call.obj or call.prefix}")
        return None

    # -- commit ------------------------------------------------------------
    def _commit_piece(self, rt: Runtime, st: AgentState) -> None:
        idx = self.active.pop(st.name, None)
        if idx is None:
            return
        self._release(st.name)
        self.frontier[st.name] = max(self.frontier.get(st.name, 0), idx + 1)
        rt.wake_blocked()

    def on_think(self, rt: Runtime, st: AgentState, call: Call) -> None:
        """Commit the piece before reasoning, releasing its locks.

        This is the mechanism: a whole-workflow policy holds the shared
        footprint across the agent's private interval, which is the
        expensive part. A certified chopping does not.
        """
        held = self.active.get(st.name)
        if held is None:
            return
        self._ensure_plan(rt)
        nxt = self._piece_of(st, call)
        if nxt.index != held:
            # a genuine boundary: release before the agent reasons
            self._commit_piece(rt, st)
            rt.wake_blocked()
        # otherwise certification merged across this gap, so the lock is
        # held through it -- which is exactly whole-workflow locking

    def on_commit(self, rt: Runtime, st: AgentState) -> None:
        self._commit_piece(rt, st)
        rt.wake_blocked()

    def on_abort(self, rt: Runtime, st: AgentState) -> None:
        self._commit_piece(rt, st)

    def on_quiescence(self, rt: Runtime) -> bool:
        blocked = [s for s in rt.states.values() if s.status is Status.BLOCKED]
        if blocked:
            rt.wake_blocked()
            return True
        return False

    def stats(self) -> Dict[str, int]:
        return {
            "initial_merges": self.initial_merges,
            "revisions": self.revisions,
            "admitted": self.admitted,
            "admitted_coarsened": self.admitted_coarsened,
            "refused": self.refused,
            "refused_through_committed": self.refused_through_committed,
        }


def _refusal(call: Call, t: float, agent: str):
    from ..runtime import Notification
    return Notification(
        obj=call.obj or call.prefix or "",
        new_value={"refused": True,
                   "reason": "this access is outside your certified "
                             "footprint and the revision is not admissible"},
        from_agent="certifier", to_agent=agent, t=t, reason="refusal",
    )
