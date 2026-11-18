"""Three ways to get more out of a certified chopping.

The anatomy experiment showed that on ten of the twelve benchmark cells the
certifier merges every workflow back to one piece, so chopping is
whole-workflow locking there. These variants ask how much of that is forced
by the workload and how much by choices the runtime makes.

``ChoppingMinimal`` changes only how a failing plan is merged: it merges the
span of pieces that carry the conflict and leaves the rest cut
(``conflict_coarsen``), instead of merging the cheapest adjacent pair.

``ChoppingReorder`` is the same protocol. Its difference is at launch, and
lives in the agent (``agents.planning``): an agent whose declared plan fails
certification is asked for an order of its own steps that gives the same
final result but puts the contended steps next to each other, so the span
that has to be merged is shorter. The model's judgement is not trusted for
isolation: the certifier checks whatever order comes back, and enforcement
stays fail-closed. A wrong "this order is equivalent" can only break the
agent's own task.

``ChoppingTwoPhaseIntent`` reorders lock *acquisition* rather than
operations. Every deadlock ``chopping-2pl`` hit on the benchmark is a lock
upgrade: two pieces read, then one needs a write lock the other's read
blocks. No reordering of operations removes that, because the read has to
come first. But the piece's certified footprint already says it will write,
so the write lock can be taken at the first touch.
"""

from __future__ import annotations

from typing import List, Sequence

from ..chopping import Access, conflict_coarsen
from ..runtime import Call
from .chopping import CertifiedChopping, accesses_for
from .chopping_2pl import ChoppingTwoPhase

__all__ = ["ChoppingMinimal", "ChoppingReorder", "ChoppingTwoPhaseIntent"]


class ChoppingMinimal(CertifiedChopping):
    """Certified chopping that merges no more than the conflict requires."""

    name = "chopping-min"
    _coarsen = staticmethod(conflict_coarsen)


class ChoppingReorder(ChoppingMinimal):
    """``chopping-min`` over plans the agents were asked to reorder."""

    name = "chopping-reorder"
    # read by the runner: negotiate step order with the agents before launch
    negotiates_order = True


class ChoppingTwoPhaseIntent(ChoppingTwoPhase):
    """``chopping-2pl`` that takes the declared write lock at first touch."""

    name = "chopping-2pl-intent"

    def __init__(self) -> None:
        super().__init__()
        self.intent_upgrades = 0

    def _needed(self, piece, call: Call) -> Sequence[Access]:
        need: List[Access] = []
        for a in accesses_for(call):
            # A write the piece has declared on anything this access
            # overlaps will be requested later anyway. Taking it now only
            # ever strengthens a lock, so the piece is still strict 2PL.
            later = [d for d in piece.footprint
                     if d.is_write and d.overlaps(a) and d != a]
            if later and not a.is_write:
                self.intent_upgrades += 1
            need.extend(later or [a])
            if later and a.is_write and a not in need:
                need.append(a)
        return need

    def stats(self):
        s = super().stats()
        s["intent_upgrades"] = self.intent_upgrades
        return s
