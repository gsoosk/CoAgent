"""Protocol registry (paper section 7.1: five protocols on the same middleware)."""

from typing import Callable, Dict

from .base import NaiveProtocol, Outcome, Protocol, SerialProtocol
from .classical import OptimisticCC, TwoPhaseLocking
from .classical_prio import TwoPhaseLockingPriority
from .chopping import CertifiedChopping
from .chopping_2pl import ChoppingTwoPhase
from .chopping_prio import ChoppingPriority
from .chopping_tight import (
    ChoppingMinimal, ChoppingReorder, ChoppingTwoPhaseIntent,
)
from .mtpo import MTPO

PROTOCOLS: Dict[str, Callable[[], Protocol]] = {
    "serial": SerialProtocol,
    "naive": NaiveProtocol,
    "2PL-saga": TwoPhaseLocking,
    "OCC-saga": OptimisticCC,
    "MTPO": MTPO,
    "chopping": CertifiedChopping,
    "chopping-2pl": ChoppingTwoPhase,
    "chopping-prio": ChoppingPriority,
    # follow-ups to the anatomy experiment; not part of PROTOCOL_ORDER, so
    # the published matrix and its figures are unaffected
    "chopping-min": ChoppingMinimal,
    "chopping-reorder": ChoppingReorder,
    "chopping-2pl-intent": ChoppingTwoPhaseIntent,
    # the control for chopping-prio: the priority rule with no chopping
    "2PL-prio": TwoPhaseLockingPriority,
    # the section 5.2 straw-man: notification with no pre-order
    "broadcast": lambda: MTPO(broadcast=True),
}

PROTOCOL_ORDER = ["serial", "naive", "2PL-saga", "OCC-saga", "MTPO",
                  "chopping", "chopping-2pl", "chopping-prio"]

# What the figure scripts draw, in order. They only draw protocols a run
# actually recorded, so a results folder without the follow-ups is unchanged.
# chopping-min is left out: it was run on the two cells it changes, and the
# figures drop any cell a drawn protocol did not cover.
FIGURE_ORDER = PROTOCOL_ORDER + ["chopping-2pl-intent", "chopping-reorder"]
FIGURE_ORDER.insert(FIGURE_ORDER.index("2PL-saga") + 1, "2PL-prio")

__all__ = [
    "PROTOCOLS",
    "PROTOCOL_ORDER",
    "FIGURE_ORDER",
    "Protocol",
    "Outcome",
    "SerialProtocol",
    "NaiveProtocol",
    "TwoPhaseLocking",
    "OptimisticCC",
    "MTPO",
    "CertifiedChopping",
    "ChoppingTwoPhase",
    "ChoppingPriority",
    "TwoPhaseLockingPriority",
    "ChoppingMinimal",
    "ChoppingReorder",
    "ChoppingTwoPhaseIntent",
]
