"""Protocol registry (paper section 7.1: five protocols on the same middleware)."""

from typing import Callable, Dict

from .base import NaiveProtocol, Outcome, Protocol, SerialProtocol
from .classical import OptimisticCC, TwoPhaseLocking
from .chopping import CertifiedChopping
from .mtpo import MTPO

PROTOCOLS: Dict[str, Callable[[], Protocol]] = {
    "serial": SerialProtocol,
    "naive": NaiveProtocol,
    "2PL-saga": TwoPhaseLocking,
    "OCC-saga": OptimisticCC,
    "MTPO": MTPO,
    "chopping": CertifiedChopping,
    # the section 5.2 straw-man: notification with no pre-order
    "broadcast": lambda: MTPO(broadcast=True),
}

PROTOCOL_ORDER = ["serial", "naive", "2PL-saga", "OCC-saga", "MTPO",
                  "chopping"]

__all__ = [
    "PROTOCOLS",
    "PROTOCOL_ORDER",
    "Protocol",
    "Outcome",
    "SerialProtocol",
    "NaiveProtocol",
    "TwoPhaseLocking",
    "OptimisticCC",
    "MTPO",
    "CertifiedChopping",
]
