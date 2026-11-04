"""Contended-cell definition and the invariant check.

Paper section 7.1: "Each cell ships with a set of hand-written invariants:
any concurrent execution must be equivalent to either agent-1 . agent-2 or
agent-2 . agent-1; a run satisfying neither is recorded as a violation."

We derive the two reference states by actually running the cell serially in
both orders, rather than hand-transcribing them, so the oracle cannot drift
from the workload.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core import TargetSystem


@dataclass
class Cell:
    """One contended workload: two agents built to race on shared state."""

    name: str
    suite: str                     # "aiopslab" | "workbench"
    anomaly: str                   # the textbook anomaly the pair exhibits
    initial: Dict[str, Any]
    build_agents: Callable[..., List[Any]]
    description: str = ""
    # optional extra semantic predicate over the final state
    semantic_invariant: Optional[Callable[[Dict[str, Any]], bool]] = None
    # what a real LLM Worker needs to run this cell: one natural-language
    # task per agent, plus the tool catalog it may call. Declared as plain
    # data so the workload package stays free of the HTTP backend.
    llm_spec: Optional[Dict[str, Any]] = None

    def fresh_target(self) -> TargetSystem:
        return TargetSystem(copy.deepcopy(self.initial))


def normalize(state: Dict[str, Any]) -> Dict[str, Any]:
    """Drop objects that were never created, so absent == None."""
    return {k: v for k, v in sorted(state.items()) if v is not None}


def states_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return normalize(a) == normalize(b)


def invariant_ok(
    final: Dict[str, Any],
    references: List[Dict[str, Any]],
    cell: Optional[Cell] = None,
) -> bool:
    """A run is correct iff it matches some serial order (and any extra rule)."""
    if not any(states_equal(final, ref) for ref in references):
        return False
    if cell is not None and cell.semantic_invariant is not None:
        return bool(cell.semantic_invariant(final))
    return True


CELLS: Dict[str, Cell] = {}


def register(cell: Cell) -> Cell:
    CELLS[cell.name] = cell
    return cell
