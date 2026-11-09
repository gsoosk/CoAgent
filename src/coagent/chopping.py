"""Certified chopping with prefix-aware recertification.

Implements the mechanism of "Certified Chopping for Agent Tool Workflows:
Admitting Plan Revisions over Committed Effects" (idea/idea.pdf).

A workflow is split into *pieces* that execute and commit independently. The
split is legal when the SC graph has no simple cycle mixing sibling (S) and
conflict (C) edges -- the classical chopping criterion of Shasha et al.

What this file adds beyond the classical test is admission of *mid-flight
revisions*. Once a prefix has committed, merging a committed piece with a
future one is not a physical operation, so the only repair left is to
regroup the uncommitted suffix. Theorem 2 says that is decidable by a single
cheap test on the coarsest candidate:

    a revision is admissible under future-only coarsening
    iff  seal(P', phi)  is SC-cycle-free

where seal merges every workflow's entire uncommitted remainder into one
piece. Corollary 1 is the teeth: when the residual cycle runs through a
committed piece, no future-only repair exists and waiting does not help,
because a committed workflow keeps both its S clique and its C edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


class Mode(Enum):
    READ = "read"
    WRITE = "write"


@dataclass(frozen=True)
class Access:
    """One declared access: an exact key or a half-open range [lo, hi).

    A range models a predicate read. A write to a key inside a declared range
    conflicts with a read of that range, which is what makes a phantom
    insertion visible to certification rather than silently legal.
    """

    mode: Mode
    lo: str
    hi: str

    @staticmethod
    def key(k: str, mode: Mode) -> "Access":
        return Access(mode, k, k + "\x00")

    @staticmethod
    def prefix(p: str, mode: Mode) -> "Access":
        """A subtree scan: every id starting with ``p``."""
        return Access(mode, p, p[:-1] + chr(ord(p[-1]) + 1) if p else "￿")

    @property
    def is_write(self) -> bool:
        return self.mode is Mode.WRITE

    def overlaps(self, other: "Access") -> bool:
        return self.lo < other.hi and other.lo < self.hi

    def conflicts(self, other: "Access") -> bool:
        return self.overlaps(other) and (self.is_write or other.is_write)

    def covers(self, key: str) -> bool:
        return self.lo <= key < self.hi


@dataclass(frozen=True)
class Piece:
    """A unit that executes and commits atomically, with its footprint."""

    workflow: str
    index: int
    footprint: frozenset        # of Access
    label: str = ""
    # how many originally-declared pieces this one covers. >1 means
    # certification merged across a reasoning gap, so its locks must be
    # held through that gap -- that is whole-workflow locking, and
    # releasing early would silently undo the repair.
    span: int = 1

    @property
    def id(self) -> Tuple[str, int]:
        return (self.workflow, self.index)

    def conflicts(self, other: "Piece") -> bool:
        return any(a.conflicts(b) for a in self.footprint
                   for b in other.footprint)

    def permits(self, mode: Mode, key: str) -> bool:
        """Fail-closed enforcement: is this access inside the manifest?"""
        for a in self.footprint:
            if not a.covers(key):
                continue
            if mode is Mode.READ or a.is_write:
                return True
        return False

    def merged_with(self, others: Sequence["Piece"], index: int) -> "Piece":
        fp: Set[Access] = set(self.footprint)
        labels = [self.label]
        for o in others:
            fp |= set(o.footprint)
            labels.append(o.label)
        return Piece(self.workflow, index, frozenset(fp),
                     " + ".join(x for x in labels if x),
                     span=self.span + sum(o.span for o in others))


class Plan:
    """A batch: an ordered list of pieces per workflow invocation."""

    def __init__(self, pieces: Dict[str, List[Piece]]):
        self.pieces = {w: list(ps) for w, ps in pieces.items()}

    @property
    def workflows(self) -> List[str]:
        return list(self.pieces)

    def all_pieces(self) -> List[Piece]:
        return [p for ps in self.pieces.values() for p in ps]

    def copy(self) -> "Plan":
        return Plan({w: list(ps) for w, ps in self.pieces.items()})

    def piece_count(self) -> int:
        return sum(len(ps) for ps in self.pieces.values())

    def __repr__(self) -> str:
        return "Plan(" + ", ".join(
            f"{w}:{len(ps)}" for w, ps in self.pieces.items()) + ")"


# -- SC graph ---------------------------------------------------------------
def sc_edges(plan: Plan) -> List[Tuple[Tuple[str, int], Tuple[str, int], str]]:
    """Undirected labelled edges: S within an invocation, C across them."""
    out = []
    pieces = plan.all_pieces()
    for w, ps in plan.pieces.items():
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                out.append((ps[i].id, ps[j].id, "S"))
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            a, b = pieces[i], pieces[j]
            if a.workflow == b.workflow:
                continue
            if a.conflicts(b):
                out.append((a.id, b.id, "C"))
    return out


def _biconnected_blocks(nodes, edges):
    """Hopcroft-Tarjan: partition edges into vertex-biconnected blocks.

    Any two edges of a block lie on a common simple cycle, so a block holding
    both an S and a C edge is exactly a mixed simple cycle.
    """
    adj: Dict[object, List[Tuple[object, int]]] = {n: [] for n in nodes}
    for idx, (u, v, _lab) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))

    disc: Dict[object, int] = {}
    low: Dict[object, int] = {}
    stack: List[int] = []
    blocks: List[List[int]] = []
    timer = [0]

    for root in nodes:
        if root in disc:
            continue
        # iterative DFS, so a deep plan cannot blow the Python stack
        frame = [(root, None, iter(adj[root]))]
        disc[root] = low[root] = timer[0]
        timer[0] += 1
        while frame:
            u, pe, it = frame[-1]
            advanced = False
            for v, ei in it:
                if ei == pe:
                    continue
                if v not in disc:
                    stack.append(ei)
                    disc[v] = low[v] = timer[0]
                    timer[0] += 1
                    frame.append((v, ei, iter(adj[v])))
                    advanced = True
                    break
                if disc[v] < disc[u]:
                    stack.append(ei)
                    low[u] = min(low[u], disc[v])
            if advanced:
                continue
            frame.pop()
            if frame:
                p = frame[-1][0]
                low[p] = min(low[p], low[u])
                if low[u] >= disc[p]:
                    blk = []
                    while stack and stack[-1] != pe:
                        blk.append(stack.pop())
                    if stack:
                        blk.append(stack.pop())
                    if blk:
                        blocks.append(blk)
    if stack:
        blocks.append(list(stack))
    return blocks


def mixed_block(plan: Plan):
    """Return a witness block mixing S and C edges, or None if cycle-free."""
    edges = sc_edges(plan)
    nodes = [p.id for p in plan.all_pieces()]
    for blk in _biconnected_blocks(nodes, edges):
        labels = {edges[i][2] for i in blk}
        if "S" in labels and "C" in labels:
            return [edges[i] for i in blk]
    return None


def certify(plan: Plan) -> Tuple[bool, Optional[list]]:
    """The classical test: no simple cycle mixes an S and a C edge."""
    blk = mixed_block(plan)
    return (blk is None), blk


# -- prefix-aware recertification -------------------------------------------
Frontier = Dict[str, int]


def seal(plan: Plan, frontier: Frontier) -> Plan:
    """Merge each workflow's entire uncommitted remainder into one piece.

    Lemma 1: this is the coarsest future-only coarsening, so every other one
    is a refinement of it.
    """
    out: Dict[str, List[Piece]] = {}
    for w, ps in plan.pieces.items():
        k = frontier.get(w, 0)
        head = ps[:k]
        tail = ps[k:]
        if tail:
            head = head + [tail[0].merged_with(tail[1:], k)]
        out[w] = head
    return Plan(out)


class Decision(Enum):
    ADMITTED = "admitted"              # proposal certified as-is
    ADMITTED_COARSENED = "coarsened"   # certified after merging the suffix
    REFUSED = "refused"                # no future-only repair exists


@dataclass
class Admission:
    decision: Decision
    plan: Optional[Plan]
    witness: Optional[list] = None
    merges: int = 0

    @property
    def ok(self) -> bool:
        return self.decision is not Decision.REFUSED


def _merge_cost(piece_a: Piece, piece_b: Piece) -> int:
    """Least footprint-duration area: prefer merging the smallest pair."""
    return len(set(piece_a.footprint) | set(piece_b.footprint))


def guided_coarsen(plan: Plan, frontier: Frontier) -> Tuple[Plan, int]:
    """Merge within the uncommitted suffix until certification passes.

    Repairs every mixed block each pass and, per workflow, merges the
    adjacent pair whose union adds the least footprint. Each pass strictly
    reduces the piece count, so it terminates; Theorem 2 guarantees it
    succeeds once sealing has certified.
    """
    cur = plan.copy()
    merges = 0
    while True:
        blk = mixed_block(cur)
        if blk is None:
            return cur, merges
        implicated = {u[0] for u, v, lab in blk if lab == "S"}
        implicated |= {v[0] for u, v, lab in blk if lab == "S"}
        progressed = False
        for w in sorted(implicated):
            ps = cur.pieces[w]
            k = frontier.get(w, 0)
            if len(ps) - k < 2:
                continue
            best, best_cost = None, None
            for i in range(k, len(ps) - 1):
                c = _merge_cost(ps[i], ps[i + 1])
                if best_cost is None or c < best_cost:
                    best, best_cost = i, c
            merged = ps[best].merged_with([ps[best + 1]], best)
            cur.pieces[w] = ps[:best] + [merged] + ps[best + 2:]
            for idx, p in enumerate(cur.pieces[w]):
                cur.pieces[w][idx] = Piece(p.workflow, idx, p.footprint,
                                           p.label, p.span)
            merges += 1
            progressed = True
        if not progressed:
            return cur, merges


def admit(proposed: Plan, frontier: Frontier) -> Admission:
    """Three-step admission (paper section 3.4).

    Certify the proposal; if it passes install it unchanged. Otherwise test
    the sealed plan -- if that fails, refuse with the witness cycle, because
    Theorem 2 says searching the exponentially many intermediate choppings is
    pointless. Otherwise coarsen the suffix to the finest plan that certifies.
    """
    ok, _ = certify(proposed)
    if ok:
        return Admission(Decision.ADMITTED, proposed)
    sealed_ok, witness = certify(seal(proposed, frontier))
    if not sealed_ok:
        return Admission(Decision.REFUSED, None, witness)
    repaired, merges = guided_coarsen(proposed, frontier)
    return Admission(Decision.ADMITTED_COARSENED, repaired, merges=merges)


def cycle_through_committed(witness: list, frontier: Frontier) -> bool:
    """Corollary 1: does every mixed cycle run through a committed piece?

    If so there is no future-only repair, and waiting for other workflows to
    drain does not help -- a committed workflow keeps its S clique and its C
    edges, because those interleavings really happened.
    """
    if not witness:
        return False
    for u, v, lab in witness:
        if lab != "S":
            continue
        for (w, i) in (u, v):
            if i < frontier.get(w, 0):
                return True
    return False
