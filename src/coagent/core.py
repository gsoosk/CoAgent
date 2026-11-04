"""Core state model: objects, three-phase tools, and write trajectories.

Implements the substrate MTPO is defined over (paper sections 5.1, 6.1, 6.3):
an object tree, footprint-declared tool calls, and a per-object write
trajectory whose materialization is the sigma-serial state of that object.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


def is_descendant(obj: str, prefix: str) -> bool:
    """Path-segment containment: 'd/geo-canary' is NOT inside 'd/geo'."""
    if obj == prefix:
        return False
    return obj.startswith(prefix if prefix.endswith("/") else prefix + "/")


def covered_by(obj: str, scope: str) -> bool:
    """True if a read recorded on ``scope`` covers ``obj`` (self or subtree)."""
    return obj == scope or is_descendant(obj, scope)


class WriteClass(Enum):
    """Paper section 2.1: idempotence separates the two write classes."""

    BLIND = "blind"   # PUT/DELETE: replays harmlessly, shadows everything before it
    RMW = "rmw"       # POST/append: composes with the prior state, replay is not safe


@dataclass(frozen=True)
class Footprint:
    """Objects a tool call reads and writes, declared at registration."""

    reads: frozenset[str] = frozenset()
    writes: frozenset[str] = frozenset()

    @staticmethod
    def of(reads: Iterable[str] = (), writes: Iterable[str] = ()) -> "Footprint":
        return Footprint(frozenset(reads), frozenset(writes))

    def touches(self, other: "Footprint") -> bool:
        """Two calls conflict on footprint when one's write set hits the other."""
        return bool(
            (self.writes & (other.reads | other.writes))
            or (other.writes & (self.reads | self.writes))
        )


@dataclass
class Tool:
    """A three-phase, footprint-declared tool (paper section 6.3).

    ``prepare`` captures whatever ``reverse`` will need, before ``exec`` runs.
    ``reverse`` is a self-contained undo the framework may invoke at any time.
    A tool with ``reversible=False`` is the ``unrecoverable`` tag of section 6.3.
    """

    name: str
    wclass: WriteClass
    # exec: (current_value, args) -> new_value
    exec_fn: Callable[[Any, Dict[str, Any]], Any]
    # prepare: (current_value, args) -> captured state for the inverse
    prepare_fn: Callable[[Any, Dict[str, Any]], Any] = lambda v, a: copy.deepcopy(v)
    # reverse: (current_value, captured, args) -> restored value
    reverse_fn: Optional[Callable[[Any, Any, Dict[str, Any]], Any]] = None
    # which sub-paths of the object this write determines. An empty set means
    # "the whole object". A blind write only shadows a lower-sigma write whose
    # coverage it contains -- setting `image` must not swallow a write to
    # `replicas`, even though both tools are blind.
    covers_fn: Optional[Callable[[Dict[str, Any]], frozenset]] = None
    reversible: bool = True
    is_read: bool = False

    def covers(self, args: Dict[str, Any]) -> frozenset:
        if self.covers_fn is None:
            return frozenset()          # whole object
        return self.covers_fn(args)

    def shadows(self, other: "Tool", my_args, other_args) -> bool:
        mine = self.covers(my_args)
        theirs = other.covers(other_args)
        if not mine:
            return True                 # whole-object write shadows anything
        if not theirs:
            return False                # cannot shadow a whole-object write
        return theirs <= mine

    def reverse(self, current: Any, captured: Any, args: Dict[str, Any]) -> Any:
        if not self.reversible:
            raise RuntimeError(f"tool {self.name} is tagged unrecoverable")
        if self.reverse_fn is None:
            # Default inverse for a blind write: restore what prepare captured.
            return copy.deepcopy(captured)
        return self.reverse_fn(current, captured, args)


@dataclass
class WriteOp:
    """One write, placed in a trajectory at its author's sigma rank."""

    obj: str
    tool: Tool
    args: Dict[str, Any]
    agent: str
    sigma: int
    step: int            # program order within the authoring agent
    t: float = 0.0       # physical arrival time (the paper's <_t)
    captured: Any = None  # filled by prepare, consumed by reverse
    applied_live: bool = False
    label: str = ""

    @property
    def key(self) -> Tuple[int, int]:
        """Trajectory order: sigma first, program order within an agent."""
        return (self.sigma, self.step)

    @property
    def wclass(self) -> WriteClass:
        return self.tool.wclass

    def apply_to(self, value: Any) -> Any:
        return self.tool.exec_fn(value, self.args)

    def describe(self) -> str:
        return self.label or f"{self.tool.name}({self.obj})"


class Trajectory:
    """The sigma-ordered write list for one object (paper section 5.1)."""

    def __init__(self, obj: str, initial: Any):
        self.obj = obj
        self.initial = copy.deepcopy(initial)
        self.entries: List[WriteOp] = []

    def insert(self, w: WriteOp) -> None:
        """Insert in sigma order; this is a physical act, not a version slot."""
        self.entries.append(w)
        self.entries.sort(key=lambda e: e.key)

    def materialize(self, sigma_limit: Optional[int] = None) -> Any:
        """M(o, sigma): compose every entry at or below ``sigma_limit``.

        A true composition, so RMW entries see the value their predecessors left.
        """
        value = copy.deepcopy(self.initial)
        for w in self.entries:
            if sigma_limit is None or w.sigma <= sigma_limit:
                value = w.apply_to(value)
        return value

    def materialize_premise(
        self, sigma_limit: int, reader: str, read_step: int
    ) -> Any:
        """The value the reader's read would return if re-issued now.

        A notification refreshes a premise, so it must reproduce that read's
        position: the reader's own writes that *preceded* the read are part of
        what it saw (read-your-own-writes), while the ones it issued *after*
        the read are not -- folding those back in would hand the agent a value
        no serial execution ever produced. The paper does not pin this down;
        see docs/reproduction-notes.md.
        """
        value = copy.deepcopy(self.initial)
        for w in self.entries:
            if w.sigma > sigma_limit:
                continue
            if w.agent == reader and w.step > read_step:
                continue
            value = w.apply_to(value)
        return value

    def suffix_above(self, sigma: int) -> List[WriteOp]:
        """Entries ranked strictly above ``sigma``, ascending."""
        return [w for w in self.entries if w.sigma > sigma]

    def shadowing_entry(self, w: "WriteOp") -> Optional["WriteOp"]:
        """A higher-sigma blind entry whose coverage contains ``w``'s.

        Only then is it sound to record ``w`` in the trajectory and never
        replay it onto the live copy (the Thomas write rule).
        """
        for other in reversed(self.suffix_above(w.sigma)):
            if other.wclass is not WriteClass.BLIND:
                continue
            if other.tool.shadows(w.tool, other.args, w.args):
                return other
        return None


class TargetSystem:
    """The single live copy agents act on: no fork, no buffer (section 3.4)."""

    def __init__(self, initial: Dict[str, Any]):
        self.initial = copy.deepcopy(initial)
        self.genesis = set(initial)   # objects present before any agent ran
        self.live: Dict[str, Any] = copy.deepcopy(initial)
        self.trajectories: Dict[str, Trajectory] = {
            k: Trajectory(k, v) for k, v in initial.items()
        }
        self.undo_count = 0
        self.apply_count = 0

    # -- object tree -------------------------------------------------------
    def ensure(self, obj: str, default: Any = None) -> None:
        """Objects instantiate lazily, on first mention by a toolcall.

        A lazily created object must not appear in ``live`` until something
        writes it: merely *reading* a non-existent object used to materialize
        a None-valued phantom that range reads then reported as a real child,
        which made the correctness oracle depend on which protocol ran.
        """
        if obj not in self.trajectories:
            self.trajectories[obj] = Trajectory(obj, default)
            if default is not None:
                self.live[obj] = copy.deepcopy(default)
                self.initial.setdefault(obj, copy.deepcopy(default))

    def traj(self, obj: str) -> Trajectory:
        self.ensure(obj)
        return self.trajectories[obj]

    def children(self, prefix: str) -> List[str]:
        """Tree query, used by range reads such as `list deployments`."""
        return sorted(
            k for k, v in self.live.items()
            if is_descendant(k, prefix) and v is not None
        )

    def visible_children(self, prefix: str, sigma: int) -> List[str]:
        """Children a reader at ``sigma`` may see: genesis objects, plus those
        created by a write ranked at or below ``sigma``."""
        out = []
        for k in self.trajectories:
            if not is_descendant(k, prefix):
                continue
            if k in self.genesis:
                out.append(k)
                continue
            if any(w.sigma <= sigma for w in self.trajectories[k].entries):
                out.append(k)
        return sorted(out)

    # -- live mutation -----------------------------------------------------
    def read_live(self, obj: str) -> Any:
        self.ensure(obj)
        return copy.deepcopy(self.live.get(obj))

    def apply_live(self, w: WriteOp) -> None:
        self.ensure(w.obj)
        w.captured = w.tool.prepare_fn(self.live.get(w.obj), w.args)
        self.live[w.obj] = w.apply_to(self.live.get(w.obj))
        w.applied_live = True
        self.apply_count += 1

    def undo_live(self, w: WriteOp) -> None:
        if not w.applied_live:
            return
        self.live[w.obj] = w.tool.reverse(
            self.live.get(w.obj), w.captured, w.args
        )
        w.applied_live = False
        self.undo_count += 1

    def reapply_live(self, w: WriteOp) -> None:
        self.apply_live(w)

    # -- reference oracles -------------------------------------------------
    def serial_state(self, sigma_limit: Optional[int] = None) -> Dict[str, Any]:
        """What the sigma-serial execution would leave on every object."""
        return {k: t.materialize(sigma_limit) for k, t in self.trajectories.items()}

    def snapshot(self) -> Dict[str, Any]:
        return copy.deepcopy(self.live)


def blind_set(name: str = "set") -> Tool:
    """Blind write on one field; the inverse restores only that field.

    Restoring a whole captured snapshot would silently revert a concurrent
    write to a different field of the same object, which is how a saga undo
    can destroy a committed peer write.
    """

    def _exec(value: Any, args: Dict[str, Any]) -> Any:
        out = copy.deepcopy(value) if isinstance(value, dict) else {}
        out[args["field"]] = args["value"]
        return out

    def _reverse(current: Any, captured: Any, args: Dict[str, Any]) -> Any:
        out = copy.deepcopy(current) if isinstance(current, dict) else {}
        field = args["field"]
        if isinstance(captured, dict) and field in captured:
            out[field] = copy.deepcopy(captured[field])
        else:
            out.pop(field, None)
        return out

    return Tool(
        name=name, wclass=WriteClass.BLIND, exec_fn=_exec,
        reverse_fn=_reverse,
        covers_fn=lambda a: frozenset({a["field"]}),
    )


def rmw_append(name: str = "append") -> Tool:
    """RMW write: append to a collection; inverse removes exactly what it added.

    Restoring a prepare-time snapshot would be wrong here, since a concurrent
    append must survive this undo. The inverse is element-wise on purpose.
    """

    def _exec(value: Any, args: Dict[str, Any]) -> Any:
        out = list(value or [])
        out.append(copy.deepcopy(args["item"]))
        return out

    def _reverse(current: Any, captured: Any, args: Dict[str, Any]) -> Any:
        out = list(current or [])
        item = args["item"]
        for i in range(len(out) - 1, -1, -1):
            if out[i] == item:
                del out[i]
                break
        return out

    return Tool(
        name=name,
        wclass=WriteClass.RMW,
        exec_fn=_exec,
        prepare_fn=lambda v, a: copy.deepcopy(v),
        reverse_fn=_reverse,
    )
