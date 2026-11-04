"""The CoAgent ToolSmith (paper section 6.4).

A privileged tool-building agent: unconstrained in *reading* the target
system, forbidden to mutate it. Because every conflict in the protocol is
caused by a write, an agent that never writes invalidates no premise and
needs no concurrency control at all -- which is what lets the ToolSmith roam
a bash-only target freely while every Worker write stays footprint-bound.

When a Worker hits a need no registered tool covers, it submits a request
(natural language, or the bash command it wanted to run). The ToolSmith
either returns an existing tool (its context carries the whole catalog, so
similar requests deduplicate) or synthesizes a new one: it marks the read and
write sets, registers any objects the tree lacks, attaches ``prepare`` and
``reverse``, and tags anything with no inverse as unrecoverable.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .core import Footprint, Tool, WriteClass


class ToolKind(Enum):
    SNAPSHOT_READ = "snapshot read"   # state cheap to copy; replay on a copy
    LIVE_READ = "live read"           # must run live; backed by undo
    UNDOABLE_WRITE = "undoable write"
    UNRECOVERABLE = "unrecoverable"   # no inverse at any cost


@dataclass
class ToolRecord:
    name: str
    kind: ToolKind
    reads: Tuple[str, ...]
    writes: Tuple[str, ...]
    tool: Optional[Tool]
    created_at_task: int
    reverse_desc: str = ""

    @property
    def is_write(self) -> bool:
        return self.kind in (ToolKind.UNDOABLE_WRITE, ToolKind.UNRECOVERABLE)


@dataclass
class SynthesisResult:
    record: ToolRecord
    reused: bool
    seconds: float


class ToolSmith:
    """Grows a footprint-declared, undoable tool library online."""

    # Seconds the ToolSmith spends per request. Synthesis means probing the
    # target and writing prepare/exec/reverse; a catalog hit is near-free.
    SYNTH_SECONDS = 20.0
    PROBE_SECONDS = 9.0
    REUSE_SECONDS = 1.6

    def __init__(self, bootstrap: Optional[List[Dict[str, Any]]] = None):
        self.catalog: Dict[str, ToolRecord] = {}
        self.objects: Set[str] = set()
        self.requests = 0
        self.reuses = 0
        self.seconds = 0.0
        self.writes_attempted = 0   # must stay zero: the ToolSmith never writes
        if bootstrap:
            for spec in bootstrap:
                self.register(task_index=0, **spec)

    # -- registration ------------------------------------------------------
    def register(
        self,
        name: str,
        kind: ToolKind,
        reads: Tuple[str, ...] = (),
        writes: Tuple[str, ...] = (),
        task_index: int = 0,
        reverse_desc: str = "",
        exec_fn: Optional[Callable] = None,
        reverse_fn: Optional[Callable] = None,
    ) -> ToolRecord:
        tool = None
        if kind is not ToolKind.SNAPSHOT_READ and kind is not ToolKind.LIVE_READ:
            tool = Tool(
                name=name,
                wclass=WriteClass.BLIND,
                exec_fn=exec_fn or (lambda v, a: dict(v or {}, **a)),
                reverse_fn=reverse_fn,
                reversible=(kind is not ToolKind.UNRECOVERABLE),
            )
        rec = ToolRecord(
            name=name, kind=kind, reads=tuple(reads), writes=tuple(writes),
            tool=tool, created_at_task=task_index, reverse_desc=reverse_desc,
        )
        self.catalog[name] = rec
        self.objects.update(reads)
        self.objects.update(writes)
        return rec

    # -- the A2A entry point ----------------------------------------------
    def request(self, spec: Dict[str, Any], task_index: int) -> SynthesisResult:
        """Serve a Worker's synthesis request, reusing the catalog when possible."""
        self.requests += 1
        name = spec["name"]
        if name in self.catalog:
            self.reuses += 1
            self.seconds += self.REUSE_SECONDS
            return SynthesisResult(self.catalog[name], True, self.REUSE_SECONDS)
        secs = self.SYNTH_SECONDS + (
            self.PROBE_SECONDS if not self._objects_known(spec) else 0.0
        )
        rec = self.register(task_index=task_index, **spec)
        self.seconds += secs
        return SynthesisResult(rec, False, secs)

    def _objects_known(self, spec: Dict[str, Any]) -> bool:
        wanted = set(spec.get("reads", ())) | set(spec.get("writes", ()))
        return wanted.issubset(self.objects)

    # -- the read-only invariant ------------------------------------------
    def attempt_write(self, obj: str) -> None:
        """Called if anything ever routes a mutation through the ToolSmith."""
        self.writes_attempted += 1
        raise PermissionError(
            f"ToolSmith is read-only on the target system (attempted {obj})"
        )

    # -- reporting ---------------------------------------------------------
    def composition(self) -> Dict[str, int]:
        out = {k.value: 0 for k in ToolKind}
        for rec in self.catalog.values():
            out[rec.kind.value] += 1
        return out

    def every_write_has_an_inverse(self) -> bool:
        return all(
            rec.tool is not None and rec.tool.reversible
            for rec in self.catalog.values()
            if rec.kind is ToolKind.UNDOABLE_WRITE
        )

    def audit_bash(self, command: str) -> Dict[str, Any]:
        """Classify a bash command into a footprint-declared tool spec.

        This is the 'or audits the command' path of section 6.4: the Worker
        says what it wanted to run and the ToolSmith marks its read and write
        sets rather than letting an untracked access through.
        """
        cmd = command.strip()
        verb = re.sub(r"^kubectl\s+", "", cmd).split()[0] if cmd else ""
        read_verbs = {"get", "describe", "top", "explain"}
        live_verbs = {"logs", "events"}
        write_verbs = {"apply", "scale", "set", "patch", "delete", "rollout",
                       "label", "annotate", "edit"}
        target = " ".join(cmd.split()[2:4]) or "cluster"
        if verb in live_verbs:
            kind = ToolKind.LIVE_READ
        elif verb in read_verbs:
            kind = ToolKind.SNAPSHOT_READ
        elif verb in write_verbs:
            kind = ToolKind.UNDOABLE_WRITE
        else:
            kind = ToolKind.UNRECOVERABLE
        return {
            "name": f"{verb}_{re.sub(r'[^a-z0-9]+', '_', target.lower())}".strip("_"),
            "kind": kind,
            "reads": (target,) if kind in (ToolKind.SNAPSHOT_READ,
                                           ToolKind.LIVE_READ) else (),
            "writes": () if kind in (ToolKind.SNAPSHOT_READ,
                                     ToolKind.LIVE_READ) else (target,),
        }
