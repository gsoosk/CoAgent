"""Workloads where the order of an agent's own steps decides how much is cut.

On the twelve benchmark cells no reordering can help: an exhaustive search
over every order and every cut finds nothing finer than the best cut in the
declared order, because every step touches a contended object. These three
cells have what those lack: private work sitting between two contended
steps. In the declared order the certifier must merge the contended steps
and everything between them into one piece. An order that puts the
contended steps next to each other leaves the private work outside it.

They differ in what the agent has to notice to reorder correctly:

``ro-free``  the private steps are independent blind writes; any order works
``ro-dep``   the private steps use the value of the first contended read;
             they can move after the last contended step, not before the
             first
``ro-trap``  the last contended write uses a private read; the read can move
             to the front, and moving it to the back breaks the task

Kept out of ``CELLS`` on purpose: they are designed to show one effect, and
the benchmark matrix and its figures should not change because they exist.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..agents.scripted import PlanStep, ScriptedAgent
from ..runtime import CallKind
from .base import Cell
from .chopping_cells import X, Y, set_v

__all__ = ["REORDER_CELLS"]

REORDER_CELLS: Dict[str, Cell] = {}

_TOOLS = [
    {"name": "get_object", "kind": "read",
     "description": "read one object's fields"},
    {"name": "set_v", "kind": "write", "tool": set_v,
     "args": {"v": "the new v value"},
     "description": "set an object's v (reversible)"},
]


def _v(view: Dict[str, Any], premise: str) -> str:
    return str((view.get(premise) or {}).get("v"))


def _write(obj: str, value: Any, label: str, depends=()) -> PlanStep:
    return PlanStep(kind=CallKind.WRITE, obj=obj, tool=set_v,
                    args=(value if callable(value) else {"v": value}),
                    writes=(obj,), duration=3.0, label=label,
                    depends_on=tuple(depends))


def _read(obj: str, premise: str, label: str) -> PlanStep:
    return PlanStep(kind=CallKind.READ, obj=obj, reads=(obj,),
                    premise=premise, duration=3.0, label=label)


def _agent(name: str, plan: List[PlanStep], oracle, rng, kw) -> ScriptedAgent:
    return ScriptedAgent(name=name, oracle=oracle, rng=rng, plan=plan,
                         p_task_failure=kw.get("p_task_failure", 0.0),
                         p_redo_truncate=kw.get("p_redo_truncate", 0.0))


def _mirror(oracle, rng, kw) -> ScriptedAgent:
    """B: read x, then publish what it saw to y and to a private record."""
    return _agent("B-mirror", [
        _read(X, "x", "read x"),
        _write(Y, lambda view: {"v": _v(view, "x") + "-seen"}, "write y",
               depends=("x",)),
        _write("private/b", lambda view: {"v": _v(view, "x") + "-seen"},
               "write private record", depends=("x",)),
    ], oracle, rng, kw)


def _matches_a_serial_order(keys):
    def check(final, refs):
        return any(all(final.get(k) == r.get(k) for k in keys) for r in refs)
    return check


def _register(cell: Cell) -> None:
    REORDER_CELLS[cell.name] = cell


# -- ro-free -------------------------------------------------------------
def _free_agents(oracle, rng, **kw):
    a = _agent("A-release", [
        _write(X, "rel-1", "write x"),
        _write("notes/a1", "n1", "write note 1"),
        _write("notes/a2", "n2", "write note 2"),
        _write("notes/a3", "n3", "write note 3"),
        _write(Y, "rel-2", "write y"),
    ], oracle, rng, kw)
    return [a, _mirror(oracle, rng, kw)]


_B_TASK = (f"Read '{X}' to get its v. Then write v='<that v>-seen' to "
           f"'{Y}', and write the same string as v to 'private/b'.")

_register(Cell(
    name="ro-free", suite="reorder",
    anomaly="two contended writes with independent private writes between",
    initial={X: {"v": "init"}, Y: {"v": "init"}},
    build_agents=_free_agents,
    anomaly_free=_matches_a_serial_order((X, Y, "private/b")),
    llm_spec={"tasks": [
        f"Complete this checklist:\n"
        f"- write v='rel-1' to '{X}'\n"
        f"- write v='n1' to 'notes/a1'\n"
        f"- write v='n2' to 'notes/a2'\n"
        f"- write v='n3' to 'notes/a3'\n"
        f"- write v='rel-2' to '{Y}'",
        _B_TASK,
    ], "tools": _TOOLS},
))


# -- ro-dep --------------------------------------------------------------
def _dep_agents(oracle, rng, **kw):
    a = _agent("A-copy", [
        _read(X, "x", "read x"),
        _write("notes/a1", lambda view: {"v": _v(view, "x") + "-copy1"},
               "write note 1", depends=("x",)),
        _write("notes/a2", lambda view: {"v": _v(view, "x") + "-copy2"},
               "write note 2", depends=("x",)),
        _write(Y, "A-final", "write y"),
    ], oracle, rng, kw)
    b = _agent("B-publish", [
        _write(X, "B", "write x"),
        _write(Y, "B-final", "write y"),
    ], oracle, rng, kw)
    return [a, b]


_register(Cell(
    name="ro-dep", suite="reorder",
    anomaly="private writes between a contended read and a contended write, "
            "derived from the read",
    initial={X: {"v": "init"}, Y: {"v": "init"}},
    build_agents=_dep_agents,
    anomaly_free=_matches_a_serial_order((X, Y, "notes/a1", "notes/a2")),
    llm_spec={"tasks": [
        f"Read '{X}' to get its v. Write v='<that v>-copy1' to 'notes/a1' "
        f"and v='<that v>-copy2' to 'notes/a2'. Write v='A-final' to '{Y}'.",
        f"Write v='B' to '{X}'. Then write v='B-final' to '{Y}'.",
    ], "tools": _TOOLS},
))


# -- ro-trap -------------------------------------------------------------
def _trap_agents(oracle, rng, **kw):
    a = _agent("A-config", [
        _write(X, "A", "write x"),
        _read("config/a", "cfg", "read config"),
        _write(Y, lambda view: {"v": _v(view, "cfg")}, "write y",
               depends=("cfg",)),
    ], oracle, rng, kw)
    return [a, _mirror(oracle, rng, kw)]


_register(Cell(
    name="ro-trap", suite="reorder",
    anomaly="a private read between two contended writes, feeding the second",
    initial={X: {"v": "init"}, Y: {"v": "init"}, "config/a": {"v": "k7"}},
    build_agents=_trap_agents,
    anomaly_free=_matches_a_serial_order((X, Y, "private/b")),
    llm_spec={"tasks": [
        f"Write v='A' to '{X}'. Read 'config/a' to get its v. Write that v "
        f"to '{Y}'.",
        _B_TASK,
    ], "tools": _TOOLS},
))
