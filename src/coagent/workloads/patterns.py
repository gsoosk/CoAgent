"""Reusable shapes for contended cells.

The paper hand-constructs a partner agent for each task "chosen so the pair
exhibits a textbook concurrency anomaly" but does not publish them. These
three builders cover the anomaly families that make protocols diverge:

  sweep_derive  -- range scan vs. point-read-then-create (phantom + stale read)
  lost_update   -- two read-modify-writes on one object (lost update)
  rmw_summary   -- blind writes vs. a non-idempotent append derived from them
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..agents.scripted import PlanStep, ScriptedAgent
from ..core import Tool, WriteClass, rmw_append
from ..runtime import CallKind
from .base import Cell


def _blind(field: str) -> Tool:
    def _exec(value, args):
        out = dict(value or {})
        out[field] = args[field]
        return out

    return Tool(name=f"set_{field}", wclass=WriteClass.BLIND, exec_fn=_exec)


def _create() -> Tool:
    return Tool(
        name="create",
        wclass=WriteClass.RMW,
        exec_fn=lambda v, a: copy.deepcopy(a["spec"]),
        prepare_fn=lambda v, a: copy.deepcopy(v),
        reverse_fn=lambda cur, cap, a: copy.deepcopy(cap),
    )


def sweep_derive(
    name: str,
    suite: str,
    prefix: str,
    field: str,
    good: Any,
    source: str,
    target: str,
    initial: Dict[str, Any],
    anomaly: str = "phantom insert + stale read (rw antidependency cycle)",
    description: str = "",
    spec_extra: Optional[Dict[str, Any]] = None,
) -> Cell:
    """A sweeps ``prefix`` fixing ``field``; B derives ``target`` from ``source``."""

    fix_tool = _blind(field)
    create_tool = _create()
    extra = spec_extra or {}

    def build(oracle, rng, **kw):
        pf = kw.get("p_task_failure", 0.0)
        rtr = kw.get("p_redo_truncate", 0.0)

        def a_fixes(view) -> List[PlanStep]:
            scope = view.get("scope") or {}
            steps = []
            for obj in sorted(scope):
                spec = scope[obj] or {}
                if spec.get(field) != good:
                    steps.append(
                        PlanStep(
                            kind=CallKind.WRITE, obj=obj, tool=fix_tool,
                            args={field: good}, writes=(obj,),
                            duration=5.0, tokens=300,
                            label=f"fix {obj.split('/')[-1]}.{field}",
                            depends_on=("scope",),
                        )
                    )
            return steps

        agent_a = ScriptedAgent(
            name="A-sweep", oracle=oracle, rng=rng, p_task_failure=pf, p_redo_truncate=rtr,
            plan=[
                PlanStep(kind=CallKind.READ, prefix=prefix, reads=(prefix,),
                         premise="scope", duration=4.0, tokens=600,
                         label=f"scan {prefix}"),
                PlanStep(kind=CallKind.THINK, duration=3.0, tokens=260,
                         label="plan repairs"),
                PlanStep(kind=CallKind.WRITE, expand=a_fixes, label="apply fixes",
                     depends_on=("scope",)),
                PlanStep(kind=CallKind.READ, prefix=prefix, reads=(prefix,),
                         duration=4.0, tokens=400, label="verify",
                         repairable=False),
            ],
        )

        agent_b = ScriptedAgent(
            name="B-derive", oracle=oracle, rng=rng, p_task_failure=pf, p_redo_truncate=rtr,
            plan=[
                PlanStep(kind=CallKind.READ, obj=source, reads=(source,),
                         premise="src", duration=4.5, tokens=400,
                         label=f"read {source.split('/')[-1]}"),
                PlanStep(kind=CallKind.THINK, duration=3.0, tokens=260,
                         label="plan derived object"),
                PlanStep(
                    kind=CallKind.WRITE, obj=target, tool=create_tool,
                    args=lambda view: {
                        "spec": dict(
                            extra,
                            **{field: (view.get("src") or {}).get(field)},
                        )
                    },
                    writes=(target,), duration=12.0, tokens=520,
                    label=f"create {target.split('/')[-1]}",
                    depends_on=("src",),
                    repair_with=[
                        PlanStep(
                            kind=CallKind.WRITE, obj=target, tool=fix_tool,
                            args=lambda view: {
                                field: (view.get("src") or {}).get(field)
                            },
                            writes=(target,), duration=5.0, tokens=300,
                            label=f"patch {target.split('/')[-1]}.{field}",
                            repairable=False,
                        )
                    ],
                ),
            ],
        )
        return [agent_a, agent_b]

    def invariant(final: Dict[str, Any]) -> bool:
        tgt, src = final.get(target), final.get(source)
        if tgt is None or src is None:
            return False
        return tgt.get(field) == src.get(field) == good

    llm_spec = {
        "tasks": [
            f"Every object under '{prefix}' must have {field} = {good!r}. "
            f"List them, find every one whose {field} differs, and fix each "
            f"one. Then verify your work.",
            f"Read '{source}' and create a new object '{target}' that "
            f"mirrors its {field}"
            + (f", also setting {extra}" if extra else "")
            + f". The new object must end up carrying the {field} that "
            f"'{source}' is actually running.",
        ],
        "tools": [
            {"name": "list_objects", "kind": "read", "prefix": prefix,
             "description": f"list every object under '{prefix}' with its fields"},
            {"name": "get_object", "kind": "read",
             "description": "read one object's fields"},
            {"name": f"set_{field}", "kind": "write", "tool": fix_tool,
             "args": {field: f"new {field} value"},
             "description": f"set an object's {field} (reversible)"},
            {"name": "create_object", "kind": "write", "tool": create_tool,
             "args": {"spec": "object of fields for the new object"},
             "description": "create a new object (reversible)"},
        ],
    }

    def anomaly_free(final, refs):
        """The derived object must mirror its source, whatever else broke."""
        tgt, src = final.get(target), final.get(source)
        if tgt is None or src is None:
            return False
        return tgt.get(field) == src.get(field)

    return Cell(
        name=name, suite=suite, anomaly=anomaly, description=description,
        initial=initial, build_agents=build, semantic_invariant=invariant,
        llm_spec=llm_spec, anomaly_free=anomaly_free,
    )


def lost_update(
    name: str,
    suite: str,
    obj: str,
    initial: Dict[str, Any],
    field: str,
    a_fn: Callable[[Any], Any],
    b_fn: Callable[[Any], Any],
    a_label: str,
    b_label: str,
    description: str = "",
    a_desc: str = "",
    b_desc: str = "",
) -> Cell:
    """Both agents read ``obj`` and write a value derived from what they read."""

    tool = _blind(field)

    def build(oracle, rng, **kw):
        pf = kw.get("p_task_failure", 0.0)
        rtr = kw.get("p_redo_truncate", 0.0)

        def mk(agent_name, fn, label):
            return ScriptedAgent(
                name=agent_name, oracle=oracle, rng=rng, p_task_failure=pf, p_redo_truncate=rtr,
                plan=[
                    PlanStep(kind=CallKind.READ, obj=obj, reads=(obj,),
                             premise="cur", duration=4.0, tokens=420,
                             label=f"read {obj.split('/')[-1]}"),
                    PlanStep(kind=CallKind.THINK, duration=3.5, tokens=280,
                             label="compute update"),
                    PlanStep(
                        kind=CallKind.WRITE, obj=obj, tool=tool,
                        args=lambda view: {
                            field: fn((view.get("cur") or {}).get(field))
                        },
                        writes=(obj,), duration=7.0, tokens=360,
                        label=label, depends_on=("cur",),
                    ),
                ],
            )

        return [mk("A-" + a_label, a_fn, a_label), mk("B-" + b_label, b_fn, b_label)]

    llm_spec = {
        "tasks": [
            f"Read '{obj}' and update its {field}: {a_desc or a_label}. "
            f"Your new value must be computed from the {field} the object "
            f"actually holds.",
            f"Read '{obj}' and update its {field}: {b_desc or b_label}. "
            f"Your new value must be computed from the {field} the object "
            f"actually holds.",
        ],
        "tools": [
            {"name": "get_object", "kind": "read",
             "description": "read one object's fields"},
            {"name": f"set_{field}", "kind": "write", "tool": tool,
             "args": {field: f"new {field} value"},
             "description": f"set the object's {field} (reversible)"},
        ],
    }

    def anomaly_free(final, refs):
        """Neither update may be lost: the value must be a serial outcome."""
        got = (final.get(obj) or {}).get(field)
        legal = {(r.get(obj) or {}).get(field) for r in refs}
        return got in legal

    return Cell(
        name=name, suite=suite,
        anomaly="lost update (ww + rw on one object)",
        description=description, initial=initial, build_agents=build,
        llm_spec=llm_spec, anomaly_free=anomaly_free,
    )


def rmw_summary(
    name: str,
    suite: str,
    prefix: str,
    field: str,
    good: Any,
    log_obj: str,
    initial: Dict[str, Any],
    description: str = "",
    subject: str = "",
) -> Cell:
    """Both agents append to one log; B's entry derives from a value A overwrites.

    This exercises the parts of MTPO that blind writes never reach. The append
    is non-idempotent, so its inverse has to be element-wise (a concurrent
    append must survive the undo), the log's trajectory has to be reordered
    rather than shadowed when A's entry lands after B's, and B's repair has to
    retract its stale record instead of adding a second one.
    """

    fix_tool = _blind(field)
    append_tool = rmw_append("append_log")

    def _replace_exec(value, args):
        out = [x for x in (value or []) if x != args.get("old")]
        out.append(copy.deepcopy(args["item"]))
        return out

    def _replace_reverse(cur, cap, args):
        out = [x for x in (cur or []) if x != args["item"]]
        if args.get("old") is not None:
            out.append(copy.deepcopy(args["old"]))
        return out

    replace_tool = Tool(
        name="replace_log_entry",
        wclass=WriteClass.RMW,
        exec_fn=_replace_exec,
        prepare_fn=lambda v, a: copy.deepcopy(v),
        reverse_fn=_replace_reverse,
    )

    subj = subject or sorted(k for k in initial if k.startswith(prefix))[0]

    def build(oracle, rng, **kw):
        pf = kw.get("p_task_failure", 0.0)
        rtr = kw.get("p_redo_truncate", 0.0)

        agent_a = ScriptedAgent(
            name="A-remediate", oracle=oracle, rng=rng, p_task_failure=pf, p_redo_truncate=rtr,
            plan=[
                PlanStep(kind=CallKind.READ, obj=subj, reads=(subj,),
                         premise="cur", duration=4.0, tokens=440,
                         label=f"read {subj.split('/')[-1]}"),
                PlanStep(kind=CallKind.THINK, duration=3.0, tokens=260,
                         label="plan remediation"),
                PlanStep(kind=CallKind.WRITE, obj=subj, tool=fix_tool,
                         args={field: good}, writes=(subj,), duration=6.0,
                         tokens=320, label=f"fix {subj.split('/')[-1]}",
                         repairable=False),
                PlanStep(kind=CallKind.WRITE, obj=log_obj, tool=append_tool,
                         args={"item": {"by": "A", "event": "remediated",
                                        "obj": subj}},
                         writes=(log_obj,), duration=6.0, tokens=300,
                         label="append remediation record", repairable=False),
            ],
        )

        def record(view):
            return {"item": {"by": "B", "event": "observed",
                             field: (view.get("src") or {}).get(field)}}

        def replacement(view):
            prev = (view.get("rec") or {}).get("item")
            return dict(record(view), old=prev)

        agent_b = ScriptedAgent(
            name="B-audit", oracle=oracle, rng=rng, p_task_failure=pf, p_redo_truncate=rtr,
            plan=[
                PlanStep(kind=CallKind.READ, obj=subj, reads=(subj,),
                         premise="src", duration=4.5, tokens=440,
                         label=f"audit-read {subj.split('/')[-1]}"),
                # long enough that B's record lands after A's, so a stale
                # read produces an interleaving neither serial order allows
                PlanStep(kind=CallKind.THINK, duration=12.0, tokens=280,
                         label="compose audit record"),
                PlanStep(kind=CallKind.WRITE, obj=log_obj, tool=append_tool,
                         args=record, writes=(log_obj,), duration=9.0,
                         tokens=420, label="append audit record",
                         depends_on=("src",), record_as="rec",
                         repair_with=[
                             PlanStep(kind=CallKind.WRITE, obj=log_obj,
                                      tool=replace_tool, args=replacement,
                                      writes=(log_obj,), duration=6.0,
                                      tokens=320,
                                      label="retract and re-file audit record",
                                      record_as="rec", repairable=False)
                         ]),
            ],
        )
        return [agent_a, agent_b]

    # The reference states are produced by the scripted agents, whose records
    # have a fixed key schema. A model writing a semantically identical record
    # with different keys would be scored wrong for no protocol-relevant
    # reason, so the exact shape is part of the task.
    llm_spec = {
        "tasks": [
            f"Read '{subj}'. Set its {field} to {good!r}. Then append exactly "
            f"this record to the log '{log_obj}':\n"
            f'  {{"by": "A", "event": "remediated", "obj": "{subj}"}}\n'
            f"Use those exact keys and values.",
            f"Read '{subj}', then append exactly one audit record to the log "
            f"'{log_obj}' with this exact shape:\n"
            f'  {{"by": "B", "event": "observed", "{field}": <the {field} '
            f'value {subj} is actually running>}}\n'
            f"Use those exact keys. If you later learn that value changed, "
            f"correct your existing record with replace_log_entry rather than "
            f"filing a second one -- the log must end up with exactly one "
            f"record whose \"by\" is \"B\".",
        ],
        "tools": [
            {"name": "get_object", "kind": "read",
             "description": "read one object's fields"},
            {"name": f"set_{field}", "kind": "write", "tool": fix_tool,
             "args": {field: f"new {field} value"},
             "description": f"set an object's {field} (reversible)"},
            {"name": "append_log", "kind": "write", "tool": append_tool,
             "args": {"item": "the record object to append"},
             "description": "append one record to a log (reversible)"},
            {"name": "replace_log_entry", "kind": "write", "tool": replace_tool,
             "args": {"old": "the record to retract", "item": "its replacement"},
             "description": "retract a record and file a corrected one"},
        ],
    }

    def anomaly_free(final, refs):
        """The log must read as one of the two serial orders.

        Here the anomaly is the *pairing* of order and observed value, not
        either alone. Both orders are legal -- B first and it observes the
        pre-remediation value, A first and it observes the repaired one --
        but A's record followed by B reporting the stale value is an
        interleaving neither serial order produces. Checking only B's value
        would pass that; checking only the order would too.
        """
        def shape(log):
            return [
                (e.get("by"), e.get(field))
                for e in (log or []) if isinstance(e, dict)
            ]

        return shape(final.get(log_obj)) in [
            shape(r.get(log_obj)) for r in refs
        ]

    return Cell(
        name=name, suite=suite,
        anomaly="non-idempotent append + stale premise (RMW composition)",
        description=description, initial=initial, build_agents=build,
        llm_spec=llm_spec, anomaly_free=anomaly_free,
    )
