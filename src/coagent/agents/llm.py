"""An LLM-backed Worker that drives the same middleware as ScriptedAgent.

Swapping ScriptedAgent for LLMAgent replaces the two knobs standing in for
the paper's agent assumptions with measurements: ``p_task_failure`` becomes
whether the model actually completes its task, and ``p_misjudge`` becomes
whether it actually recognises a notification as a premise conflict. That
second number is the entire correctness gap the paper reports, so it is the
one worth measuring rather than assuming.

The provider is pluggable (see ``backends``): OpenRouter/deepseek-v4-flash
reproduces the paper's setup, Anthropic measures a different model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core import Footprint
from ..runtime import AgentState, Call, CallKind, Runtime
from .backends import Completion, make_backend

SYSTEM = """You are one of several agents working concurrently on the same \
shared system. Work only through the tools listed below; never assume state \
you have not read.

The runtime may inject a NOTIFICATION telling you that an object you read has \
changed. When that happens, decide for yourself whether the change actually \
invalidates a premise your plan depends on.
  - If it does not, say so and carry on. Do not redo work.
  - If it does, repair only the operations that depended on the old value. \
Do not restart your task.

Object ids are full paths, exactly as returned by a listing tool. Never
invent one. Finish your whole task before replying done.

Reply with exactly one JSON object per turn and nothing else:
  {"thought": "...", "tool": "<name>", "args": {...}}
to act, or
  {"thought": "...", "done": true}
when your task is complete."""


@dataclass
class ToolSpec:
    """A registered tool as the Worker sees it, with its declared footprint."""

    name: str
    description: str
    tool: Optional[Any]
    kind: CallKind
    args_schema: Dict[str, str] = field(default_factory=dict)
    obj_param: str = "obj"
    prefix_param: Optional[str] = None
    # A range tool scans a fixed subtree. Binding it here rather than asking
    # the model for it keeps the footprint declared by the framework, which
    # is the whole point of a footprint-declared tool.
    fixed_prefix: Optional[str] = None

    def render(self) -> str:
        args = ", ".join(f"{k}: {v}" for k, v in self.args_schema.items())
        return f"- {self.name}({args}) -- {self.description}"


class LLMAgent:
    """A Worker whose plan comes from a model instead of a script."""

    def __init__(
        self,
        name: str,
        task: str,
        tools: List[ToolSpec],
        backend: Any = None,
        provider: str = "openrouter",
        model: Optional[str] = None,
        max_rounds: int = 14,
        **backend_kw: Any,
    ):
        self.name = name
        self.task = task
        self.tools = {t.name: t for t in tools}
        self.backend = backend or make_backend(provider, model=model, **backend_kw)
        self.max_rounds = max_rounds
        self.messages: List[Dict[str, str]] = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.rounds = 0
        self.refusals = 0
        self.unparsed = 0
        self.transcript: List[Dict[str, Any]] = []
        # (agent, kind, label, t, dur) per dispatched call, for timelines
        self.events: List[Dict[str, Any]] = []

    @property
    def model(self) -> str:
        return self.backend.model

    def _system(self) -> str:
        catalog = "\n".join(t.render() for t in self.tools.values())
        return f"{SYSTEM}\n\nTools available:\n{catalog}"

    def _reset(self) -> None:
        self.messages = [{"role": "user", "content": self.task}]

    def _complete(self) -> Completion:
        out = self.backend.complete(self._system(), self.messages)
        self.prompt_tokens += out.prompt_tokens
        self.completion_tokens += out.completion_tokens
        if out.parsed.get("_refused"):
            self.refusals += 1
        if out.parsed.get("_unparsed"):
            self.unparsed += 1
        self.messages.append({"role": "assistant",
                              "content": out.raw or json.dumps(out.parsed)})
        self.transcript.append({"agent": self.name, "reply": out.parsed,
                                "latency": out.latency})
        return out

    def _drain(self, st: AgentState) -> None:
        """Consume notifications before the next action (assumption A2)."""
        while st.inbox:
            note = st.inbox.pop(0)
            st.notes_seen += 1
            self.messages.append({
                "role": "user",
                "content": (
                    f"NOTIFICATION: object '{note.obj}' was written by a "
                    f"concurrent agent ordered before you. Its value as of "
                    f"your position is now:\n{json.dumps(note.new_value, default=str)}\n"
                    f"Decide whether this invalidates a premise of your plan. "
                    f"If it does, repair only the affected operations."
                ),
            })

    def run(self, rt: Runtime, st: AgentState):
        self._reset()
        yield from self._loop(rt, st, self.max_rounds)

    def repair_run(self, rt: Runtime, st: AgentState):
        """Re-opened by a late notification; the context is kept, not reset."""
        if not self.messages:
            self._reset()
        yield from self._loop(rt, st, 6)

    def _loop(self, rt: Runtime, st: AgentState, budget: int):
        for _ in range(budget):
            self._drain(st)
            out = self._complete()
            self.rounds += 1
            reply = out.parsed
            if reply.get("done"):
                return
            spec = self.tools.get(reply.get("tool", ""))
            if spec is None:
                self.messages.append({
                    "role": "user",
                    "content": (f"No such tool: {reply.get('tool')!r}. "
                                f"Choose one from the list, or reply "
                                f'{{"done": true}} if you are finished.'),
                })
                continue
            args = reply.get("args", {}) or {}
            obj = args.get(spec.obj_param, "")
            prefix = spec.fixed_prefix or (
                args.get(spec.prefix_param) if spec.prefix_param else None
            )
            call = Call(
                kind=spec.kind,
                obj=obj,
                tool=spec.tool,
                args={k: v for k, v in args.items()
                      if k not in (spec.obj_param, spec.prefix_param)},
                footprint=(
                    Footprint.of(reads=[prefix or obj])
                    if spec.kind is CallKind.READ
                    else Footprint.of(writes=[obj])
                ),
                duration=out.latency,          # real measured latency
                tokens=out.prompt_tokens or 400,
                label=f"{spec.name}({obj or prefix})",
                prefix=prefix,
            )
            # stamp the call with the virtual clock so the run can be drawn
            self.events.append({
                "agent": self.name, "kind": spec.kind.value,
                "label": call.label, "t": rt.now, "dur": call.duration,
            })
            result = yield call
            self.messages.append({
                "role": "user",
                "content": f"RESULT of {spec.name}: {json.dumps(result, default=str)}",
            })
