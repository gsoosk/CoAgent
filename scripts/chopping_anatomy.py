#!/usr/bin/env python3
"""Chopping anatomy: watch certification chop, merge, and rule on replans.

The matrix experiments score outcomes. This one records the *decisions* that
produce them, so the mechanism can be inspected rather than inferred.

Design
------
Five scenarios, each built to exercise exactly one decision path:

  clean-cut        the declared chop certifies as-is; locks are released
                   across the reasoning gap
  forced-merge     the declared chop has a mixed S/C cycle; certification
                   merges it back to one piece per workflow
  admitted-replan  an agent reaches for an undeclared object mid-flight and
                   the revision certifies as proposed
  coarsened-replan the revision certifies only after the uncommitted suffix
                   of *both* workflows is merged
  refused-replan   the cycle runs through a committed piece; no future-only
                   repair exists (Corollary 1)

Each scenario runs once with scripted agents (deterministic, for the
walkthrough) and N times with real model-driven agents (to see whether the
same decision arises when nothing is scripted).

Nothing in the protocol is modified. ``AuditedChopping`` subclasses the
certifier and records what it was asked and what it decided: the plan as
declared, its SC graph, the certified plan, every lock hold, and for every
mid-flight revision the proposed plan, the sealed plan, the verdict and the
witness cycle.

Every verdict is then checked against a brute-force oracle that enumerates
all interleavings of the pieces and tests workflow-level conflict
serializability, so "the certifier said no" can be compared with "an unsafe
interleaving really exists".
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

from coagent.agents.scripted import PlanStep, ScriptedAgent      # noqa: E402
from coagent.chopping import (                                   # noqa: E402
    Decision, Piece, Plan, admit, certify, cycle_through_committed,
    sc_edges, seal,
)
from coagent.protocols.chopping import (                         # noqa: E402
    CertifiedChopping, accesses_for,
)
from coagent.runner import TrialConfig, _build                   # noqa: E402
from coagent.runtime import CallKind, Runtime                    # noqa: E402
from coagent.workloads import CELLS                              # noqa: E402
from coagent.workloads.base import Cell, normalize               # noqa: E402
from coagent.workloads.chopping_cells import X, Y, set_v         # noqa: E402
from test_chopping import all_interleavings_serializable        # noqa: E402

Z = "shared/z"
NOTES = "private/a-notes"


# -- serialisation ----------------------------------------------------------
def _acc(a) -> Dict[str, str]:
    key = a.lo if a.hi == a.lo + "\x00" else a.lo + "*"
    return {"m": "w" if a.is_write else "r", "k": key}


def _pid(p) -> str:
    return f"{p[0]}#{p[1]}"


def snapshot(plan: Plan) -> Dict[str, Any]:
    """A plan and its SC graph as plain data."""
    pieces = {
        w: [{"id": _pid(p.id), "label": p.label, "span": p.span,
             "fp": sorted((_acc(a) for a in p.footprint),
                          key=lambda d: (d["k"], d["m"]))}
            for p in ps]
        for w, ps in plan.pieces.items()
    }
    by_id = {p.id: p for p in plan.all_pieces()}
    edges = []
    for u, v, lab in sc_edges(plan):
        on = []
        if lab == "C":
            on = sorted({_acc(a)["k"] for a in by_id[u].footprint
                         for b in by_id[v].footprint if a.conflicts(b)})
        edges.append({"a": _pid(u), "b": _pid(v), "kind": lab, "on": on})
    ok, witness = certify(plan)
    return {
        "pieces": pieces, "edges": edges, "certifies": ok,
        "witness": [{"a": _pid(u), "b": _pid(v), "kind": lab}
                    for u, v, lab in (witness or [])],
        # ground truth: does *any* interleaving of these pieces break
        # workflow-level conflict serializability?
        "oracle_safe": all_interleavings_serializable(plan),
    }


# -- the audit wrapper ------------------------------------------------------
class AuditedChopping(CertifiedChopping):
    """The certifier, unchanged, with a record of everything it decided."""

    name = "chopping-audited"

    def __init__(self) -> None:
        super().__init__()
        self.log: List[Dict[str, Any]] = []
        self._rt: Optional[Runtime] = None

    def _now(self) -> float:
        return round(self._rt.now, 3) if self._rt is not None else 0.0

    def _ev(self, kind: str, **kw: Any) -> None:
        self.log.append({"t": self._now(), "ev": kind, **kw})

    # launch-time certification
    def _ensure_plan(self, rt: Runtime) -> None:
        if self.certified:
            return
        self._rt = rt
        declared = Plan({
            name: [Piece(name, i, frozenset(fp), label)
                   for i, (fp, label) in enumerate(
                       getattr(st.agent, "declare_pieces", lambda: [])())]
            or [Piece(name, 0, frozenset(), "whole")]
            for name, st in rt.states.items()
        })
        self._ev("declared", plan=snapshot(declared))
        super()._ensure_plan(rt)
        self._ev("certified", plan=snapshot(self.plan),
                 merges=self.initial_merges)

    # lock holds
    def _acquire(self, piece, owner):
        super()._acquire(piece, owner)
        self._ev("lock", wf=owner, piece=_pid(piece.id),
                 fp=[_acc(a) for a in piece.footprint])

    def _release(self, owner):
        held = any(h.owner == owner for h in self.locks)
        super()._release(owner)
        if held:
            self._ev("unlock", wf=owner)

    def _commit_piece(self, rt, st):
        idx = self.active.get(st.name)
        super()._commit_piece(rt, st)
        if idx is not None:
            self._ev("commit", wf=st.name, piece=f"{st.name}#{idx}",
                     frontier=dict(self.frontier))

    # calls
    def dispatch(self, rt, st, call):
        self._rt = rt
        out = super().dispatch(rt, st, call)
        refused = isinstance(out.value, dict) and out.value.get("refused")
        self._ev("blocked" if out.blocked else "call", wf=st.name,
                 op=call.kind.value, obj=call.obj or call.prefix,
                 label=call.label, on=out.blocked_on, refused=bool(refused))
        return out

    def on_think(self, rt, st, call):
        self._rt = rt
        self._ev("think", wf=st.name, label=call.label)
        super().on_think(rt, st, call)

    def on_commit(self, rt, st):
        self._rt = rt
        super().on_commit(rt, st)
        self._ev("done", wf=st.name)

    # mid-flight revision
    def _revise(self, rt, st, call, piece):
        self._rt = rt
        proposed = self.plan.copy()
        extra = frozenset(set(piece.footprint) | set(accesses_for(call)))
        proposed.pieces[st.name][piece.index] = Piece(
            st.name, piece.index, extra, piece.label)
        frontier = dict(self.frontier)
        res = admit(proposed, frontier)        # pure; super() repeats it
        sealed = seal(proposed, frontier)
        rec = {
            "wf": st.name, "piece": _pid(piece.id),
            "access": [_acc(a) for a in accesses_for(call)],
            "frontier": frontier,
            "proposed": snapshot(proposed),
            "sealed": snapshot(sealed),
            "decision": res.decision.value,
            "witness": [{"a": _pid(u), "b": _pid(v), "kind": lab}
                        for u, v, lab in (res.witness or [])],
            "through_committed": bool(
                res.decision is Decision.REFUSED
                and cycle_through_committed(res.witness, frontier)),
        }
        out = super()._revise(rt, st, call, piece)
        rec["after"] = snapshot(self.plan)
        self._ev("revision", **rec)
        return out


# -- scenarios ----------------------------------------------------------------
def _peer(oracle, rng, pf, rtr):
    """B writes x then reads y, as in the impossibility witness."""
    return ScriptedAgent(
        name="B", oracle=oracle, rng=rng, p_task_failure=pf,
        p_redo_truncate=rtr,
        plan=[
            PlanStep(kind=CallKind.WRITE, obj=X, tool=set_v,
                     args={"v": "from-B"}, writes=(X,), duration=3.0,
                     label="write x"),
            PlanStep(kind=CallKind.READ, obj=Y, reads=(Y,), premise="y",
                     duration=3.0, label="read y"),
        ])


def _admitted_agents(oracle, rng, **kw):
    pf, rtr = kw.get("p_task_failure", 0.0), kw.get("p_redo_truncate", 0.0)
    a = ScriptedAgent(
        name="A", oracle=oracle, rng=rng, p_task_failure=pf,
        p_redo_truncate=rtr,
        plan=[
            PlanStep(kind=CallKind.READ, obj=X, reads=(X,), premise="x",
                     duration=3.0, label="read x"),
            PlanStep(kind=CallKind.THINK, duration=12.0, label="reason"),
            # undeclared, but nobody else touches it
            PlanStep(kind=CallKind.WRITE, obj=NOTES, tool=set_v,
                     args=lambda view: {"v": "note-from-A"}, writes=(),
                     duration=4.0, label="replan: write notes",
                     depends_on=("x",)),
        ])
    return [a, _peer(oracle, rng, pf, rtr)]


def _coarsened_agents(oracle, rng, b_delay: float = 20.0, **kw):
    pf, rtr = kw.get("p_task_failure", 0.0), kw.get("p_redo_truncate", 0.0)
    a = ScriptedAgent(
        name="A", oracle=oracle, rng=rng, p_task_failure=pf,
        p_redo_truncate=rtr,
        plan=[
            PlanStep(kind=CallKind.READ, obj=X, reads=(X,), premise="x",
                     duration=3.0, label="read x"),
            PlanStep(kind=CallKind.THINK, duration=6.0, label="reason"),
            # undeclared: the replan
            PlanStep(kind=CallKind.WRITE, obj=Y, tool=set_v,
                     args=lambda view: {"v": "y-from-A"}, writes=(),
                     duration=4.0, label="replan: write y",
                     depends_on=("x",)),
            PlanStep(kind=CallKind.THINK, duration=6.0, label="reason"),
            PlanStep(kind=CallKind.WRITE, obj=Z, tool=set_v,
                     args={"v": "z-from-A"}, writes=(Z,), duration=4.0,
                     label="write z"),
        ])
    b = ScriptedAgent(
        name="B", oracle=oracle, rng=rng, p_task_failure=pf,
        p_redo_truncate=rtr,
        plan=[
            # B has committed nothing when A replans, so its future can
            # still be regrouped
            PlanStep(kind=CallKind.THINK, duration=b_delay, label="reason"),
            PlanStep(kind=CallKind.WRITE, obj=Y, tool=set_v,
                     args={"v": "y-from-B"}, writes=(Y,), duration=4.0,
                     label="write y"),
            PlanStep(kind=CallKind.THINK, duration=6.0, label="reason"),
            PlanStep(kind=CallKind.WRITE, obj=Z, tool=set_v,
                     args={"v": "z-from-B"}, writes=(Z,), duration=4.0,
                     label="write z"),
        ])
    return [a, b]


_TOOLS = [
    {"name": "get_object", "kind": "read",
     "description": "read one object's fields"},
    {"name": "set_v", "kind": "write", "tool": set_v,
     "args": {"v": "the new v value"},
     "description": "set an object's v (reversible)"},
]
_INIT = {X: {"v": "init"}, Y: {"v": "init"}, Z: {"v": "init"},
         NOTES: {"v": ""}}

ADMITTED = Cell(
    name="admitted-replan", suite="anatomy", anomaly="benign escalation",
    initial=dict(_INIT), build_agents=_admitted_agents,
    llm_spec={"tasks": [
        # "exactly these two calls": without it the model stops after the
        # read in most trials, and a replan that is never attempted tells
        # us nothing about how one is judged
        f"Read '{X}'. Then write v='note-from-A' to '{NOTES}'. Do exactly "
        f"these two calls in this order. If an access is refused, say so "
        f"and stop.",
        f"Write v='from-B' to '{X}'. Then read '{Y}'.",
    ], "tools": _TOOLS})

COARSENED = Cell(
    name="coarsened-replan", suite="anatomy",
    anomaly="escalation repairable by merging the future",
    initial=dict(_INIT), build_agents=_coarsened_agents,
    llm_spec={"tasks": [
        f"Read '{X}'. Then write v='y-from-A' to '{Y}'. Then write "
        f"v='z-from-A' to '{Z}'. Do exactly these three calls in this "
        f"order. If an access is refused, say so and stop.",
        f"Write v='y-from-B' to '{Y}'. Then write v='z-from-B' to '{Z}'. "
        f"Do exactly these two calls in this order.",
    ], "tools": _TOOLS})

SCENARIOS = [
    dict(key="clean-cut", cell=CELLS["cc-claim"], expect="certified as declared",
         title="A clean cut",
         story="A reads a shared pool, reasons for a long time, then writes "
               "a private claim. B updates the pool and writes its own "
               "claim. The only conflict joins the two first pieces."),
    dict(key="forced-merge", cell=CELLS["k8s-replica-scale"],
         expect="merged at launch",
         title="A cut that must not be made",
         story="Both agents read the replica count, reason, and write a new "
               "count derived from it. Cutting at the reasoning gap would "
               "let both read 4 and one update vanish."),
    dict(key="admitted-replan", cell=ADMITTED, expect="admitted",
         title="A replan that is harmless",
         story="A reads x and commits. Only then does it decide to write a "
               "notes object it never declared. Nobody else touches that "
               "object."),
    dict(key="coarsened-replan", cell=COARSENED, expect="coarsened",
         title="A replan that needs the future regrouped",
         story="A reads x and commits, then decides to write y, which B "
               "also plans to write. As chopped, the new plan has a cycle. "
               "B has not started, so both futures can still be merged."),
    dict(key="refused-replan", cell=CELLS["cc-escalate"], expect="refused",
         title="A replan that cannot be made safe",
         story="A reads x and commits. B writes x and reads y in one piece "
               "and commits. Only then does A decide to write y. The cycle "
               "runs through A's committed read."),
]


def benchmark_scenarios() -> List[Dict[str, Any]]:
    """The cells of the protocol comparison, as they are.

    Nothing is designed to trigger a decision here, so there is no expected
    outcome: the experiment only observes what certification does with the
    workloads the matrix was scored on.
    """
    return [dict(key=name, cell=cell, expect=None, title=name,
                 story=cell.anomaly, suite=cell.suite,
                 tasks=list(cell.llm_spec["tasks"]))
            for name, cell in CELLS.items() if getattr(cell, "llm_spec", None)]


# -- running ------------------------------------------------------------------
def _summarise(log: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Lock-hold intervals and the hold fraction, per workflow."""
    holds: Dict[str, List[List[float]]] = {}
    open_at: Dict[str, float] = {}
    first: Dict[str, float] = {}
    last: Dict[str, float] = {}
    for e in log:
        wf = e.get("wf")
        if wf is None:
            continue
        first.setdefault(wf, e["t"])
        last[wf] = e["t"]
        if e["ev"] == "lock" and wf not in open_at:
            open_at[wf] = e["t"]
        elif e["ev"] == "unlock" and wf in open_at:
            start = open_at.pop(wf)
            iv = holds.setdefault(wf, [])
            if iv and abs(iv[-1][1] - start) < 1e-9:
                iv[-1][1] = e["t"]           # re-acquire at the same instant
            else:
                iv.append([start, e["t"]])
    out = {}
    for wf in first:
        span = max(last[wf] - first[wf], 1e-9)
        held = sum(b - a for a, b in holds.get(wf, []))
        out[wf] = {"start": first[wf], "end": last[wf],
                   "holds": holds.get(wf, []),
                   "hold_fraction": round(held / span, 3)}
    return out


def _decisions(log) -> Dict[str, Any]:
    launch = next((e for e in log if e["ev"] == "certified"), None)
    declared = next((e for e in log if e["ev"] == "declared"), None)
    revs = [e for e in log if e["ev"] == "revision"]
    return {
        "launch": ("certified as declared" if declared["plan"]["certifies"]
                   else "merged at launch"),
        "merges": launch["merges"] if launch else 0,
        "revisions": [r["decision"] for r in revs],
    }


def run_scripted(cell: Cell) -> Dict[str, Any]:
    rng = random.Random(0)
    cfg = TrialConfig(p_task_failure=0.0, p_misjudge=0.0)
    agents = _build(cell, [0, 1], cfg, rng)
    proto = AuditedChopping()
    target = cell.fresh_target()
    rt = Runtime(target, agents, proto, rng)
    rt.jitter_sigma = 0.0
    rt.time_scale = 1.0
    rt.run()
    return {"log": proto.log, "workflows": _summarise(proto.log),
            "decisions": _decisions(proto.log),
            "final": normalize(target.live), "stats": proto.stats()}


def run_llm(cell: Cell, seed: int, model: str) -> Dict[str, Any]:
    from coagent.agents.llm import LLMAgent
    from run_llm_matrix import _JudgeStub, to_toolspecs

    tools = to_toolspecs(cell.llm_spec)
    probe = cell.build_agents(_JudgeStub(), random.Random(0),
                              p_task_failure=0.0)
    declared = [a.declare_pieces() for a in probe]
    agents = [
        LLMAgent(f"agent-{i}", task, tools, provider="anthropic", model=model,
                 pieces=declared[i])
        for i, task in enumerate(cell.llm_spec["tasks"])
    ]
    proto = AuditedChopping()
    target = cell.fresh_target()
    rt = Runtime(target, agents, proto, random.Random(seed))
    rt.jitter_sigma = 0.0
    rt.time_scale = 1.0
    rt.run()
    calls = [f"{e['wf']}:{e['op']}({e['obj']})"
             + (" REFUSED" if e["refused"] else "")
             for e in proto.log if e["ev"] == "call"]
    revs = [{"t": e["t"], "wf": e["wf"], "access": e["access"],
             "frontier": e["frontier"], "decision": e["decision"],
             "through_committed": e["through_committed"]}
            for e in proto.log if e["ev"] == "revision"]
    return {"seed": seed, "decisions": _decisions(proto.log), "calls": calls,
            "revisions": revs, "log": proto.log,
            # the oracle judges the model-driven decisions too
            "oracle": oracle_report(proto.log),
            "workflows": _summarise(proto.log),
            "final": normalize(target.live),
            "usd": sum(__import__("coagent.agents.backends", fromlist=["price"])
                       .price(model, a.prompt_tokens, a.completion_tokens)
                       for a in agents)}


def window_sweep() -> List[Dict[str, Any]]:
    """When does the same replan stop being repairable?

    The coarsened-replan scenario, with one knob: how long B waits before
    its first write. A always replans at the same moment. The only thing
    that changes is whether B's first piece has committed by then -- and
    Theorem 2 says that is exactly what decides between merging the future
    and refusing.
    """
    out = []
    for delay in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20]:
        cell = Cell(
            name=f"window-{delay}", suite="anatomy", anomaly="",
            initial=dict(_INIT),
            build_agents=lambda o, r, d=delay, **kw: _coarsened_agents(
                o, r, b_delay=float(d), **kw))
        res = run_scripted(cell)
        rev = next(e for e in res["log"] if e["ev"] == "revision")
        b_first_commit = next((e["t"] for e in res["log"]
                               if e["ev"] == "commit" and e["wf"] == "B"),
                              None)
        out.append({
            "b_delay": delay, "replan_at": rev["t"],
            "b_first_commit": b_first_commit,
            "b_committed_at_replan": rev["frontier"]["B"],
            "decision": rev["decision"],
            "through_committed": rev["through_committed"],
            "oracle_agrees": all(o["certifier_ok"] == o["oracle_safe"]
                                 for o in oracle_report(res["log"])),
            "final": res["final"],
        })
    return out


def oracle_report(log) -> List[Dict[str, Any]]:
    """Each verdict next to the brute-force truth about the plan it judged."""
    out = []
    d = next(e for e in log if e["ev"] == "declared")["plan"]
    c = next(e for e in log if e["ev"] == "certified")["plan"]
    out.append({"what": "declared chop", "certifier_ok": d["certifies"],
                "oracle_safe": d["oracle_safe"]})
    out.append({"what": "plan run at launch", "certifier_ok": c["certifies"],
                "oracle_safe": c["oracle_safe"]})
    for r in (e for e in log if e["ev"] == "revision"):
        out.append({"what": "revision as proposed",
                    "certifier_ok": r["proposed"]["certifies"],
                    "oracle_safe": r["proposed"]["oracle_safe"]})
        out.append({"what": "revision, future sealed",
                    "certifier_ok": r["sealed"]["certifies"],
                    "oracle_safe": r["sealed"]["oracle_safe"]})
        if r["decision"] != "refused":
            out.append({"what": "plan installed after revision",
                        "certifier_ok": r["after"]["certifies"],
                        "oracle_safe": r["after"]["oracle_safe"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", type=int, default=0,
                    help="model-driven trials per scenario (0 = scripted only)")
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--suite", choices=["designed", "benchmark"],
                    default="designed",
                    help="designed: five scenarios, one per decision path; "
                         "benchmark: the twelve cells of the protocol "
                         "comparison, unmodified")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        stem = ("anatomy" if args.suite == "designed"
                else f"benchmark_{args.model}")
        args.out = str(ROOT / "results" / "anatomy" / f"{stem}.json")
    scenarios = SCENARIOS if args.suite == "designed" else benchmark_scenarios()

    results = []
    for sc in scenarios:
        cell = sc["cell"]
        scripted = run_scripted(cell)
        entry = {k: sc[k] for k in ("key", "title", "story", "expect",
                                    "suite", "tasks") if k in sc}
        entry["cell"] = cell.name
        entry["scripted"] = scripted
        entry["oracle"] = oracle_report(scripted["log"])
        got = scripted["decisions"]
        observed = got["revisions"][0] if got["revisions"] else got["launch"]
        entry["observed"] = observed
        agree = all(o["certifier_ok"] == o["oracle_safe"]
                    for o in entry["oracle"])
        print(f"{sc['key']:18s} expected {str(sc['expect']):22s} observed "
              f"{observed:22s} oracle agrees: {agree}   hold "
              + "  ".join(f"{w}={v['hold_fraction']:.0%}"
                          for w, v in scripted["workflows"].items()))
        entry["llm"] = []
        for seed in range(args.llm):
            try:
                r = run_llm(cell, seed, args.model)
            except Exception as exc:                      # noqa: BLE001
                print(f"   llm seed {seed} failed: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                continue
            entry["llm"].append(r)
            d = r["decisions"]
            agree = all(o["certifier_ok"] == o["oracle_safe"]
                        for o in r["oracle"])
            print(f"   llm seed {seed}: launch={d['launch']} "
                  f"revisions={d['revisions']} oracle agrees: {agree} "
                  f"${r['usd']:.4f}", flush=True)
        results.append(entry)

    window = window_sweep() if args.suite == "designed" else []
    print("\nreplan window (B's start delay -> decision on A's replan):")
    for w in window:
        print(f"   delay {w['b_delay']:>2}s  B pieces committed at replan: "
              f"{w['b_committed_at_replan']}  -> {w['decision']:9s} "
              f"oracle agrees: {w['oracle_agrees']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "scenarios": results,
                               "window": window},
                              indent=1, default=str))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
