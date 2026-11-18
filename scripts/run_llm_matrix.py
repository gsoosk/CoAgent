#!/usr/bin/env python3
"""Run the contended cells with real LLM Workers instead of scripted plans.

This is the experiment the simulated matrix cannot do. It replaces the two
knobs standing in for the paper's agent assumptions with measurements, and in
particular it measures the A3 (self-healing) rate: how often a notified model
correctly recognises that a conflicting write invalidated a premise of its
plan. The paper reports a 5% miss rate on deepseek-v4-flash, and that 5% is
the whole of MTPO's correctness gap below serial -- so it is the single
number in the paper most worth checking independently.

    cp .env.example .env && $EDITOR .env      # put the key here
    python3 scripts/run_llm_matrix.py --dry-run
    python3 scripts/run_llm_matrix.py --cells k8s-canary --trials 5

Runs append to results/llm_runs.jsonl and are resumable: a (cell, protocol,
seed) already in the file is skipped unless --force is passed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coagent.agents.backends import (                  # noqa: E402
    DEFAULTS, load_key, price,
)
from coagent.agents.llm import LLMAgent, ToolSpec       # noqa: E402
from coagent.agents.planning import (                  # noqa: E402
    PlanningLLMAgent, negotiate_order,
)
from coagent.protocols import PROTOCOL_ORDER, PROTOCOLS  # noqa: E402
from coagent.runner import reference_states            # noqa: E402
from coagent.runtime import CallKind, Runtime          # noqa: E402
from coagent.workloads import CELLS as _BENCH_CELLS     # noqa: E402
from coagent.workloads.reorder_cells import REORDER_CELLS  # noqa: E402
from coagent.workloads.realm_urs import REALM_CELLS     # noqa: E402
from coagent.workloads.realm_jssp import JSSP_CELLS     # noqa: E402
from coagent.workloads.realm_wedding import WEDDING_CELLS  # noqa: E402
from coagent.workloads.realm_tours import TOURS_CELLS     # noqa: E402
from coagent.workloads.realm_thanksgiving import THANKS_CELLS  # noqa: E402
from coagent.workloads.base import invariant_ok, normalize  # noqa: E402


# The reorder and REALM cells are addressable by name but are not part of
# "all", so the benchmark matrix stays the twelve cells it was scored on.
CELLS = {**_BENCH_CELLS, **REORDER_CELLS, **REALM_CELLS, **JSSP_CELLS,
         **WEDDING_CELLS, **TOURS_CELLS, **THANKS_CELLS}


class _JudgeStub:
    """The probe only needs declare_pieces(), never a judgement."""

    def is_relevant(self, truly):
        return truly

# rough per-trial usage, from the simulated token model; only used for --dry-run
EST_PROMPT = 40_000
EST_COMPLETION = 2_500


def to_toolspecs(spec: dict) -> list:
    """Turn a cell's declarative tool list into what the Worker sees.

    A range tool takes no arguments at all: its subtree is fixed by the cell,
    so the framework binds it and the model cannot widen the footprint.
    """
    out = []
    for t in spec["tools"]:
        is_read = t["kind"] == "read"
        ranged = "prefix" in t
        args = {} if ranged else {"obj": "object id"}
        args.update(t.get("args", {}))
        out.append(ToolSpec(
            name=t["name"],
            description=(
                f"{t['description']} (takes no arguments)" if ranged
                else t["description"]
            ),
            tool=t.get("tool"),
            kind=CallKind.READ if is_read else CallKind.WRITE,
            args_schema=args,
            obj_param="obj",
            prefix_param=None,
            fixed_prefix=t["prefix"] if ranged else None,
        ))
    return out


def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
            out.add((r["cell"], r["protocol"], r["seed"]))
        except (json.JSONDecodeError, KeyError):
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=["k8s-canary"],
                    help="cell names, or 'all'")
    ap.add_argument("--protocols", nargs="*", default=["MTPO"],
                    help="protocol names, or 'all'")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--first-seed", type=int, default=0,
                    help="seeds run are first-seed .. first-seed+trials-1, so "
                         "one cell's seeds can be split across processes")
    ap.add_argument("--provider", default="openrouter",
                    choices=["openrouter", "anthropic"],
                    help="openrouter/deepseek reproduces the paper; "
                         "anthropic measures a different model")
    ap.add_argument("--model", default=None,
                    help="defaults to the provider's default model")
    ap.add_argument("--effort", default="medium",
                    choices=["low", "medium", "high", "xhigh", "max"],
                    help="Anthropic only: thinking depth vs spend")
    ap.add_argument("--no-thinking", action="store_true",
                    help="Anthropic only: disable adaptive thinking")
    ap.add_argument("--max-rounds", type=int, default=14)
    ap.add_argument("--dry-run", action="store_true",
                    help="estimate cost and exit without calling the API")
    ap.add_argument("--free-planning", action="store_true",
                    help="do not charge the order-negotiation call to the "
                         "clock: what a plan made ahead of time, or reused "
                         "across runs, would cost. Recorded under the "
                         "protocol name with a '-free' suffix")
    ap.add_argument("--force", action="store_true",
                    help="re-run trials already present in the output file")
    ap.add_argument("--retry-unparsed", action="store_true",
                    help="on a reply that is not JSON, remind the model of "
                         "the format instead of ending the agent")
    ap.add_argument("--interleave", action="store_true",
                    help="run every protocol for one seed before the next "
                         "seed, so model latency drifting over the session "
                         "lands on all protocols alike")
    ap.add_argument("--out", default=str(ROOT / "results" / "llm_runs.jsonl"))
    args = ap.parse_args()

    model = args.model or DEFAULTS[args.provider]
    key_var = ("ANTHROPIC_API_KEY" if args.provider == "anthropic"
               else "OPENROUTER_API_KEY")
    cells = list(_BENCH_CELLS) if args.cells == ["all"] else args.cells
    protos = (list(PROTOCOL_ORDER) if args.protocols == ["all"]
              else args.protocols)
    unknown = [c for c in cells if c not in CELLS]
    if unknown:
        print(f"unknown cells: {unknown}", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    seen = set() if args.force else done_keys(out_path)
    seeds = range(args.first_seed, args.first_seed + args.trials)
    if args.interleave:
        plan = [(c, p, s) for c in cells for s in seeds
                for p in protos if (c, p, s) not in seen]
    else:
        plan = [(c, p, s) for c in cells for p in protos
                for s in seeds if (c, p, s) not in seen]

    est = price(model, EST_PROMPT, EST_COMPLETION) * len(plan)
    print(f"provider   {args.provider}")
    print(f"model      {model}")
    print(f"cells      {len(cells)}  protocols {len(protos)}  "
          f"trials {args.trials}")
    print(f"to run     {len(plan)} trials "
          f"({len(seen)} already done)" if seen else f"to run     {len(plan)} trials")
    print(f"estimated  ${est:.2f}  (~${est / max(len(plan), 1):.4f}/trial)")
    if args.provider == "anthropic" and not args.no_thinking:
        print("           thinking tokens bill as output and are NOT in that "
              "estimate")
    print()
    if args.dry_run:
        print("dry run: nothing was called and nothing was spent.")
        return 0
    if not load_key(key_var):
        print(f"{key_var} not found in the environment or .env.",
              file=sys.stderr)
        print("Copy .env.example to .env and put the key there. "
              "Nothing was run.", file=sys.stderr)
        return 2
    if not plan:
        print("nothing to do.")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with open(out_path, "a") as fh:
        for cell_name, proto, seed in plan:
            cell = CELLS[cell_name]
            if not cell.llm_spec:
                print(f"skip {cell_name}: no llm_spec")
                continue
            refs = reference_states(cell)
            tools = to_toolspecs(cell.llm_spec)
            target = cell.fresh_target()
            # Certified chopping needs a declared plan. Footprints come from
            # the workload (the paper supplies them the same way), so take
            # them from the cell's own scripted agents.
            import random as _r
            _probe = cell.build_agents(
                _JudgeStub(), _r.Random(0), p_task_failure=0.0)
            declared = [getattr(a, "declare_pieces", lambda: [])()
                        for a in _probe]
            protocol = PROTOCOLS[proto]()
            negotiates = getattr(protocol, "negotiates_order", False)
            agent_cls = PlanningLLMAgent if negotiates else LLMAgent
            agents = [
                agent_cls(f"agent-{i}", task, tools,
                         provider=args.provider, model=model,
                         max_rounds=args.max_rounds,
                         pieces=declared[i] if i < len(declared) else [],
                         retry_unparsed=args.retry_unparsed,
                         effort=args.effort, thinking=not args.no_thinking)
                for i, task in enumerate(cell.llm_spec["tasks"])
            ]
            rt = Runtime(target, agents, protocol, random.Random(seed))
            rt.jitter_sigma = 0.0
            rt.time_scale = 1.0
            negotiation = None
            try:
                if negotiates:
                    negotiation = negotiate_order(agents)
                    if args.free_planning:
                        for ag in agents:
                            ag.plan_latency = 0.0
                rt.run()
            except Exception as exc:                      # noqa: BLE001
                # Name the type: a dropped trial is invisible in the output
                # file, so this line is the only record of why it went
                # missing, and 429-vs-400-vs-bug changes what to do about it.
                print(f"{cell_name}/{proto}/{seed}: run failed: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                continue

            ok = invariant_ok(target.live, refs, cell)
            notes = sum(s.notes_seen for s in rt.states.values())
            # a notification was acted on if the agent wrote after seeing it
            repaired = sum(1 for s in rt.states.values()
                           if s.notes_seen and len(s.writes) > 1)
            pt = sum(a.prompt_tokens for a in agents)
            ct = sum(a.completion_tokens for a in agents)
            wall, _ = rt.totals()
            rec = {
                "cell": cell_name,
                "protocol": proto + ("-free" if negotiates
                                     and args.free_planning else ""),
                "seed": seed,
                "provider": args.provider, "model": model, "invariant_ok": ok, "wall": wall,
                "prompt_tokens": pt, "completion_tokens": ct,
                "usd": price(model, pt, ct),
                "notifications": notes, "repaired": repaired,
                "rounds": sum(a.rounds for a in agents),
                "final_state": normalize(target.live),
                "trace": rt.trace_lines,
                "transcript": [a.transcript for a in agents],
                "events": [e for a in agents for e in a.events],
            }
            if cell.anomaly_free is not None:
                rec["anomaly_free"] = bool(
                    cell.anomaly_free(rec["final_state"], refs))
            if cell.quality is not None:
                # the external benchmark's own plan-quality metrics
                rec["realm"] = cell.quality(rec["final_state"])
            if hasattr(protocol, "stats"):
                rec["proto_stats"] = protocol.stats()
            if getattr(protocol, "plan", None) is not None:
                # the plan as it stood at the end: what was actually cut
                rec["plan"] = {w: [p.label for p in ps]
                               for w, ps in protocol.plan.pieces.items()}
            if negotiation is not None:
                rec["negotiation"] = negotiation
            results.append(rec)
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            af = rec.get("anomaly_free")
            print(f"{cell_name:24} {proto:10} seed={seed} "
                  f"{'PASS' if ok else 'FAIL'} "
                  + ("" if af is None else f"AF={'y' if af else 'n'} ")
                  + f"wall={wall:.0f}s "
                  f"notif={notes} rounds={rec['rounds']} "
                  f"${rec['usd']:.4f}")

    if not results:
        return 0
    print("\n" + "=" * 68)
    by = defaultdict(list)
    for r in results:
        by[r["protocol"]].append(r)

    print(f"{'protocol':10}{'correct':>9}{'trials':>8}{'notif':>7}"
          f"{'acted on':>10}{'rounds/trial':>14}{'$/trial':>10}")
    print("-" * 68)
    scores = {}
    for proto in PROTOCOL_ORDER:
        rs = by.get(proto)
        if not rs:
            continue
        ok = 100 * sum(r["invariant_ok"] for r in rs) / len(rs)
        scores[proto] = ok
        notes = sum(r["notifications"] for r in rs)
        rep = sum(r["repaired"] for r in rs)
        rounds = sum(r["rounds"] for r in rs) / len(rs)
        usd = sum(r["usd"] for r in rs) / len(rs)
        print(f"{proto:10}{ok:8.0f}%{len(rs):8d}{notes:7d}{rep:10d}"
              f"{rounds:14.1f}{usd:10.4f}")

    # The paper attributes MTPO's whole correctness gap below serial to A3
    # (self-healing). Absolute correctness also carries A1 -- an agent that
    # botches its own task would botch it serially too -- so the gap, not the
    # level, is the number to compare.
    if "serial" in scores and "MTPO" in scores:
        gap = scores["serial"] - scores["MTPO"]
        print(f"\nserial - MTPO = {gap:.0f} points   (paper: 5)")
        print("  that gap is the A3 (self-healing) cost; the level also "
              "carries A1.")
    elif "MTPO" in scores:
        print("\nrun the serial baseline too -- without it the MTPO level "
              "cannot be split into A1 and A3.")
    print(f"\ntotal spend ${sum(r['usd'] for r in results):.3f}  "
          f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
