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
from coagent.protocols import PROTOCOL_ORDER, PROTOCOLS  # noqa: E402
from coagent.runner import reference_states            # noqa: E402
from coagent.runtime import CallKind, Runtime          # noqa: E402
from coagent.workloads import CELLS                    # noqa: E402
from coagent.workloads.base import invariant_ok, normalize  # noqa: E402

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
    ap.add_argument("--force", action="store_true",
                    help="re-run trials already present in the output file")
    ap.add_argument("--out", default=str(ROOT / "results" / "llm_runs.jsonl"))
    args = ap.parse_args()

    model = args.model or DEFAULTS[args.provider]
    key_var = ("ANTHROPIC_API_KEY" if args.provider == "anthropic"
               else "OPENROUTER_API_KEY")
    cells = list(CELLS) if args.cells == ["all"] else args.cells
    protos = (list(PROTOCOL_ORDER) if args.protocols == ["all"]
              else args.protocols)
    unknown = [c for c in cells if c not in CELLS]
    if unknown:
        print(f"unknown cells: {unknown}", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    seen = set() if args.force else done_keys(out_path)
    plan = [(c, p, s) for c in cells for p in protos
            for s in range(args.trials) if (c, p, s) not in seen]

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
            agents = [
                LLMAgent(f"agent-{i}", task, tools,
                         provider=args.provider, model=model,
                         max_rounds=args.max_rounds,
                         effort=args.effort, thinking=not args.no_thinking)
                for i, task in enumerate(cell.llm_spec["tasks"])
            ]
            rt = Runtime(target, agents, PROTOCOLS[proto](), random.Random(seed))
            rt.jitter_sigma = 0.0
            rt.time_scale = 1.0
            try:
                rt.run()
            except Exception as exc:                      # noqa: BLE001
                print(f"{cell_name}/{proto}/{seed}: run failed: {exc}")
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
                "cell": cell_name, "protocol": proto, "seed": seed,
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
            results.append(rec)
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            print(f"{cell_name:24} {proto:10} seed={seed} "
                  f"{'PASS' if ok else 'FAIL'} "
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
