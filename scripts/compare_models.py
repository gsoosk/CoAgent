#!/usr/bin/env python3
"""Compare two model runs of the same matrix side by side.

The protocols and workloads are identical across runs, so any difference is
the agent. That makes the pair a direct measurement of how much each
protocol's guarantee depends on the model driving it -- which is the whole
question behind the paper's assumption A3.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coagent.protocols import PROTOCOL_ORDER        # noqa: E402
from coagent.runner import reference_states         # noqa: E402
from coagent.workloads import CELLS                 # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, c - h), min(1.0, c + h)


def load(path, refs):
    runs = [json.loads(l) for l in open(path)]

    def free(r):
        cell = CELLS[r["cell"]]
        return bool(cell.anomaly_free(r["final_state"], refs[r["cell"]]))

    by = defaultdict(list)
    for r in runs:
        by[(r["cell"], r["protocol"])].append(r)
    have = [c for c in CELLS
            if all(len(by[(c, p)]) >= 5 for p in PROTOCOL_ORDER)]
    clean = [c for c in have if all(free(r) for r in by[(c, "serial")])]
    return runs, by, have, clean, free


def summarise(by, scope, free):
    out = {}
    for p in PROTOCOL_ORDER:
        rs = [r for c in scope for r in by[(c, p)]]
        if not rs:
            continue
        k, n = sum(1 for r in rs if free(r)), len(rs)
        _, lo, hi = wilson(k, n)
        out[p] = dict(
            safe=100 * k / n, lo=lo * 100, hi=hi * 100, n=n,
            ok=100 * sum(r["invariant_ok"] for r in rs) / n,
            speed=statistics.mean(
                statistics.median(x["wall"] for x in by[(c, "serial")])
                / statistics.median(x["wall"] for x in by[(c, p)])
                for c in scope),
            usd=statistics.mean(r["usd"] for r in rs),
            rounds=statistics.mean(r["rounds"] for r in rs),
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="haiku")
    ap.add_argument("--b", default="sonnet")
    args = ap.parse_args()

    refs = {c: reference_states(CELLS[c]) for c in CELLS}
    runs_a, by_a, have_a, clean_a, free = load(
        ROOT / "results" / args.a / "llm_runs.jsonl", refs)
    runs_b, by_b, have_b, clean_b, _ = load(
        ROOT / "results" / args.b / "llm_runs.jsonl", refs)

    # compare on the cells trustworthy in BOTH runs, or the comparison is
    # between different cell sets rather than between models
    scope = [c for c in clean_a if c in clean_b]
    dropped_a = [c for c in have_a if c not in clean_a]
    dropped_b = [c for c in have_b if c not in clean_b]

    ma = runs_a[-1].get("model", args.a)
    mb = runs_b[-1].get("model", args.b)
    sa = summarise(by_a, scope, free)
    sb = summarise(by_b, scope, free)

    print(f"{args.a}: {ma}, {len(runs_a)} trials, "
          f"${sum(r['usd'] for r in runs_a):.2f}")
    print(f"{args.b}: {mb}, {len(runs_b)} trials, "
          f"${sum(r['usd'] for r in runs_b):.2f}")
    print(f"compared on {len(scope)} cells trustworthy in both runs")
    if dropped_a or dropped_b:
        print(f"  predicate unreliable — {args.a}: {dropped_a or 'none'}; "
              f"{args.b}: {dropped_b or 'none'}")
    print()
    head = (f"{'protocol':10}{'anomaly-free':>26}{'overall ok':>16}"
            f"{'speedup':>16}{'$/trial':>16}")
    print(head)
    print(f"{'':10}{args.a + ' / ' + args.b:>26}"
          f"{args.a + ' / ' + args.b:>16}{args.a + ' / ' + args.b:>16}"
          f"{args.a + ' / ' + args.b:>16}")
    print("-" * len(head))
    for p in PROTOCOL_ORDER:
        if p not in sa or p not in sb:
            continue
        a, b = sa[p], sb[p]
        safe = f"{a['safe']:.0f}% / {b['safe']:.0f}%"
        ok = f"{a['ok']:.0f}% / {b['ok']:.0f}%"
        sp = f"{a['speed']:.2f} / {b['speed']:.2f}"
        usd = f"{a['usd']:.4f} / {b['usd']:.4f}"
        print(f"{p:10}{safe:>26}{ok:>16}{sp:>16}{usd:>16}")

    # A1/A3 on two scopes. The shared scope is the only honest basis for
    # comparing the models, but it is the intersection, so a cell excluded
    # in either run disappears -- including cells where the other model
    # produced the run's most interesting result. Reporting the shared
    # scope alone would quote an A3 that each model's own data contradicts.
    def assumptions(by_, scope_):
        ser = [r for c in scope_ for r in by_[(c, "serial")]]
        mt = [r for c in scope_ for r in by_[(c, "MTPO")] if r["notifications"]]
        healed = sum(1 for r in mt if free(r))
        a1 = 100 * sum(r["invariant_ok"] for r in ser) / max(len(ser), 1)
        out = f"A1 {a1:3.0f}%"
        if mt:
            _, lo, hi = wilson(healed, len(mt))
            out += (f"   A3 {healed}/{len(mt)} = "
                    f"{100*healed/len(mt):.0f}% [{lo*100:.0f}, {hi*100:.0f}]")
        return out

    print()
    print(f"shared scope ({len(scope)} cells) -- the model comparison")
    for tag, by_ in ((args.a, by_a), (args.b, by_b)):
        print(f"  {tag:8} {assumptions(by_, scope)}")
    print()
    print("own scope -- each model on every cell its own run can score")
    for tag, by_, cl in ((args.a, by_a, clean_a), (args.b, by_b, clean_b)):
        print(f"  {tag:8} {assumptions(by_, cl)}   ({len(cl)} cells)")
    print()
    print("  paper    A1  98%   A3 assumed 95%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
