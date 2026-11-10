#!/usr/bin/env python3
"""Analyse a real-LLM run (results/llm_runs.jsonl).

Overall task correctness mixes two things the paper keeps separate: whether
each agent did its own job (A1), and whether the protocol prevented the
concurrency anomaly. With a weak model A1 dominates and the protocol signal
disappears. Every cell therefore carries an `anomaly_free` predicate that
isolates the second, and that is the column to read.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coagent.protocols import PROTOCOL_ORDER        # noqa: E402
from coagent.runner import reference_states         # noqa: E402
from coagent.workloads import CELLS                 # noqa: E402


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, c - h), min(1.0, c + h)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="haiku",
                    help="results subfolder for one model: "
                         "sets both the runs file and the "
                         "figure output directory")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--min-trials", type=int, default=5,
                    help="skip cells with fewer trials than this per protocol")
    args = ap.parse_args()
    base = ROOT / "results" / args.dir
    if args.runs is None:
        args.runs = str(base / "llm_runs.jsonl")
    if getattr(args, "outdir", None) is None:
        args.outdir = str(base / "figures")

    runs = [json.loads(l) for l in open(args.runs)]
    if not runs:
        print("no runs recorded")
        return 1
    model = runs[-1].get("model", "?")
    refs_cache = {c: reference_states(CELLS[c]) for c in {r["cell"] for r in runs}
                  if c in CELLS}

    def anomaly_free(r):
        cell = CELLS.get(r["cell"])
        if cell is None or cell.anomaly_free is None:
            return None
        return cell.anomaly_free(r["final_state"], refs_cache[r["cell"]])

    by = defaultdict(list)
    for r in runs:
        by[(r["cell"], r["protocol"])].append(r)
    cells = [c for c in CELLS if any((c, p) in by for p in PROTOCOL_ORDER)]

    # ---- per cell ---------------------------------------------------------
    head = f"{'cell':24}" + "".join(f"{p:>11}" for p in PROTOCOL_ORDER)
    print(f"Anomaly-free rate by cell — {model}")
    print(head)
    print("-" * len(head))
    for c in cells:
        row = f"{c:24}"
        for p in PROTOCOL_ORDER:
            rs = by.get((c, p), [])
            if len(rs) < args.min_trials:
                row += f"{'·':>11}"
                continue
            k = sum(1 for r in rs if anomaly_free(r))
            row += f"{f'{k}/{len(rs)}':>11}"
        print(row)

    # ---- which cells can the predicate be trusted on? --------------------
    # Serial has no concurrency, so anything it flags is the predicate
    # misfiring -- almost always an A1-mangled artefact it cannot tell from a
    # real anomaly. Those cells cannot separate protocols and are reported
    # separately rather than silently averaged in.
    clean, dirty = [], []
    for c in cells:
        rs = by.get((c, "serial"), [])
        if rs and all(anomaly_free(r) for r in rs):
            clean.append(c)
        else:
            dirty.append(c)
    if dirty:
        print()
        print("predicate NOT trustworthy on these cells (serial itself fails "
              "them, so an A1-mangled result is indistinguishable from an "
              "anomaly):")
        for c in dirty:
            rs = by.get((c, "serial"), [])
            k = sum(1 for r in rs if anomaly_free(r))
            print(f"    {c:24} serial {k}/{len(rs)}")

    # ---- aggregate --------------------------------------------------------
    print()
    head2 = (f"{'protocol':10}{'anomaly-free':>14}{'95% CI':>14}"
             f"{'overall ok':>12}{'notif':>7}{'rounds/t':>10}{'$/trial':>9}")
    print(f"aggregate over the {len(clean)} cells where the predicate holds:")
    print(head2)
    print("-" * len(head2))
    summary = {}
    scope = clean if clean else cells
    for p in PROTOCOL_ORDER:
        rs = [r for c in scope for r in by.get((c, p), [])
              if len(by.get((c, p), [])) >= args.min_trials]
        if not rs:
            continue
        n = len(rs)
        k = sum(1 for r in rs if anomaly_free(r))
        _, lo, hi = wilson(k, n)
        ok = sum(r["invariant_ok"] for r in rs)
        notif = sum(r["notifications"] for r in rs)
        rounds = sum(r["rounds"] for r in rs) / n
        usd = sum(r["usd"] for r in rs) / n
        summary[p] = 100 * k / n
        print(f"{p:10}{f'{k}/{n} = {100*k/n:.0f}%':>14}"
              f"{f'[{lo*100:.0f}, {hi*100:.0f}]':>14}"
              f"{f'{ok}/{n}':>12}{notif:>7}{rounds:>10.1f}{usd:>9.4f}")

    # ---- the two agent assumptions ---------------------------------------
    print()
    ser = [r for c in scope for r in by.get((c, "serial"), [])]
    if ser:
        a1 = 100 * sum(r["invariant_ok"] for r in ser) / len(ser)
        print(f"A1 individual success (serial, overall): {a1:.0f}%   "
              f"(paper: 98%)")
    mt = [r for c in scope for r in by.get((c, "MTPO"), [])]
    notified = [r for r in mt if r["notifications"]]
    if notified:
        healed = sum(1 for r in notified if anomaly_free(r))
        _, lo, hi = wilson(healed, len(notified))
        print(f"A3 self-healing (MTPO trials with a notification): "
              f"{healed}/{len(notified)} = {100*healed/len(notified):.0f}% "
              f"[{lo*100:.0f}, {hi*100:.0f}]   (paper assumes 95%)")

    print(f"\n{len(runs)} trials, ${sum(r['usd'] for r in runs):.2f} total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
