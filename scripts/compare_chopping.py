#!/usr/bin/env python3
"""Compare the chopping variants against the baselines on one LLM run.

The matrix analysis reports one row per protocol. The question these
variants exist to answer is narrower and per-cell: priority-ordered locking
is supposed to keep chopping's deadlock-freedom while taking 2PL's shorter
holds, so what matters is which cells move and in which direction.

Trial records carry no deadlock or abort counters -- those live on the
Runtime, not the JSON -- but the recorded trace does, so the mechanism
counts here are read back out of the trace lines.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coagent.runner import reference_states        # noqa: E402
from coagent.workloads import CELLS                # noqa: E402

BASE = ["serial", "2PL-saga", "MTPO", "chopping"]
VARIANTS = ["chopping-2pl", "chopping-prio"]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, c - h), min(1.0, c + h)


def mech(runs):
    """Deadlocks, wounds and aborts, recovered from the trace lines."""
    dl = wo = ab = 0
    for r in runs:
        for line in r.get("trace") or []:
            if "DEADLOCK detected" in line:
                dl += 1
            elif "WOUND " in line:
                wo += 1
            if "ABORT (" in line:
                ab += 1
    n = max(len(runs), 1)
    return dl / n, wo / n, ab / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="haiku")
    args = ap.parse_args()
    path = ROOT / "results" / args.dir / "llm_runs.jsonl"
    runs = [json.loads(l) for l in open(path) if l.strip()]

    by = collections.defaultdict(list)
    for r in runs:
        by[(r["cell"], r["protocol"])].append(r)
    present = [p for p in BASE + VARIANTS if any(k[1] == p for k in by)]
    shown = [p for p in present if p != "serial"]

    refs = {c: reference_states(CELLS[c]) for c in CELLS}

    def free(r):
        cell = CELLS[r["cell"]]
        return bool(cell.anomaly_free(r["final_state"], refs[r["cell"]]))

    cells = [c for c in CELLS
             if all(len(by[(c, p)]) >= 5 for p in present)]
    # Serial has no concurrency, so a cell where serial itself trips the
    # predicate cannot distinguish an agent error from a protocol one.
    clean = [c for c in cells if all(free(r) for r in by[(c, "serial")])]
    dropped = [c for c in cells if c not in clean]

    model = runs[-1].get("model", "?")
    print(f"{model}: {len(runs)} trials, "
          f"{len(clean)} cells with a trustworthy predicate"
          + (f" (dropped {', '.join(dropped)})" if dropped else ""))
    print()
    if not clean:
        # A run in flight has a protocol below the threshold on every cell.
        print(f"no cell yet has >=5 trials for all of: {', '.join(present)}")
        return 1

    # ---- per-cell speedup -------------------------------------------------
    hdr = f"{'cell':24}" + "".join(f"{p:>14}" for p in shown)
    print("Speedup vs serial, per cell (median wall)")
    print(hdr)
    print("-" * len(hdr))
    agg = {p: [] for p in shown}
    for c in clean:
        s = statistics.median(x["wall"] for x in by[(c, "serial")])
        row = f"{c:24}"
        best = max(s / statistics.median(x["wall"] for x in by[(c, p)])
                   for p in shown)
        for p in shown:
            v = s / statistics.median(x["wall"] for x in by[(c, p)])
            agg[p].append(v)
            mark = "*" if abs(v - best) < 1e-9 else " "
            row += f"{v:>13.2f}{mark}"
        print(row)
    print("-" * len(hdr))
    print(f"{'mean':24}" + "".join(
        f"{statistics.mean(agg[p]):>13.2f} " for p in shown))
    print(f"{'worst cell':24}" + "".join(
        f"{min(agg[p]):>13.2f} " for p in shown))
    print()

    # ---- correctness and mechanism ---------------------------------------
    hdr2 = (f"{'protocol':16}{'anomaly-free':>18}{'95% CI':>14}"
            f"{'dl/t':>8}{'wound/t':>9}{'abort/t':>9}{'notif/t':>9}"
            f"{'$/trial':>10}")
    print("Correctness and mechanism, over the trustworthy cells")
    print(hdr2)
    print("-" * len(hdr2))
    for p in present:
        rs = [r for c in clean for r in by[(c, p)]]
        k, n = sum(1 for r in rs if free(r)), len(rs)
        _, lo, hi = wilson(k, n)
        dl, wo, ab = mech(rs)
        notes = sum(r["notifications"] for r in rs) / max(n, 1)
        usd = statistics.mean(r["usd"] for r in rs)
        print(f"{p:16}{f'{k}/{n} = {100*k/n:.0f}%':>18}"
              f"{f'[{lo*100:.0f}, {hi*100:.0f}]':>14}"
              f"{dl:>8.2f}{wo:>9.2f}{ab:>9.2f}{notes:>9.2f}{usd:>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
