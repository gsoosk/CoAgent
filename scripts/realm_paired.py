#!/usr/bin/env python3
"""Paired comparisons on the REALM runs, over (instance, seed) pairs.

Speed and cost are ratios of wall-clock and spend between two protocols on
the same instance and seed, so per-instance difficulty and the session's
latency cancel; consistency is the difference in anomaly-free rate. 95%
bootstrap intervals over the pairs.

    python scripts/realm_paired.py results/realm/parts
"""

from __future__ import annotations

import json
import math
import random
import statistics as st
import sys
from pathlib import Path

PAIRS = [("chopping-prio", "2PL-saga"), ("chopping-prio", "MTPO"),
         ("chopping-prio", "serial"), ("MTPO", "2PL-saga"),
         ("MTPO", "serial"), ("2PL-saga", "serial")]


def boot(vals, rng, b=20000):
    ms = sorted(st.mean(rng.choice(vals) for _ in vals) for _ in range(b))
    return st.mean(vals), ms[int(.025 * b)], ms[int(.975 * b)]


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from score_realm import load
    rows = load(sorted(Path(sys.argv[1]).glob("*.jsonl")))
    by = {(r["cell"], r["seed"], r["protocol"]): r for r in rows}
    keys = sorted({(r["cell"], r["seed"]) for r in rows})
    rng = random.Random(0)
    print(f"paired over {len(keys)} (instance, seed) pairs, 95% bootstrap CI")
    for a, b in PAIRS:
        ks = [k for k in keys if (*k, a) in by and (*k, b) in by]
        sp = [math.log(by[(*k, b)]["wall"] / by[(*k, a)]["wall"]) for k in ks]
        co = [math.log(by[(*k, a)]["usd"] / by[(*k, b)]["usd"]) for k in ks]
        af = [by[(*k, a)]["anomaly_free"] - by[(*k, b)]["anomaly_free"]
              for k in ks]
        (s, sl, sh), (c, cl, ch), (f, fl, fh) = (boot(sp, rng), boot(co, rng),
                                                  boot(af, rng))
        print(f"  {a:14} vs {b:9} speed x{math.exp(s):.2f} "
              f"[{math.exp(sl):.2f}, {math.exp(sh):.2f}]  cost "
              f"x{math.exp(c):.2f} [{math.exp(cl):.2f}, {math.exp(ch):.2f}]  "
              f"consistent {f:+.2f} [{fl:+.2f}, {fh:+.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
