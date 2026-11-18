#!/usr/bin/env python3
"""The three REALM problems side by side: speed and consistency per protocol.

Speed is the paired wall-clock ratio against serial on the same instance and
seed (geometric mean, 95% bootstrap interval). Under each problem is how much
of a serial agent's model time is private work -- the stretch certified
chopping can release -- which is what chopping's gain should track.

    python scripts/make_realm_overview.py
"""

from __future__ import annotations

import math
import random
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_realm import load  # noqa: E402

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9994"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
PROBLEMS = [("realm", "P3 ride-sharing", ("get_travel_times", "write_route")),
            ("realm_wedding", "P5 wedding", ("get_travel_times", "write_plan")),
            ("realm_jssp", "J1 job-shop", ("write_schedule",))]
PROTOS = [("2PL-saga", ORANGE), ("MTPO", AQUA), ("chopping-prio", BLUE)]


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def paired_speed(rows, proto, rng, b=10000):
    by = {(r["cell"], r["seed"], r["protocol"]): r for r in rows}
    lr = [math.log(by[(c, s, "serial")]["wall"] / r["wall"])
          for (c, s, p), r in by.items()
          if p == proto and (c, s, "serial") in by]
    ms = sorted(st.mean(rng.choice(lr) for _ in lr) for _ in range(b))
    return (math.exp(st.mean(lr)), math.exp(ms[int(.025 * b)]),
            math.exp(ms[int(.975 * b)]))


def private_share(rows, prefixes):
    s = [r for r in rows if r["protocol"] == "serial"]
    return st.mean(sum(e["dur"] for e in r["events"]
                       if e["label"].startswith(prefixes))
                   / sum(e["dur"] for e in r["events"]) for r in s)


def tidy(ax, labels):
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=8.5)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def main() -> int:
    rng = random.Random(0)
    data = []
    for d, name, priv in PROBLEMS:
        rows = load(sorted((ROOT / "results" / d / "parts").glob("*.jsonl")))
        data.append((name, rows, private_share(rows, priv)))
    labels = [f"{n}\nprivate work {100 * p:.0f}%" for n, _, p in data]

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    w = 0.26
    for k, (proto, col) in enumerate(PROTOS):
        for i, (_, rows, _) in enumerate(data):
            m, lo, hi = paired_speed(rows, proto, rng)
            x = i + (k - 1) * w
            a.bar(x, m, width=w * 0.9, color=col, zorder=2)
            a.plot([x, x], [lo, hi], color=INK, lw=1.3, zorder=3)
            a.text(x, hi + 0.04, f"{m:.2f}", ha="center", fontsize=7.2,
                   color=INK)
            rs = [r for r in rows if r["protocol"] == proto]
            n, kk = len(rs), sum(bool(r["anomaly_free"]) for r in rs)
            lo2, hi2 = wilson(kk, n)
            b.bar(x, 100 * kk / n, width=w * 0.9, color=col, zorder=2)
            b.plot([x, x], [100 * lo2, 100 * hi2], color=INK, lw=1.3,
                   zorder=3)
    a.axhline(1.0, color=MUTED, lw=1.1, ls="--", zorder=1)
    a.set_ylabel("speed vs serial, paired (x)", fontsize=8.5, color=MUTED)
    a.set_title("(a) speed: chopping's gain tracks the private share",
                fontsize=9.5, loc="left")
    tidy(a, labels)
    for i, (_, rows, _) in enumerate(data):
        rs = [r for r in rows if r["protocol"] == "serial"]
        b.axhline(100 * sum(bool(r["anomaly_free"]) for r in rs) / len(rs),
                  xmin=(i + 0.08) / 3, xmax=(i + 0.92) / 3, color=MUTED,
                  ls="--", lw=1.1, zorder=4)
    b.set_ylim(0, 112)
    b.set_ylabel("consistent trials (%)", fontsize=8.5, color=MUTED)
    b.set_title("(b) consistency (dashed: serial, i.e. agent errors alone)",
                fontsize=9.5, loc="left")
    tidy(b, labels)
    fig.legend(handles=[Patch(facecolor=c, label=p) for p, c in PROTOS],
               loc="upper center", ncol=3, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = ROOT / "results" / "realm_overview.png"
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
