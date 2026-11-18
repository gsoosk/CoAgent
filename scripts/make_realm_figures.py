#!/usr/bin/env python3
"""Figure for the REALM-Bench P3 runs (concurrent vehicle dispatchers).

Reads results/realm/summary.json (written by score_realm.py), so the numbers
drawn are the ones in its tables.

  (a) anomaly-free trials, pooled over instances, Wilson 95% interval
  (b) speed against each instance's serial median
  (c) cost against each instance's serial median
  (d) REALM-Bench plan quality: passengers served, constraints satisfied
  (e) how conflicts were resolved: deadlock aborts, wounds, notifications
  (f) mean model latency per call: speed ratios hold only if these match

Bars in (b)-(f) are the mean over instances; dots are the instances.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
ORDER = ["serial", "naive", "2PL-saga", "MTPO", "chopping-prio"]


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def tidy(ax, protos) -> None:
    ax.set_xticks(range(len(protos)))
    ax.set_xticklabels(protos, fontsize=8, rotation=20, ha="right")
    for lbl in ax.get_xticklabels():
        if lbl.get_text() == "chopping-prio":
            lbl.set_fontweight("bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def per_instance(summary, proto, key):
    return [ps[proto][key] for ps in summary.values()
            if proto in ps and ps[proto].get(key) is not None]


def bars_with_dots(ax, protos, summary, key, fmt, ref=None) -> None:
    for i, p in enumerate(protos):
        vals = per_instance(summary, p, key)
        if not vals:
            continue
        m = st.mean(vals)
        ax.bar(i, m, width=0.56, color=BLUE, zorder=2)
        ax.scatter([i] * len(vals), vals, s=14, color="white",
                   edgecolor=INK, lw=0.8, zorder=3)
        ax.text(i + 0.31, m, fmt.format(m), va="center", fontsize=7.5,
                color=INK)
    if ref is not None:
        ax.axhline(ref, color=MUTED, lw=1.1, ls="--", zorder=1)
    tidy(ax, protos)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(ROOT / "results/realm/summary.json"))
    ap.add_argument("--out", default=str(ROOT / "results/realm/figures/fig_realm.png"))
    ap.add_argument("--model", default="claude-haiku-4-5")
    args = ap.parse_args()

    summary = json.loads(Path(args.summary).read_text())
    protos = [p for p in ORDER if any(p in ps for ps in summary.values())]
    n_inst = len(summary)

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
    (a, b, c), (d, e, f) = axes

    # (a) isolation
    for i, p in enumerate(protos):
        k = sum(per_instance(summary, p, "af"))
        n = sum(per_instance(summary, p, "n"))
        lo, hi = wilson(k, n)
        a.bar(i, 100 * k / n, width=0.56, color=BLUE, zorder=2)
        a.plot([i, i], [100 * lo, 100 * hi], color=INK, lw=1.6, zorder=3)
        a.text(i, 100 * hi + 2, f"{k}/{n}", ha="center", fontsize=7.5,
               color=INK)
    a.set_ylim(0, 112)
    a.set_ylabel("anomaly-free trials (%)", fontsize=8.5, color=MUTED)
    a.set_title("(a) no double dispatch", fontsize=9.5, loc="left")
    tidy(a, protos)

    bars_with_dots(b, protos, summary, "speedup", "{:.2f}", ref=1.0)
    b.set_ylabel("speed vs serial (×)", fontsize=8.5, color=MUTED)
    b.set_title("(b) speed (higher is faster)", fontsize=9.5, loc="left")

    bars_with_dots(c, protos, summary, "cost", "{:.2f}", ref=1.0)
    c.set_ylabel("cost vs serial (×)", fontsize=8.5, color=MUTED)
    c.set_title("(c) API cost (lower is cheaper)", fontsize=9.5, loc="left")

    # (d) REALM's own metrics
    w = 0.38
    for k, (key, col) in enumerate((("goal_satisfaction", BLUE),
                                    ("constraint_satisfaction", ORANGE))):
        for i, p in enumerate(protos):
            vals = per_instance(summary, p, key)
            if not vals:
                continue
            x = i + (k - 0.5) * w
            d.bar(x, st.mean(vals), width=w * 0.9, color=col, zorder=2)
            d.scatter([x] * len(vals), vals, s=10, color="white",
                      edgecolor=INK, lw=0.7, zorder=3)
    d.set_ylim(0, 125)
    d.set_yticks(range(0, 101, 20))
    d.set_ylabel("REALM-Bench metric (%)", fontsize=8.5, color=MUTED)
    d.set_title("(d) plan quality, REALM-Bench metrics", fontsize=9.5,
                loc="left")
    d.legend(handles=[Patch(facecolor=BLUE, label="passengers served"),
                      Patch(facecolor=ORANGE, label="constraints satisfied")],
             fontsize=7.5, frameon=False, loc="upper left", ncol=2)
    tidy(d, protos)

    # (e) conflict handling
    series = (("deadlocks", "deadlock aborts", ORANGE),
              ("wounds", "wounds", AQUA),
              ("notifications", "notifications", BLUE))
    w = 0.26
    for k, (key, label, col) in enumerate(series):
        for i, p in enumerate(protos):
            vals = per_instance(summary, p, key)
            if not vals or not any(vals):
                continue
            x = i + (k - 1) * w
            e.bar(x, st.mean(vals), width=w * 0.9, color=col, zorder=2)
            e.text(x, st.mean(vals), f"{st.mean(vals):.1f}", ha="center",
                   va="bottom", fontsize=6.8, color=INK)
    e.set_ylabel("per trial", fontsize=8.5, color=MUTED)
    e.set_title("(e) how conflicts were resolved", fontsize=9.5, loc="left")
    e.legend(handles=[Patch(facecolor=col, label=label)
                      for _, label, col in series],
             fontsize=7.5, frameon=False, loc="upper left")
    tidy(e, protos)

    bars_with_dots(f, protos, summary, "latency", "{:.2f}")
    f.set_ylabel("seconds per model call", fontsize=8.5, color=MUTED)
    f.set_title("(f) model latency per call", fontsize=9.5, loc="left")

    fig.suptitle(
        f"REALM-Bench P3 urban ride-sharing, three concurrent vehicle "
        f"dispatchers ({args.model}; {n_inst} instances, "
        f"{sum(per_instance(summary, protos[0], 'n'))} trials per protocol)",
        fontsize=10.5, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
