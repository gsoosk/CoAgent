#!/usr/bin/env python3
"""Vector figures for the paper (paper/figs/*.pdf).

Separate from the exploratory PNG scripts on purpose: a paper figure has no
title inside it (the caption carries that), is sized for a column, and has to
survive greyscale printing, so the three chopping variants are separated by
hatch as well as hue.
"""

from __future__ import annotations

import collections
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coagent.runner import reference_states         # noqa: E402
from coagent.workloads import CELLS                 # noqa: E402

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8,
    "font.family": "serif", "axes.linewidth": 0.6,
})

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
ORDER = ["serial", "naive", "2PL-saga", "OCC-saga", "MTPO",
         "chopping", "chopping-2pl", "chopping-prio"]
SHORT = {"serial": "Serial", "naive": "Naive", "2PL-saga": "2PL",
         "OCC-saga": "OCC", "MTPO": "MTPO", "chopping": "CC",
         "chopping-2pl": "CC-2PL", "chopping-prio": r"CC-$\sigma$"}
OURS = {"chopping", "chopping-2pl", "chopping-prio"}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, c - h), min(1.0, c + h)


def load(model):
    rows = [json.loads(l) for l in open(ROOT / "results" / model / "llm_runs.jsonl")
            if l.strip()]
    refs = {c: reference_states(CELLS[c]) for c in CELLS}

    def free(r):
        return bool(CELLS[r["cell"]].anomaly_free(r["final_state"], refs[r["cell"]]))

    by = collections.defaultdict(list)
    for r in rows:
        by[(r["cell"], r["protocol"])].append(r)
    present = [p for p in ORDER if any(k[1] == p for k in by)]
    have = [c for c in CELLS if all(len(by[(c, p)]) >= 5 for p in present)]
    clean = [c for c in have if all(free(r) for r in by[(c, "serial")])]
    return by, present, have, clean, free


def contended(cells):
    """Cells on which every protocol performs the same work."""
    return [c for c in cells if not c.startswith("cc-")]


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=7, length=2)
    ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def fig_main(model, out):
    by, present, _, clean, free = load(model)
    med = statistics.median
    tok = lambda r: r["prompt_tokens"] + r["completion_tokens"]   # noqa: E731
    # Safety is scored on every trustworthy cell. Speed and cost are scored on
    # the contended cells only: on cc-escalate the chopping variants refuse
    # the escalated write, so they finish sooner by doing less, and averaging
    # that in would credit them for work they declined.
    same_work = contended(clean)
    corr, err, speed, cost = [], [], [], []
    for p in present:
        rs = [r for c in clean for r in by[(c, p)]]
        k, n = sum(free(r) for r in rs), len(rs)
        corr.append(100 * k / n)
        err.append(tuple(100 * v for v in wilson(k, n)))
        speed.append(statistics.mean(
            med(x["wall"] for x in by[(c, "serial")]) / med(x["wall"] for x in by[(c, p)])
            for c in same_work))
        cost.append(statistics.mean(
            med(tok(x) for x in by[(c, p)]) / med(tok(x) for x in by[(c, "serial")])
            for c in same_work))
    for p, s, c in zip(present, speed, cost):
        print(f"   {p:14s} speed {s:.2f} cost {c:.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.95))
    labels = [SHORT[p] for p in present]
    cols = [ORANGE if p in OURS else BLUE for p in present]
    xs = list(range(len(present)))
    panels = [(corr, "anomaly-free trials (%)", "{:.0f}", None, (0, 112)),
              (speed, "speedup over serial", "{:.2f}", 1.0, (0, max(speed) * 1.18)),
              (cost, "token cost over serial", "{:.2f}", 1.0, (0, max(cost) * 1.15))]
    for ax, (vals, ylabel, fmt, hline, ylim), tag in zip(axes, panels, "abc"):
        ax.bar(xs, vals, width=0.62, color=cols, zorder=2)
        for i, v in enumerate(vals):
            top = v
            if vals is corr:
                lo, hi = err[i]
                ax.plot([i, i], [lo, hi], color=INK, lw=1.0, zorder=3)
                top = hi
            ax.text(i, top + ylim[1] * 0.012, fmt.format(v), ha="center",
                    va="bottom", fontsize=6.2, color=INK)
        if hline is not None:
            ax.axhline(hline, color=MUTED, lw=0.7, ls="--", zorder=1)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=6.8)
        ax.set_ylabel(ylabel, fontsize=7.5)
        ax.set_ylim(*ylim)
        ax.set_title(f"({tag})", fontsize=8, loc="left", pad=2)
        tidy(ax)
    fig.tight_layout(pad=0.4, w_pad=1.0)
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def fig_percell(model, out):
    by, present, have, _, _ = load(model)
    med = statistics.median
    series = [("chopping", "CC", BLUE, None),
              ("chopping-2pl", "CC-2PL", AQUA, "////"),
              ("chopping-prio", r"CC-$\sigma$", ORANGE, None)]
    series = [s for s in series if s[0] in present]
    rows = []
    for c in have:
        s2 = med(x["wall"] for x in by[(c, "2PL-saga")])
        rows.append((c, [s2 / med(x["wall"] for x in by[(c, p)]) for p, *_ in series]))
    rows.sort(key=lambda r: -r[1][-1])
    n = len(series)
    fig, ax = plt.subplots(figsize=(3.35, 4.3))
    h = 0.8 / n
    ys = range(len(rows))
    for k, (_, label, col, hatch) in enumerate(series):
        off = (k - (n - 1) / 2) * h
        ax.barh([y + off for y in ys], [r[1][k] for r in rows], height=h,
                color=col, hatch=hatch, edgecolor="white", lw=0.3, zorder=2,
                label=label)
        for y, r in zip(ys, rows):
            ax.text(r[1][k] + 0.02, y + off, f"{r[1][k]:.2f}", va="center",
                    fontsize=5.4, color=INK)
    ax.axvline(1.0, color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlabel("speedup over whole-workflow 2PL", fontsize=7.5)
    ax.set_xlim(0, max(max(r[1]) for r in rows) * 1.13)
    # Below the axis: every corner inside the plot has bars or labels in it.
    ax.legend(fontsize=6.8, frameon=False, loc="upper center", ncol=3,
              bbox_to_anchor=(0.42, -0.085), handlelength=1.3,
              columnspacing=1.4)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=6.8, length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def main() -> int:
    out = ROOT / "paper" / "figs"
    out.mkdir(parents=True, exist_ok=True)
    fig_main("haiku", out / "main_haiku.pdf")
    fig_percell("haiku", out / "percell_haiku.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
