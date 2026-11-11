#!/usr/bin/env python3
"""Figures for certified chopping (idea/idea.pdf).

Three figures:

  fig_chopping.png        safety, cost and the Pareto, against every other
                          protocol, from real LLM Workers
  fig_chopping_where.png  where chopping pays and where it correctly
                          collapses to whole-workflow locking
  fig_chopping_admission.png  what certification actually decided

Colour follows the `dataviz` method: a single measure across labelled
categories gets one validated hue and lets the axis carry identity, because
six categorical hues cannot clear all-pairs CVD separation. Where two series
are genuinely needed, slots 1 and 2 are used.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402
from matplotlib.patches import Patch            # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coagent.protocols import PROTOCOL_ORDER    # noqa: E402
from coagent.runner import (                    # noqa: E402
    TrialConfig, reference_states, run_trial,
)
from coagent.workloads import CELLS             # noqa: E402

# Per-cell aggregates use the MEDIAN, not the mean: one 2PL trial in
# 722 ended with an agent still blocked and its virtual clock ran away
# to 15029s against a 16s cell median. A single such trial moves a
# mean by two orders of magnitude; it moves a median not at all.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
SUBJECT = "chopping"


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, c - h), min(1.0, c + h)


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def load(path):
    runs = [json.loads(l) for l in open(path)]
    refs = {c: reference_states(CELLS[c]) for c in CELLS}

    def free(r):
        cell = CELLS[r["cell"]]
        return bool(cell.anomaly_free(r["final_state"], refs[r["cell"]]))

    by = defaultdict(list)
    for r in runs:
        by[(r["cell"], r["protocol"])].append(r)
    # Only the protocols this run recorded: gating on all of PROTOCOL_ORDER
    # drops every cell from a run made before a protocol existed.
    present = [p for p in PROTOCOL_ORDER if any(k[1] == p for k in by)]
    have = [c for c in CELLS
            if all(len(by[(c, p)]) >= 5 for p in present)]
    clean = [c for c in have if all(free(r) for r in by[(c, "serial")])]
    return runs, by, have, clean, free


def bars(ax, vals, labels, title, ylabel, errs=None, fmt="{:.0f}",
         hline=None):
    xs = range(len(labels))
    colours = [ORANGE if l == SUBJECT else BLUE for l in labels]
    ax.bar(xs, vals, width=0.58, color=colours, zorder=2)
    for i, (x, v) in enumerate(zip(xs, vals)):
        top = v
        if errs:
            lo, hi = errs[i]
            ax.plot([x, x], [lo, hi], color=INK, lw=2,
                    solid_capstyle="round", zorder=3)
            top = hi
        ax.text(x, top, fmt.format(v), ha="center", va="bottom", fontsize=8,
                color=INK)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8.5)
    for lbl in ax.get_xticklabels():
        if lbl.get_text() == SUBJECT:
            lbl.set_fontweight("bold")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    if hline is not None:
        ax.axhline(hline, color=MUTED, lw=1, ls="--", zorder=1)
    tidy(ax)


def figure_main(by, clean, free, model, outdir):
    # Derived here rather than passed in: the same rule as in load(), so a
    # run missing a protocol gets a shorter chart instead of a zero divide.
    present = [p for p in PROTOCOL_ORDER if any(k[1] == p for k in by)]
    corr, errs, speed, cost = [], [], [], []
    for p in present:
        rs = [r for c in clean for r in by[(c, p)]]
        k, n = sum(1 for r in rs if free(r)), len(rs)
        _, lo, hi = wilson(k, n)
        corr.append(100 * k / n)
        errs.append((lo * 100, hi * 100))
        speed.append(statistics.mean(
            statistics.median(x["wall"] for x in by[(c, "serial")])
            / statistics.median(x["wall"] for x in by[(c, p)]) for c in clean))
        cost.append(statistics.mean(
            statistics.median(x["prompt_tokens"] + x["completion_tokens"]
                            for x in by[(c, p)])
            / statistics.median(x["prompt_tokens"] + x["completion_tokens"]
                              for x in by[(c, "serial")]) for c in clean))

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    bars(axes[0], corr, present, "(a) Safety", "anomaly-free (%)",
         errs=errs)
    axes[0].set_ylim(0, 118)
    bars(axes[1], speed, present, "(b) Speed", "speedup vs serial",
         fmt="{:.2f}", hline=1.0)
    bars(axes[2], cost, present, "(c) Cost", "token cost vs serial",
         fmt="{:.2f}", hline=1.0)

    ax = axes[3]
    xr = (max(speed) - min(speed)) or 1.0
    yr = (max(corr) - min(corr)) or 1.0
    for i, p in enumerate(present):
        subj = p == SUBJECT
        ax.scatter(speed[i], corr[i], s=170 if subj else 95,
                   color=ORANGE if subj else BLUE, edgecolor="white",
                   linewidth=2, zorder=3)
        # Look at every other point, not just the ones already drawn: the
        # label goes above by default, so what matters is whether some point
        # sits in that space -- and the one that does may come later in the
        # loop. Checking only `placed` put MTPO's label on top of
        # chopping-prio, whose point had not been plotted yet.
        above = any(j != i
                    and abs(speed[i] - speed[j]) / xr < 0.18
                    and 0 <= (corr[j] - corr[i]) / yr < 0.22
                    for j in range(len(present)))
        ax.annotate(p, (speed[i], corr[i]), textcoords="offset points",
                    xytext=(0, -17 if above else 12), ha="center",
                    fontsize=8, color=INK, va="top" if above else "baseline",
                    fontweight="bold" if subj else "normal")
    ax.set_xlabel("speedup vs serial", fontsize=9, color=MUTED)
    ax.set_ylabel("anomaly-free (%)", fontsize=9, color=MUTED)
    ax.set_title("(d) Safety vs speed", fontsize=10)
    ax.annotate("safer and faster ↗", xy=(0.97, 0.05),
                xycoords="axes fraction", ha="right", fontsize=8,
                style="italic", color=MUTED)
    ax.margins(0.22)
    tidy(ax)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)

    n = sum(len(by[(c, p)]) for c in clean for p in present)
    fig.suptitle(f"Certified chopping against every other protocol — "
                 f"{model}, {len(clean)} cells, {n} trials", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(outdir / "fig_chopping.png", dpi=170, facecolor="white")
    print(f"wrote {outdir / 'fig_chopping.png'}")


def figure_where(by, have, outdir):
    """Per-cell speedup of chopping over whole-workflow 2PL."""
    rows = []
    for c in have:
        s2 = statistics.median(x["wall"] for x in by[(c, "2PL-saga")])
        sc = statistics.median(x["wall"] for x in by[(c, SUBJECT)])
        rows.append((c, s2 / sc))
    rows.sort(key=lambda r: -r[1])

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ys = range(len(rows))
    cols = [AQUA if v > 1.15 else BLUE for _, v in rows]
    ax.barh(list(ys), [v for _, v in rows], color=cols, height=0.62, zorder=2)
    for y, (c, v) in zip(ys, rows):
        ax.text(v + 0.012, y, f"{v:.2f}x", va="center", fontsize=8, color=INK)
    ax.axvline(1.0, color=MUTED, lw=1, ls="--", zorder=1)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([c for c, _ in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("speedup of certified chopping over whole-workflow 2PL",
                  fontsize=9, color=MUTED)
    ax.set_title("Where chopping pays, and where it correctly collapses",
                 fontsize=10.5)
    ax.annotate("a cell whose read-modify-write straddles the reasoning gap\n"
                "is merged back to whole-workflow locking — 1.0x is the\n"
                "right answer there, not a failure",
                xy=(0.52, 0.06), xycoords="axes fraction", fontsize=8,
                color=MUTED, style="italic")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(outdir / "fig_chopping_where.png", dpi=170,
                facecolor="white")
    print(f"wrote {outdir / 'fig_chopping_where.png'}")


def figure_where_prio(by, have, model, outdir):
    """Per-cell speedup over whole-workflow 2PL, for each chopping variant.

    All three series share the 2PL baseline, so the chart reads as one
    comparison. chopping-2pl sitting on the 1.0 line is the point of the
    figure as much as chopping-prio clearing it: on a contended cell
    certification coarsens the workflow into a single maximal piece, and
    2PL inside a piece that spans everything is just 2PL.
    """
    series = [("chopping", "certified chopping", BLUE),
              ("chopping-2pl", "chopping-2pl (incremental locks)", AQUA),
              ("chopping-prio", "chopping-prio (sigma-ordered)", ORANGE)]
    series = [s for s in series if any(k[1] == s[0] for k in by)]
    if len(series) < 2:
        return

    rows = []
    for c in have:
        s2 = statistics.median(x["wall"] for x in by[(c, "2PL-saga")])
        rows.append((c, [s2 / statistics.median(x["wall"] for x in by[(c, p)])
                         for p, _, _ in series]))
    rows.sort(key=lambda r: -r[1][-1])

    n = len(series)
    fig, ax = plt.subplots(figsize=(9.8, 0.185 * len(rows) * n + 1.5))
    ys = range(len(rows))
    h = 0.78 / n
    for k, (_, label, col) in enumerate(series):
        off = (k - (n - 1) / 2) * h
        ax.barh([y + off for y in ys], [r[1][k] for r in rows], height=h,
                color=col, zorder=2, label=label)
        for y, r in zip(ys, rows):
            v = r[1][k]
            ax.text(v + 0.02, y + off, f"{v:.2f}x", va="center", fontsize=7.2,
                    color=INK,
                    fontweight="bold" if v == max(r[1]) else "normal")
    ax.axvline(1.0, color=MUTED, lw=1.2, ls="--", zorder=1)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("speedup over whole-workflow 2PL", fontsize=9, color=MUTED)
    ax.set_title("What each locking discipline adds, cell by cell — "
                 f"{model}", fontsize=10.5)
    ax.annotate("on the dashed line = no better than 2PL",
                xy=(0.015, 0.008), xycoords="axes fraction", fontsize=8,
                color=MUTED, style="italic")
    ax.legend(fontsize=8.2, frameon=False, loc="lower right")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(max(r[1]) for r in rows) * 1.14)
    fig.tight_layout()
    fig.savefig(outdir / "fig_prio_where.png", dpi=170, facecolor="white")
    print(f"wrote {outdir / 'fig_prio_where.png'}")


def figure_admission(by, have, outdir):
    """What certification actually decided, at launch and mid-flight."""
    merged = kept = 0
    refused = admitted = coarsened = 0
    per_cell = {}
    for c in have:
        m = sum(1 for r in by[(c, SUBJECT)]
                if any("certification merged" in t for t in r["trace"]))
        ref = sum(1 for r in by[(c, SUBJECT)]
                  if any("REFUSED" in t for t in r["trace"]))
        adm = sum(1 for r in by[(c, SUBJECT)]
                  if any("revision admitted" in t for t in r["trace"]))
        coa = sum(1 for r in by[(c, SUBJECT)]
                  if any("coarsened on" in t for t in r["trace"]))
        n = len(by[(c, SUBJECT)])
        merged += m
        kept += n - m
        refused += ref
        admitted += adm
        coarsened += coa
        per_cell[c] = (m, n - m, ref)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    cells = list(per_cell)
    ys = range(len(cells))
    ax.barh(list(ys), [per_cell[c][0] for c in cells], color=ORANGE,
            height=0.6, label="pieces merged at launch", zorder=2)
    ax.barh(list(ys), [per_cell[c][1] for c in cells],
            left=[per_cell[c][0] for c in cells], color=BLUE, height=0.6,
            label="proposed chopping kept", zorder=2)
    for y, c in zip(ys, cells):
        if per_cell[c][2]:
            ax.text(10.5, y, f"{per_cell[c][2]} refused", va="center",
                    fontsize=8, color="#b00", fontweight="bold")
    ax.set_yticks(list(ys))
    ax.set_yticklabels(cells, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("trials", fontsize=9, color=MUTED)
    ax.set_title("(a) Certification outcome per cell", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_xlim(0, 13.5)

    ax = axes[1]
    total = admitted + coarsened + refused
    vals = [admitted, coarsened, refused]
    labs = ["admitted\nas proposed", "admitted\ncoarsened", "REFUSED"]
    cols = [BLUE, AQUA, "#b00"]
    ax.bar(range(3), vals, color=cols, width=0.56, zorder=2)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.2, str(v), ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labs, fontsize=8.5)
    ax.set_ylabel("mid-flight revisions", fontsize=9, color=MUTED)
    ax.set_title(f"(b) Admission outcomes ({total} revisions)", fontsize=10)
    ax.set_ylim(0, max(vals) * 1.35 if vals and max(vals) else 1)
    tidy(ax)
    ax.annotate("every refusal had its cycle through a committed\npiece (Corollary 1). The only cell here that\nescalates is built to be inadmissible, so this\nis a correctness result, not an admission rate.",
                xy=(0.02, 0.72), xycoords="axes fraction", fontsize=7.5,
                color=MUTED, style="italic", va="top")

    fig.suptitle("What certification decided", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(outdir / "fig_chopping_admission.png", dpi=170,
                facecolor="white")
    print(f"wrote {outdir / 'fig_chopping_admission.png'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="haiku",
                    help="results subfolder for one model: "
                         "sets both the runs file and the "
                         "figure output directory")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    base = ROOT / "results" / args.dir
    if args.runs is None:
        args.runs = str(base / "llm_runs.jsonl")
    if getattr(args, "outdir", None) is None:
        args.outdir = str(base / "figures")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    runs, by, have, clean, free = load(args.runs)
    model = runs[-1].get("model", "?")
    figure_main(by, clean, free, model, outdir)
    figure_where(by, have, outdir)
    figure_where_prio(by, have, model, outdir)
    figure_admission(by, have, outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
