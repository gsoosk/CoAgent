#!/usr/bin/env python3
"""Redraw Figure 5 (and its paper-vs-reproduction companion) from matrix.csv."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coagent.protocols import PROTOCOL_ORDER   # noqa: E402

# Protocol is the category axis, not a series, so a single measure gets a
# single hue and the axis label carries identity. Five categorical hues cannot
# clear all-pairs CVD separation -- the old palette put MTPO and OCC at
# deutan dE 5.9, i.e. indistinguishable to a red-green colourblind reader,
# which is precisely the comparison the figure exists to make.
BLUE, ORANGE = "#2a78d6", "#eb6834"      # validated categorical slots 1 and 2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
COLORS = {p: BLUE for p in
          ["serial", "naive", "2PL-saga", "OCC-saga", "MTPO"]}
PAPER = {
    "serial": (98, 1.00, 1.00), "naive": (13, 1.54, 1.06),
    "2PL-saga": (96, 1.04, 1.45), "OCC-saga": (93, 0.93, 1.83),
    "MTPO": (93, 1.43, 1.15),
}


def load(path: Path):
    rows = list(csv.DictReader(open(path)))
    by = defaultdict(list)
    for r in rows:
        by[(r["cell"], r["protocol"])].append(r)
    cells = sorted({r["cell"] for r in rows})
    s_wall = {c: statistics.mean(float(x["wall_clock"])
                                 for x in by[(c, "serial")]) for c in cells}
    s_tok = {c: statistics.mean(float(x["tokens"])
                                for x in by[(c, "serial")]) for c in cells}
    out = {}
    for p in PROTOCOL_ORDER:
        rs = [r for c in cells for r in by[(c, p)]]
        if not rs:          # a protocol this CSV predates
            continue
        corr = 100.0 * sum(int(r["invariant_ok"]) for r in rs) / len(rs)
        sp = (statistics.mean(s_wall[r["cell"]] for r in rs)
              / statistics.mean(float(r["wall_clock"]) for r in rs))
        cost = (statistics.mean(float(r["tokens"]) for r in rs)
                / statistics.mean(s_tok[r["cell"]] for r in rs))
        out[p] = (corr, sp, cost)
    return out, len(cells)


def bar_panel(ax, vals, title, ylabel, fmt="{:.2f}", ylim=None, hline=None,
              labels=None):
    xs = range(len(vals))
    ax.bar(xs, vals, width=0.58, color=BLUE, zorder=2)
    for x, v in zip(xs, vals):
        ax.text(x, v, fmt.format(v), ha="center", va="bottom", fontsize=8,
                color=INK)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(PROTOCOL_ORDER, rotation=30, ha="right", fontsize=8)
    for lbl in ax.get_xticklabels():
        if lbl.get_text() == "MTPO":
            lbl.set_fontweight("bold")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    if ylim:
        ax.set_ylim(*ylim)
    if hline is not None:
        ax.axhline(hline, color=MUTED, lw=1, ls="--", zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "results" / "matrix.csv"))
    ap.add_argument("--outdir", default=str(ROOT / "results" / "figures"))
    args = ap.parse_args()
    got, n_cells = load(Path(args.csv))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Figure 5 redrawn from our data -----------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    order = [q for q in PROTOCOL_ORDER if q in got]
    bar_panel(axes[0], [got[p][0] for p in order],
              "(a) Correctness", "Invariant-OK (%)", "{:.0f}", (0, 108),
              labels=order)
    bar_panel(axes[1], [got[p][1] for p in order],
              "(b) Speed", "Speedup vs serial", ylim=(0, 1.9), hline=1.0,
              labels=order)
    bar_panel(axes[2], [got[p][2] for p in order],
              "(c) Cost", "Cost vs serial", ylim=(0, 2.1), hline=1.0,
              labels=order)
    ax = axes[3]
    placed = []
    xr = (max(got[q][1] for q in order) - min(got[q][1] for q in order)) or 1.0
    yr = (max(got[q][2] for q in order) - min(got[q][2] for q in order)) or 1.0
    for p in order:
        subject = p == "MTPO"
        ax.scatter(got[p][1], got[p][2], s=150 if subject else 90,
                   color=BLUE, edgecolor="white", linewidth=2, zorder=3)
        # drop a label below its point when it would collide with one already
        # placed, so overlapping protocols stay readable
        below = any(abs(got[p][1] - px) / xr < 0.16
                    and abs(got[p][2] - py) / yr < 0.16 for px, py in placed)
        ax.annotate(p, (got[p][1], got[p][2]), textcoords="offset points",
                    xytext=(0, -16 if below else 12), ha="center", fontsize=8,
                    color=INK, va="top" if below else "baseline",
                    fontweight="bold" if subject else "normal")
        placed.append((got[p][1], got[p][2]))
    ax.set_xlabel("Speedup vs serial", fontsize=9, color=MUTED)
    ax.set_ylabel("Cost vs serial", fontsize=9, color=MUTED)
    ax.set_title("(d) Cost vs speed", fontsize=10)
    ax.annotate("cheaper and faster \u2198", xy=(0.97, 0.04),
                xycoords="axes fraction", ha="right", fontsize=8,
                style="italic", color=MUTED)
    ax.margins(0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    fig.suptitle(f"Figure 5 reproduced: {len(order)} protocols on "
                 f"{n_cells} cells", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outdir / "fig5_reproduction.png", dpi=170)

    # --- side-by-side with the paper's reported values --------------------
    # This panel compares against the CoAgent paper, so it shows only the
    # five protocols that paper reports. Certified chopping is a separate
    # contribution and has its own figures.
    paper_order = [q for q in order if q in PAPER]
    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 3.5))
    titles = [("Correctness (%)", 0, "{:.0f}"), ("Speedup vs serial", 1, "{:.2f}"),
              ("Cost vs serial", 2, "{:.2f}")]
    xs = range(len(paper_order))
    for ax, (title, idx, fmt) in zip(axes2, titles):
        w = 0.38
        pv = [PAPER[p][idx] for p in paper_order]
        gv = [got[p][idx] for p in paper_order]
        ax.bar([x - w / 2 for x in xs], pv, w, label="paper",
               color=ORANGE, edgecolor="none")
        ax.bar([x + w / 2 for x in xs], gv, w, label="reproduction",
               color=BLUE, edgecolor="none")
        for x, v in zip(xs, pv):
            ax.text(x - w / 2, v, fmt.format(v), ha="center", va="bottom",
                    fontsize=7)
        for x, v in zip(xs, gv):
            ax.text(x + w / 2, v, fmt.format(v), ha="center", va="bottom",
                    fontsize=7)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(paper_order, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    fig2.legend(*axes2[0].get_legend_handles_labels(), fontsize=8,
                frameon=False, loc="lower center", ncol=2,
                bbox_to_anchor=(0.5, -0.01))
    fig2.suptitle("Reported vs reproduced", fontsize=11)
    fig2.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig2.savefig(outdir / "paper_vs_reproduction.png", dpi=170)

    print(f"wrote {outdir / 'fig5_reproduction.png'}")
    print(f"wrote {outdir / 'paper_vs_reproduction.png'}")

    # --- Figure 7a: online tool-library growth ----------------------------
    ts_csv = Path(args.csv).parent / "toolsmith.csv"
    if ts_csv.exists():
        rows = list(csv.DictReader(open(ts_csv)))
        task = [int(r["task"]) for r in rows]
        fig3, ax3 = plt.subplots(figsize=(6.2, 3.6))
        for key, label, color in [
            ("snapshot_read", "snapshot read", "#1f8ac0"),
            ("live_read", "live read", "#ff8c42"),
            ("undoable_write", "undoable write", "#d7301f"),
            ("tools_total", "total", "#333333"),
        ]:
            ax3.plot(task, [int(r[key]) for r in rows], label=label,
                     color=color, lw=2 if key == "tools_total" else 1.4)
            ax3.annotate(rows[-1][key], (task[-1], int(rows[-1][key])),
                         textcoords="offset points", xytext=(6, -3),
                         fontsize=8, color=color)
        half = next(r for r in rows
                    if int(r["tools_total"]) >= int(rows[-1]["tools_total"]) / 2)
        ax3.axvline(int(half["task"]), color="#999", ls="--", lw=1)
        ax3.annotate(f"half the library\nby task {half['task']}",
                     (int(half["task"]), 3), textcoords="offset points",
                     xytext=(8, 0), fontsize=8, color="#666")
        ax3.set_xlabel("task processing step (random order)", fontsize=9)
        ax3.set_ylabel("cumulative tool count", fontsize=9)
        ax3.set_title("Figure 7a reproduced: online tool-library growth",
                      fontsize=10)
        ax3.legend(fontsize=8, frameon=False, loc="lower right")
        ax3.spines[["top", "right"]].set_visible(False)
        ax3.set_xlim(0, max(task) + 4)
        fig3.tight_layout()
        fig3.savefig(outdir / "fig7a_toolsmith_growth.png", dpi=170)
        print(f"wrote {outdir / 'fig7a_toolsmith_growth.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
