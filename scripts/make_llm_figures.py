#!/usr/bin/env python3
"""Figures for the real-LLM run (results/llm_runs.jsonl).

Three panels:
  (a) anomaly-free rate per protocol, aggregated -- the measure the cells test
  (b) the same, per cell -- shows which workloads separate the protocols
  (c) the paper's two agent assumptions, measured rather than assumed

Colour follows the `dataviz` method. Panel (a) is one measure across labelled
categories, so it uses a single validated hue and lets the axis carry
identity; five categorical hues cannot clear all-pairs CVD separation. Panel
(b) is magnitude, so it uses one sequential ramp, light to dark. Panel (c) has
two series and uses validated slots 1 and 2. Every cell is directly labelled,
which is also the required relief for sub-3:1 contrast.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch                # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coagent.protocols import PROTOCOL_ORDER        # noqa: E402
from coagent.runner import reference_states         # noqa: E402
from coagent.workloads import CELLS                 # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"      # validated categorical slots 1 and 2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
RAMP = LinearSegmentedColormap.from_list("blues", ["#eaf1fb", "#2a78d6"])


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, c - h), min(1.0, c + h)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "results" / "llm_runs.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results" / "figures"))
    ap.add_argument("--min-trials", type=int, default=5)
    args = ap.parse_args()

    runs = [json.loads(l) for l in open(args.runs)]
    model = runs[-1].get("model", "?")
    refs = {c: reference_states(CELLS[c]) for c in {r["cell"] for r in runs}
            if c in CELLS}

    def free(r):
        cell = CELLS.get(r["cell"])
        if cell is None or cell.anomaly_free is None:
            return False
        return bool(cell.anomaly_free(r["final_state"], refs[r["cell"]]))

    by = defaultdict(list)
    for r in runs:
        by[(r["cell"], r["protocol"])].append(r)
    have = [c for c in CELLS
            if all(len(by.get((c, p), [])) >= args.min_trials
                   for p in PROTOCOL_ORDER)]
    # Serial has no concurrency, so a cell where serial itself registers an
    # anomaly has a predicate that cannot tell an A1-mangled result from a
    # real one. Those cells are shown in panel (b) but excluded from the
    # aggregate, rather than silently averaged in.
    cells = [c for c in have
             if all(free(r) for r in by[(c, "serial")])] or have
    excluded = [c for c in have if c not in cells]
    if not cells:
        print("no cell has enough trials yet")
        return 1

    fig, axes = plt.subplots(
        1, 3, figsize=(15.5, 4.6),
        gridspec_kw={"width_ratios": [1.0, 1.5, 0.9]},
    )

    # ---- (a) aggregate anomaly-free, Wilson intervals --------------------
    ax = axes[0]
    xs = range(len(PROTOCOL_ORDER))
    for x, p in zip(xs, PROTOCOL_ORDER):
        rs = [r for c in cells for r in by[(c, p)]]
        k, n = sum(1 for r in rs if free(r)), len(rs)
        _, lo, hi = wilson(k, n)
        ax.bar(x, 100 * k / n, width=0.56, color=BLUE, zorder=2)
        ax.plot([x, x], [lo * 100, hi * 100], color=INK, lw=2,
                solid_capstyle="round", zorder=3)
        ax.text(x, hi * 100 + 2.5, f"{k}/{n}", ha="center", va="bottom",
                fontsize=8, color=INK)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(PROTOCOL_ORDER, rotation=28, ha="right", fontsize=8.5)
    for lbl in ax.get_xticklabels():
        if lbl.get_text() == "MTPO":
            lbl.set_fontweight("bold")
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("anomaly-free (%)", fontsize=9, color=MUTED)
    ax.set_title(f"(a) Anomaly prevented ({len(cells)} cells)", fontsize=10)

    # ---- (b) per cell ----------------------------------------------------
    ax = axes[1]
    shown = cells + excluded
    grid = [[100 * sum(1 for r in by[(c, p)] if free(r)) / len(by[(c, p)])
             for p in PROTOCOL_ORDER] for c in shown]
    im = ax.imshow(grid, cmap=RAMP, vmin=0, vmax=100, aspect="auto")
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 55 else INK)
    ax.set_xticks(range(len(PROTOCOL_ORDER)))
    ax.set_xticklabels(PROTOCOL_ORDER, rotation=28, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(shown)))
    ax.set_yticklabels(
        [c + ('  (predicate unreliable)' if c in excluded else '')
         for c in shown], fontsize=8)
    ax.set_title("(b) Anomaly-free rate per cell (%)", fontsize=10)
    for i, c in enumerate(shown):
        if c in excluded:
            ax.axhline(i, color="white", lw=0)
            for sp in range(len(PROTOCOL_ORDER)):
                ax.add_patch(plt.Rectangle(
                    (sp - 0.5, i - 0.5), 1, 1, fill=False,
                    edgecolor="#b00", lw=1.4, zorder=5))
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    # ---- (c) the two assumptions -----------------------------------------
    ax = axes[2]
    ser = [r for c in cells for r in by[(c, "serial")]]
    a1 = 100 * sum(r["invariant_ok"] for r in ser) / max(len(ser), 1)
    mt = [r for c in cells for r in by[(c, "MTPO")] if r["notifications"]]
    healed = sum(1 for r in mt if free(r))
    a3 = 100 * healed / max(len(mt), 1)
    _, a3lo, a3hi = wilson(healed, len(mt))
    groups = [("A1\nindividual success", 98, a1, None, None),
              ("A3\nself-healing", 95, a3, a3lo * 100, a3hi * 100)]
    w = 0.34
    for i, (label, paper, got, lo, hi) in enumerate(groups):
        ax.bar(i - w / 2, paper, width=w, color=ORANGE, zorder=2)
        ax.bar(i + w / 2, got, width=w, color=BLUE, zorder=2)
        if lo is not None:
            ax.plot([i + w / 2, i + w / 2], [lo, hi], color=INK, lw=2,
                    solid_capstyle="round", zorder=3)
        ax.text(i - w / 2, paper + 2, f"{paper:.0f}", ha="center",
                va="bottom", fontsize=8.5, color=INK)
        ax.text(i + w / 2, (hi if hi else got) + 2, f"{got:.0f}",
                ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([g[0] for g in groups], fontsize=8.5)
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("success rate (%)", fontsize=9, color=MUTED)
    ax.set_title("(c) The paper's agent assumptions", fontsize=10)
    ax.legend(handles=[Patch(facecolor=ORANGE, label="paper (deepseek-v4-flash)"),
                       Patch(facecolor=BLUE, label=f"measured ({model.split('/')[-1]})")],
              fontsize=7.5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.17), ncol=1, handlelength=1.1)

    for ax in (axes[0], axes[2]):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)

    n_tr = sum(len(by[(c, p)]) for c in cells for p in PROTOCOL_ORDER)
    fig.suptitle(
        f"Real LLM Workers — {model}, {len(cells)} cells, {n_tr} trials "
        f"(intervals are Wilson 95%)", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    out = Path(args.out) / "llm_matrix.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"wrote {out}  ({len(cells)} complete cells, {n_tr} trials)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
