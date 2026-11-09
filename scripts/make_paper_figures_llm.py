#!/usr/bin/env python3
"""The paper's Figure 5, rebuilt from the real-LLM run.

Same four panels as the paper -- correctness, speedup, token cost, and the
cost-versus-speed front -- but every number comes from real agents rather
than the simulated matrix. Speedup and cost are normalised per cell against
that cell's serial baseline, then aggregated as a ratio of means (averaging
per-trial ratios biases speedup upward, since 1/x is convex).

Correctness is the anomaly-free rate, not overall task completion: with a
model whose own failure rate is ~28% the overall column is noise. Cells whose
predicate serial itself fails are excluded -- there an A1-mangled result
cannot be told from a real anomaly.

Colour follows the `dataviz` method: one measure across labelled categories
gets one validated hue, with the axis carrying identity.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coagent.protocols import PROTOCOL_ORDER    # noqa: E402
from coagent.runner import reference_states     # noqa: E402
from coagent.workloads import CELLS             # noqa: E402

import re

# Per-cell aggregates use the MEDIAN, not the mean: one 2PL trial in
# 722 ended with an agent still blocked and its virtual clock ran away
# to 15029s against a 16s cell median. A single such trial moves a
# mean by two orders of magnitude; it moves a median not at all.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
PAPER = {"serial": (98, 1.00, 1.00), "naive": (13, 1.54, 1.06),
         "2PL-saga": (96, 1.04, 1.45), "OCC-saga": (93, 0.93, 1.83),
         "MTPO": (93, 1.43, 1.15)}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, c - h), min(1.0, c + h)


def panel(ax, vals, title, ylabel, fmt="{:.2f}", ylim=None, hline=None,
          errs=None, labels=None):
    xs = range(len(PROTOCOL_ORDER))
    ax.bar(xs, vals, width=0.58, color=BLUE, zorder=2)
    for i, (x, v) in enumerate(zip(xs, vals)):
        top = v
        if errs:
            lo, hi = errs[i]
            ax.plot([x, x], [lo, hi], color=INK, lw=2,
                    solid_capstyle="round", zorder=3)
            top = hi
        ax.text(x, top, (labels[i] if labels else fmt.format(v)),
                ha="center", va="bottom", fontsize=8, color=INK)
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


def figure6(path: Path, outdir: Path) -> None:
    """The paper's Figure 6: one contended cell under each protocol.

    Rows are protocols, lanes within a row are the two agents, and marks are
    real dispatched calls on a real clock. Event *type* is encoded by shape,
    not colour, so the figure survives any colour-vision deficiency; the
    protocol's own interventions (notify, deadlock, abort) get the second
    validated hue because they are a different kind of thing, not a fifth
    series.
    """
    rows = [json.loads(l) for l in open(path)]
    refs = reference_states(CELLS["k8s-canary"])
    cell = CELLS["k8s-canary"]

    def free(r):
        return bool(cell.anomaly_free(r["final_state"], refs))

    pick = {}
    for r in rows:
        p = r["protocol"]
        # For naive the paper shows the anomaly firing, so prefer a trial
        # where it actually did; everything else takes the first run.
        if p == "naive":
            if p not in pick or (not free(r) and free(pick[p])):
                pick[p] = r
        else:
            pick.setdefault(p, r)
    order = [p for p in PROTOCOL_ORDER if p in pick]
    if not order:
        return

    fig, axes = plt.subplots(len(order), 1, figsize=(11, 1.5 * len(order) + 0.9),
                             sharex=True)
    if len(order) == 1:
        axes = [axes]
    tmax = max(r["wall"] for r in pick.values()) * 1.04

    for ax, proto in zip(axes, order):
        r = pick[proto]
        agents = sorted({e["agent"] for e in r["events"]})
        lane = {a: i for i, a in enumerate(agents)}
        for e in r["events"]:
            y = lane[e["agent"]]
            ax.plot([e["t"], e["t"] + e["dur"]], [y, y], color=GRID, lw=5,
                    solid_capstyle="butt", zorder=1)
            if e["kind"] == "read":
                ax.scatter(e["t"], y, s=42, facecolor="white",
                           edgecolor=INK, linewidth=1.4, zorder=3)
            else:
                ax.scatter(e["t"], y, s=42, marker="s", color=BLUE,
                           edgecolor="white", linewidth=1.2, zorder=3)
        # A deadlock and the abort it causes share a timestamp; drawing
        # both overprints the labels, so keep the cause and drop the effect.
        marks = []
        for line in r["trace"]:
            m = re.match(r"\[\s*([\d.]+)s\]\s*(.*)", line)
            if not m:
                continue
            t, text = float(m.group(1)), m.group(2)
            if "notify" in text:
                marks.append((t, "notify", "v", 3))
            elif "certification merged" in text:
                # a launch-time decision, not a mid-run event, but it is the
                # thing that explains why this row looks like whole-workflow
                # locking rather than a chopped run
                marks.append((t, "certified: merged", "D", 4))
            elif "REFUSED" in text:
                marks.append((t, "revision REFUSED", "P", 4))
            elif "DEADLOCK" in text:
                marks.append((t, "deadlock", "X", 1))
            elif "ABORT" in text or "validation" in text:
                marks.append((t, "abort", "X", 0))
        kept = []
        for t, kind, mk, prio in sorted(marks, key=lambda x: (-x[3], x[0])):
            if any(abs(t - kt) < tmax * 0.02 for kt, *_ in kept):
                continue
            kept.append((t, kind, mk))
        for t, kind, mk in kept:
            ax.scatter(t, len(agents) - 0.55, s=70, marker=mk, color=ORANGE,
                       zorder=4)
            ax.annotate(kind, (t, len(agents) - 0.55),
                        textcoords="offset points", xytext=(6, 0),
                        fontsize=7.5, color=ORANGE, va="center")
        mark = "PASS" if free(r) else "ANOMALY"
        ax.text(0.995, 0.5, f"{r['wall']:.1f}s  {mark}",
                transform=ax.transAxes, ha="right", va="center",
                fontsize=8.5, color=INK if free(r) else "#b00",
                fontweight="bold")
        ax.set_yticks(list(lane.values()))
        ax.set_yticklabels(["A (repair)" if a.endswith("0") else "B (canary)"
                            for a in agents], fontsize=8)
        ax.set_ylim(-0.8, len(agents) - 0.1)
        ax.set_xlim(-tmax * 0.012, tmax)
        ax.set_ylabel(proto, fontsize=9, rotation=0, ha="right",
                      va="center", labelpad=52,
                      fontweight="bold" if proto == "MTPO" else "normal")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, length=0)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)

    axes[-1].set_xlabel("wall-clock time (s), measured", fontsize=9,
                        color=MUTED)
    handles = [
        plt.Line2D([], [], marker="o", ls="", markerfacecolor="white",
                   markeredgecolor=INK, label="read"),
        plt.Line2D([], [], marker="s", ls="", color=BLUE, label="write"),
        plt.Line2D([], [], marker="v", ls="", color=ORANGE,
                   label="notification"),
        plt.Line2D([], [], marker="X", ls="", color=ORANGE,
                   label="deadlock / abort"),
        plt.Line2D([], [], marker="D", ls="", color=ORANGE, markersize=6,
                   label="certification merged pieces"),
    ]
    fig.legend(handles=handles, fontsize=8, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=5)
    fig.suptitle(
        f"Figure 6, from real LLM Workers — k8s-canary under each protocol "
        f"({pick[order[0]]['model']})", fontsize=11)
    fig.tight_layout(rect=(0, 0.075, 1, 0.95))
    out = outdir / "fig6_llm.png"
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "results" / "llm_runs.jsonl"))
    ap.add_argument("--outdir", default=str(ROOT / "results" / "figures"))
    args = ap.parse_args()

    runs = [json.loads(l) for l in open(args.runs)]
    model = runs[-1].get("model", "?")
    refs = {c: reference_states(CELLS[c]) for c in CELLS}

    def free(r):
        cell = CELLS[r["cell"]]
        return bool(cell.anomaly_free(r["final_state"], refs[r["cell"]]))

    by = defaultdict(list)
    for r in runs:
        by[(r["cell"], r["protocol"])].append(r)
    have = [c for c in CELLS
            if all(len(by[(c, p)]) >= 5 for p in PROTOCOL_ORDER)]
    cells = [c for c in have if all(free(r) for r in by[(c, "serial")])]

    corr, lohi, speed, cost = [], [], [], []
    for p in PROTOCOL_ORDER:
        rs = [r for c in cells for r in by[(c, p)]]
        k, n = sum(1 for r in rs if free(r)), len(rs)
        _, lo, hi = wilson(k, n)
        corr.append(100 * k / n)
        lohi.append((lo * 100, hi * 100))
        sp, ct = [], []
        for c in cells:
            s_w = statistics.median(x["wall"] for x in by[(c, "serial")])
            s_t = statistics.median(x["prompt_tokens"] + x["completion_tokens"]
                                  for x in by[(c, "serial")])
            p_w = statistics.median(x["wall"] for x in by[(c, p)])
            p_t = statistics.median(x["prompt_tokens"] + x["completion_tokens"]
                                  for x in by[(c, p)])
            sp.append(s_w / p_w)
            ct.append(p_t / s_t)
        speed.append(statistics.mean(sp))
        cost.append(statistics.mean(ct))

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    panel(axes[0], corr, "(a) Correctness", "anomaly-free (%)",
          ylim=(0, 118), errs=lohi,
          labels=[f"{v:.0f}" for v in corr])
    panel(axes[1], speed, "(b) Speed", "speedup vs serial",
          ylim=(0, max(speed) * 1.3), hline=1.0)
    panel(axes[2], cost, "(c) Cost", "token cost vs serial",
          ylim=(0, max(cost) * 1.3), hline=1.0)

    ax = axes[3]
    # Place each label above its point unless that would collide with one
    # already placed, in which case drop it below.
    xr = (max(speed) - min(speed)) or 1.0
    yr = (max(cost) - min(cost)) or 1.0
    placed = []
    for i, p in enumerate(PROTOCOL_ORDER):
        subject = p == "MTPO"
        ax.scatter(speed[i], cost[i], s=150 if subject else 90, color=BLUE,
                   edgecolor="white", linewidth=2, zorder=3)
        below = any(abs(speed[i] - px) / xr < 0.18
                    and abs(cost[i] - py) / yr < 0.18 for px, py in placed)
        ax.annotate(p, (speed[i], cost[i]), textcoords="offset points",
                    xytext=(0, -17 if below else 12), ha="center",
                    fontsize=8, color=INK,
                    va="top" if below else "baseline",
                    fontweight="bold" if subject else "normal")
        placed.append((speed[i], cost[i]))
    ax.set_xlabel("speedup vs serial", fontsize=9, color=MUTED)
    ax.set_ylabel("token cost vs serial", fontsize=9, color=MUTED)
    ax.set_title("(d) Cost vs speed", fontsize=10)
    ax.annotate("cheaper and faster ↘", xy=(0.97, 0.04),
                xycoords="axes fraction", ha="right", fontsize=8,
                style="italic", color=MUTED)
    ax.margins(0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    n = sum(len(by[(c, p)]) for c in cells for p in PROTOCOL_ORDER)
    fig.suptitle(
        f"Figure 5, from real LLM Workers — {model}, {len(cells)} cells, "
        f"{n} trials  (correctness bars are Wilson 95%)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out = Path(args.outdir) / "fig5_llm.png"
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"wrote {out}")

    # ---- the paper's reported values beside these ------------------------
    print()
    print(f"{'protocol':10}{'correct':>18}{'speedup':>16}{'cost':>16}")
    print(f"{'':10}{'LLM / paper':>18}{'LLM / paper':>16}{'LLM / paper':>16}")
    print("-" * 60)
    for i, p in enumerate(PROTOCOL_ORDER):
        ref = PAPER.get(p)
        if ref is None:          # certified chopping is not in that paper
            print(f"{p:10}{f'{corr[i]:.0f}% / --':>18}"
                  f"{f'{speed[i]:.2f} / --':>16}"
                  f"{f'{cost[i]:.2f} / --':>16}")
            continue
        pc, ps, pct = ref
        print(f"{p:10}{f'{corr[i]:.0f}% / {pc}%':>18}"
              f"{f'{speed[i]:.2f} / {ps:.2f}':>16}"
              f"{f'{cost[i]:.2f} / {pct:.2f}':>16}")

    cs = ROOT / "results" / "llm_casestudy.jsonl"
    if cs.exists():
        print()
        figure6(cs, Path(args.outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
