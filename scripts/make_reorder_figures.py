#!/usr/bin/env python3
"""One figure for the three follow-ups to the anatomy experiment.

Reads results/reorder/summary.json (written by compare_reorder.py), so the
numbers drawn are the ones in the tables.

  (a) order negotiation on the three reorder cells
  (b) write-intent locking: deadlock aborts per trial
  (c) write-intent locking: speed
  (d) minimal merging on the two cells it changes
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
GREY = "#9a9994"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"


def tidy(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=8.5)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def grouped(ax, groups, series, value, fmt="{:.2f}", hline=None) -> None:
    """series: [(key, label, colour)]; value(group, key) -> float or None."""
    n = len(series)
    w = 0.8 / n
    for k, (key, label, col) in enumerate(series):
        xs, vs = [], []
        for g, name in enumerate(groups):
            v = value(name, key)
            if v is None:
                continue
            xs.append(g + (k - (n - 1) / 2) * w)
            vs.append(v)
        ax.bar(xs, vs, width=w * 0.9, color=col, label=label, zorder=2)
        for x, v in zip(xs, vs):
            ax.text(x, v, fmt.format(v), ha="center", va="bottom",
                    fontsize=7.2, color=INK)
    if hline is not None:
        ax.axhline(hline, color=MUTED, lw=1.1, ls="--", zorder=1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=8.5)
    tidy(ax)


def main() -> int:
    S = json.loads((ROOT / "results/reorder/summary.json").read_text())
    scored = [c for c, v in S.items()
              if v["scored"] and c.startswith(("k8s", "wb"))]

    def get(cell, proto, key):
        d = S[cell]["protocols"].get(proto)
        return None if d is None else d[key]

    def mean(proto, key):
        return statistics.mean(get(c, proto, key) for c in scored)

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4),
                             gridspec_kw={"width_ratios": [2.3, 1, 1, 1.3]})

    ax = axes[0]
    grouped(ax, ["ro-free", "ro-dep", "ro-trap"], [
        ("2PL-saga", "2PL", GREY),
        ("chopping", "chopping", BLUE),
        ("chopping-reorder", "reorder, planning call charged", ORANGE),
        ("chopping-reorder-free", "reorder, planning call not charged", AQUA),
        ("MTPO", "MTPO", VIOLET),
    ], lambda c, p: get(c, p, "speedup"), hline=1.0)
    ax.set_ylabel("speedup vs serial", fontsize=9, color=MUTED)
    ax.set_title("(a) Asking the agent to reorder its steps", fontsize=10)
    ax.legend(fontsize=7.6, frameon=False, ncol=2, loc="upper right")
    ax.set_ylim(0, 2.75)

    lock = [("2PL-saga", "2PL", GREY), ("chopping-2pl", "chopping-2pl", BLUE),
            ("chopping-2pl-intent", "chopping-2pl-intent", ORANGE),
            ("chopping-prio", "chopping-prio", AQUA)]
    for ax, key, title, ylabel, hline in (
        (axes[1], "deadlocks", "(b) Deadlock aborts", "aborts per trial", None),
        (axes[2], "speedup", "(c) Speed", "speedup vs serial", 1.0),
    ):
        vals = [mean(p, key) for p, _, _ in lock]
        ax.bar(range(len(lock)), vals, color=[c for _, _, c in lock],
               width=0.62, zorder=2)
        for x, v in enumerate(vals):
            ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8,
                    color=INK)
        if hline is not None:
            ax.axhline(hline, color=MUTED, lw=1.1, ls="--", zorder=1)
        ax.set_xticks(range(len(lock)))
        ax.set_xticklabels([lab for _, lab, _ in lock], fontsize=8,
                           rotation=28, ha="right")
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
        ax.set_title(f"{title}\n{len(scored)} scored benchmark cells",
                     fontsize=10)
        tidy(ax)

    ax = axes[3]
    grouped(ax, ["k8s-rollback-audit", "wb-email-triage"], [
        ("chopping", "chopping", BLUE),
        ("chopping-min", "chopping-min (minimal merge)", ORANGE),
        ("chopping-2pl", "chopping-2pl", GREY),
    ], lambda c, p: get(c, p, "speedup"), hline=1.0)
    ax.set_ylabel("speedup vs serial", fontsize=9, color=MUTED)
    ax.set_title("(d) Merging only what conflicts\n(speed only: serial fails "
                 "the check here)", fontsize=10)
    ax.legend(fontsize=7.6, frameon=False, loc="upper right")
    ax.set_ylim(0, 2.6)

    fig.suptitle("Follow-ups to the anatomy experiment — claude-haiku-4-5, "
                 "10 trials per cell and protocol", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = ROOT / "results" / "reorder" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_reorder.png", dpi=170, facecolor="white")
    print(f"wrote {out / 'fig_reorder.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
