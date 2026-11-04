#!/usr/bin/env python3
"""Figures for the real-LLM run (results/llm_runs.jsonl).

Three panels, each answering one question:
  (a) does the protocol prevent the anomaly?   -- the measure the cell tests
  (b) what actually went wrong?                -- A1 vs real concurrency
  (c) how do the paper's two agent assumptions hold up on this model?

Colour follows the method in the `dataviz` skill. Panel (a) encodes a single
measure across labelled categories, so it uses one hue and lets the axis carry
identity -- five categorical hues cannot clear all-pairs CVD separation, and
forcing them would be worse than not encoding by colour at all. Panel (b) is
genuinely categorical and uses the validated slots 1/2/3, which do pass
all-pairs. Every bar is directly labelled, which is also the required relief
for the sub-3:1 contrast of the aqua slot.
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
from matplotlib.patches import Patch                # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coagent.runner import reference_states         # noqa: E402
from coagent.workloads import CELLS                 # noqa: E402

ORDER = ["serial", "naive", "2PL-saga", "OCC-saga", "MTPO"]

# validated default palette, slots 1/2/3
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"

CANARY, GEO = "k8s/deploy/geo-canary", "k8s/deploy/geo"


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson interval -- the normal approximation is useless at k == n."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, centre - half), min(1.0, centre + half)


def classify(rec, ref):
    """Why did this trial fail? Distinguishes A1 from a real CC failure."""
    fs = rec["final_state"]
    canary = (fs.get(CANARY) or {}).get("image")
    geo = (fs.get(GEO) or {}).get("image")
    if canary != geo:
        return "concurrency anomaly"
    sweep = any(
        fs.get(k) != ref.get(k)
        for k in ref
        if k not in (CANARY,)
    )
    if sweep:
        return "A1: sweep incomplete"
    return "A1: create skipped"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "results" / "llm_runs.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results" / "figures"))
    args = ap.parse_args()

    runs = [json.loads(l) for l in open(args.runs)]
    ref = reference_states(CELLS["k8s-canary"])[0]
    by = defaultdict(list)
    for r in runs:
        by[r["protocol"]].append(r)
    protos = [p for p in ORDER if by.get(p)]
    model = runs[0].get("model", "model")

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))

    # ---- (a) anomaly-free rate, with Wilson intervals --------------------
    ax = axes[0]
    xs = range(len(protos))
    pts = []
    for p in protos:
        rs = by[p]
        k = sum(
            1 for r in rs
            if (r["final_state"].get(CANARY) or {}).get("image")
            == (r["final_state"].get(GEO) or {}).get("image")
        )
        pts.append((k, len(rs)) + wilson(k, len(rs))[1:])
    for x, (k, n, lo, hi) in zip(xs, pts):
        p = k / n
        ax.bar(x, p * 100, width=0.56, color=BLUE, zorder=2)
        ax.plot([x, x], [lo * 100, hi * 100], color=INK, lw=2,
                solid_capstyle="round", zorder=3)
        ax.text(x, hi * 100 + 3, f"{k}/{n}", ha="center", va="bottom",
                fontsize=9, color=INK)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(protos, rotation=25, ha="right", fontsize=9)
    for lbl in ax.get_xticklabels():
        if lbl.get_text() == "MTPO":
            lbl.set_fontweight("bold")
            lbl.set_color(INK)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("canary mirrors geo (%)", fontsize=9, color=MUTED)
    ax.set_title("(a) Did the protocol prevent the anomaly?", fontsize=10.5)

    # ---- (b) failure causes ---------------------------------------------
    ax = axes[1]
    causes = ["A1: sweep incomplete", "A1: create skipped",
              "concurrency anomaly"]
    colors = {causes[0]: BLUE, causes[1]: ORANGE, causes[2]: AQUA}
    counts = {p: defaultdict(int) for p in protos}
    for p in protos:
        for r in by[p]:
            if not r["invariant_ok"]:
                counts[p][classify(r, ref)] += 1
    for x, p in zip(xs, protos):
        bottom = 0
        for c in causes:
            v = counts[p][c]
            if not v:
                continue
            ax.bar(x, v, width=0.56, bottom=bottom, color=colors[c],
                   zorder=2, linewidth=1.6, edgecolor="white")
            ax.text(x, bottom + v / 2, str(v), ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
            bottom += v
    ax.set_xticks(list(xs))
    ax.set_xticklabels(protos, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("failed trials", fontsize=9, color=MUTED)
    ax.set_title("(b) Why did trials fail?", fontsize=10.5)
    ax.set_ylim(0, max(sum(counts[p].values()) for p in protos) + 2.6)
    ax.legend(handles=[Patch(facecolor=colors[c], label=c) for c in causes],
              fontsize=8, frameon=False, loc="upper left", ncol=1,
              handlelength=1.1, borderpad=0.2)

    # ---- (c) the two agent assumptions ----------------------------------
    ax = axes[2]
    serial = by.get("serial", [])
    a1_k = sum(r["invariant_ok"] for r in serial)
    a1_n = len(serial) or 1
    mt = by.get("MTPO", [])
    notif = sum(r["notifications"] for r in mt)
    healed = sum(
        1 for r in mt
        if (r["final_state"].get(CANARY) or {}).get("image")
        == (r["final_state"].get(GEO) or {}).get("image")
        and r["notifications"]
    )
    groups = [
        ("A1\nindividual success", 98, 100 * a1_k / a1_n),
        ("A3\nself-healing", 95, 100 * healed / max(notif, 1)),
    ]
    w = 0.34
    for i, (label, paper, got) in enumerate(groups):
        ax.bar(i - w / 2, paper, width=w, color=ORANGE, zorder=2)
        ax.bar(i + w / 2, got, width=w, color=BLUE, zorder=2)
        ax.text(i - w / 2, paper + 2, f"{paper:.0f}", ha="center",
                va="bottom", fontsize=9, color=INK)
        ax.text(i + w / 2, got + 2, f"{got:.0f}", ha="center", va="bottom",
                fontsize=9, color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("success rate (%)", fontsize=9, color=MUTED)
    ax.set_title("(c) The paper's two agent assumptions", fontsize=10.5)
    ax.legend(handles=[Patch(facecolor=ORANGE, label="paper (deepseek-v4-flash)"),
                       Patch(facecolor=BLUE, label=f"this run ({model.split('/')[-1]})")],
              fontsize=8, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=2, handlelength=1.1)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)

    n_tr = len(runs)
    fig.suptitle(
        f"Real LLM Workers on the canary cell — {model}, "
        f"{n_tr} trials  (bars in (a) are Wilson 95% intervals)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    out = Path(args.out) / "llm_canary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
