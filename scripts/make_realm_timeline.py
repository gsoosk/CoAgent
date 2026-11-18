#!/usr/bin/env python3
"""One REALM P3 trial, drawn per protocol: who ran when, and who waited.

Each lane is one vehicle's dispatcher on the virtual clock. Bars are model
calls, coloured by phase: the shared claim (list requests, read or claim a
request) or the private routing (travel times, write the route). Hatched
grey bars are calls in an attempt that was later aborted. A dotted line is
time spent blocked on a lock. The pale band behind a lane is when that
agent holds locks on the ride requests: to the end of the workflow under
2PL, only through the claim under certified chopping.

    python scripts/make_realm_timeline.py --cell realm-p3-001 --seed 2
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9a9994"
INK, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#d9d8d3", "#fbe6dc"
PROTOS = ["serial", "2PL-saga", "MTPO", "chopping-prio"]
STAMP = re.compile(r"\[\s*([\d.]+)s\] (\S+) (blocked on|ABORT)")
SHARED = ("list_requests", "assign_request", "get_object")


def lanes(r):
    """Per agent: calls placed when they actually ran, waits, and aborts.

    An event is stamped when the agent issues the call. The runtime charges
    the call's latency once the protocol grants it, so a call that blocked
    ran from (next call's issue time - its duration); the gap before that
    is the wait. A call still blocked when its agent was aborted never ran.
    """
    aborts = defaultdict(list)
    for line in r["trace"]:
        m = STAMP.match(line)
        if m and m.group(3) == "ABORT":
            aborts[m.group(2)].append(float(m.group(1)))
    raw = defaultdict(list)
    for e in r["events"]:
        raw[e["agent"]].append(e)
    calls, waits = defaultdict(list), defaultdict(list)
    for a, evs in raw.items():
        evs = sorted(evs, key=lambda e: e["t"])
        for i, e in enumerate(evs):
            nxt = evs[i + 1]["t"] if i + 1 < len(evs) else None
            cut = [x for x in aborts.get(a, []) if x >= e["t"]
                   and (nxt is None or x <= nxt)]
            if cut and (nxt is None or nxt - e["t"] > e["dur"] + 1e-6):
                # still waiting (or running) when the abort hit
                end = min(cut)
                ran = end - e["t"] >= e["dur"] - 1e-6
                if ran:
                    calls[a].append(dict(e))
                else:
                    waits[a].append((e["t"], end))
                continue
            start = e["t"] if nxt is None else max(e["t"], nxt - e["dur"])
            if start - e["t"] > 1e-6:
                waits[a].append((e["t"], start))
            calls[a].append(dict(e, t=start))
    return calls, aborts, waits


def draw(ax, r, proto, serial_wall, vehicles) -> None:
    calls, aborts, waits = lanes(r)
    agents = sorted(calls)
    for row, a in enumerate(agents):
        y = len(agents) - 1 - row
        evs = sorted(calls[a], key=lambda e: e["t"])
        last_abort = max(aborts.get(a, [-1]))
        live = [e for e in evs if e["t"] > last_abort]
        # lock band: from the final attempt's first call to the release
        if proto in ("2PL-saga", "chopping-prio") and live:
            t0 = live[0]["t"]
            if proto == "2PL-saga":
                t1 = live[-1]["t"] + live[-1]["dur"]
            else:
                private = [e for e in live
                           if not e["label"].startswith(SHARED)]
                t1 = private[0]["t"] if private else live[-1]["t"]
            ax.barh(y, t1 - t0, left=t0, height=0.86, color=BAND, zorder=1)
        for e in evs:
            wasted = any(x >= e["t"] for x in aborts.get(a, []))
            shared = e["label"].startswith(SHARED)
            ax.barh(y, e["dur"], left=e["t"], height=0.5,
                    color=GREY if wasted else (ORANGE if shared else BLUE),
                    hatch="////" if wasted else None, edgecolor="white",
                    lw=0.6, zorder=3)
        for t0, t1 in waits.get(a, []):
            ax.plot([t0, t1], [y, y], color=INK, lw=1, ls=":", zorder=2)
        for t in aborts.get(a, []):
            ax.plot(t, y, marker="x", color=INK, ms=6, mew=1.5, zorder=4)
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels([vehicles[int(a.split("-")[1])] for a in
                        reversed(agents)], fontsize=8)
    ax.set_title(f"{proto}: {r['wall']:.0f} s, {serial_wall / r['wall']:.2f}x "
                 f"serial" + ("" if r.get("anomaly_free") else
                              ", INCONSISTENT"),
                 fontsize=9.5, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=8)
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="realm-p3-001")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--parts", default=str(ROOT / "results/realm/parts"))
    ap.add_argument("--out", default=str(
        ROOT / "results/realm/figures/fig_realm_timeline.png"))
    args = ap.parse_args()
    rows = [json.loads(l) for l in
            (Path(args.parts) / f"{args.cell}.jsonl").read_text().splitlines()
            if l.strip()]
    by = {r["protocol"]: r for r in rows if r["seed"] == args.seed}
    sys_path = str(ROOT / "src")
    import sys
    sys.path.insert(0, sys_path)
    from coagent.workloads.realm_urs import REALM_CELLS, load_instance
    inst = load_instance(REALM_CELLS[args.cell].llm_spec["instance"])
    vehicles = [v["vehicle_id"] for v in inst["vehicles"]]

    fig, axes = plt.subplots(len(PROTOS), 1, figsize=(11, 8.2), sharex=True)
    for ax, p in zip(axes, PROTOS):
        draw(ax, by[p], p, by["serial"]["wall"], vehicles)
    axes[-1].set_xlabel("virtual time (s): each call lasts its measured "
                        "model latency", fontsize=8.5, color=MUTED)
    fig.legend(handles=[
        Patch(facecolor=ORANGE, label="shared: list / claim requests"),
        Patch(facecolor=BLUE, label="private: travel times / write route"),
        Patch(facecolor=GREY, hatch="////", edgecolor="white",
              label="call in an attempt later aborted"),
        Patch(facecolor=BAND, label="holding locks on requests"),
        Line2D([], [], color=INK, ls=":", label="blocked on a lock"),
        Line2D([], [], color=INK, marker="x", ls="", label="abort"),
    ], loc="upper center", ncol=3, fontsize=8, frameon=False,
        bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(f"REALM-Bench P3 {args.cell}, seed {args.seed} "
                 f"(claude-haiku-4-5): three vehicle dispatchers",
                 fontsize=10.5, x=0.01, ha="left", y=0.995, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
