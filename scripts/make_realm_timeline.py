#!/usr/bin/env python3
"""One REALM trial, drawn per protocol: who ran when, and who waited.

Each lane is one agent on the virtual clock. Bars are model calls, coloured
by phase: the shared step (reading and claiming or booking the contended
resource) or the agent's private work. Hatched grey bars are calls in an
attempt that was later aborted. A dotted line is time spent blocked on a
lock. The pale band behind a lane is when that agent holds locks on the
shared resource: to the end of the workflow under 2PL, only through the
shared step under certified chopping.

    python scripts/make_realm_timeline.py --problem realm            # P3
    python scripts/make_realm_timeline.py --problem realm_jssp --cell \
        realm-jssp-abz07 --seed 1

Without --cell/--seed the trial is picked by rule: one where serial, 2PL-saga
and chopping-prio all stayed consistent and whose speed ratios are closest
to their medians over the problem, i.e. a representative trial.
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
# per problem: shared tools, legend wording, title
PROBLEMS = {
    "realm": (("list_requests", "assign_request", "get_object"),
              "list / claim ride requests", "travel times / write route",
              "requests", "P3 ride-sharing: three vehicle dispatchers"),
    "realm_wedding": (("list_cars", "book_car", "cancel_car"),
                      "list / book cars", "travel times / write plan",
                      "cars", "P5 wedding: three errand planners"),
    "realm_jssp": (("get_machine", "book_slot", "cancel_slot"),
                   "read / book machines", "write schedule", "machines",
                   "J1 job-shop: three job schedulers"),
    "realm_tours": (("list_guides", "book_guide", "cancel_guide"),
                    "list / book guides", "travel times / write plan",
                    "guides", "P2 campus tours: three group planners"),
    "realm_thanks": (("list_people", "book_person", "cancel_person"),
                     "list / book people", "write plan", "people",
                     "P6/P9 Thanksgiving: three task planners"),
}
SHARED: tuple = PROBLEMS["realm"][0]


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


def draw(ax, r, proto, serial_wall, names) -> None:
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
    ax.set_yticklabels([names[int(a.split("-")[1])] for a in
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


def pick(rows):
    """A representative trial: consistent under serial, 2PL-saga and
    chopping-prio, speed ratios nearest their medians over the problem."""
    import statistics as st
    by = {(r["cell"], r["seed"], r["protocol"]): r for r in rows}
    keys = sorted({(c, s) for c, s, p in by if p == "serial"})
    ratio = {p: {k: by[(*k, "serial")]["wall"] / by[(*k, p)]["wall"]
                 for k in keys if (*k, p) in by}
             for p in ("2PL-saga", "chopping-prio")}
    med = {p: st.median(v.values()) for p, v in ratio.items()}
    ok = [k for k in keys
          if all((*k, p) in by and by[(*k, p)]["anomaly_free"]
                 for p in ("serial", "2PL-saga", "chopping-prio"))]
    pool = ok or [k for k in keys if all(k in ratio[p] for p in ratio)]
    return min(pool, key=lambda k: sum(abs(ratio[p][k] - med[p])
                                       for p in ratio))


def main() -> int:
    global SHARED
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="realm", choices=list(PROBLEMS))
    ap.add_argument("--cell", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    SHARED, shared_lbl, private_lbl, what, title = PROBLEMS[args.problem]
    parts = ROOT / "results" / args.problem / "parts"
    rows = [json.loads(l) for p in sorted(parts.glob("*.jsonl"))
            for l in p.read_text().splitlines() if l.strip()]
    cell, seed = ((args.cell, args.seed) if args.cell and args.seed is not None
                  else pick(rows))
    by = {r["protocol"]: r for r in rows
          if r["cell"] == cell and r["seed"] == seed}

    sys.path.insert(0, str(ROOT / "src"))
    import random as _r
    from coagent.agents.scripted import JudgmentOracle
    from coagent.workloads.realm_jssp import JSSP_CELLS
    from coagent.workloads.realm_thanksgiving import THANKS_CELLS
    from coagent.workloads.realm_tours import TOURS_CELLS
    from coagent.workloads.realm_urs import REALM_CELLS
    from coagent.workloads.realm_wedding import WEDDING_CELLS
    cells ={**REALM_CELLS, **JSSP_CELLS, **WEDDING_CELLS, **TOURS_CELLS,
             **THANKS_CELLS}
    names = [a.name for a in cells[cell].build_agents(
        JudgmentOracle(_r.Random(0), 0.0), _r.Random(0))]

    fig, axes = plt.subplots(len(PROTOS), 1, figsize=(11, 8.2), sharex=True)
    for ax, p in zip(axes, PROTOS):
        draw(ax, by[p], p, by["serial"]["wall"], names)
    axes[-1].set_xlabel("virtual time (s): each call lasts its measured "
                        "model latency", fontsize=8.5, color=MUTED)
    fig.legend(handles=[
        Patch(facecolor=ORANGE, label=f"shared: {shared_lbl}"),
        Patch(facecolor=BLUE, label=f"private: {private_lbl}"),
        Patch(facecolor=GREY, hatch="////", edgecolor="white",
              label="call in an attempt later aborted"),
        Patch(facecolor=BAND, label=f"holding locks on {what}"),
        Line2D([], [], color=INK, ls=":", label="blocked on a lock"),
        Line2D([], [], color=INK, marker="x", ls="", label="abort"),
    ], loc="upper center", ncol=3, fontsize=8, frameon=False,
        bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(f"REALM-Bench {title} ({cell}, seed {seed}, "
                 f"claude-haiku-4-5)",
                 fontsize=10.5, x=0.01, ha="left", y=0.995, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = Path(args.out or ROOT / "results" / args.problem / "figures"
               / f"fig_{args.problem}_timeline.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"wrote {out} ({cell}, seed {seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
