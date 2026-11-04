#!/usr/bin/env python3
"""Run the ten contended cells under the five protocols (paper section 7.2).

For every (cell, protocol) pair we run N trials and report correctness, mean
wall-clock relative to serial, and mean token cost relative to serial.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coagent.protocols import PROTOCOL_ORDER                  # noqa: E402
from coagent.runner import TrialConfig, reference_states, run_trial  # noqa: E402
from coagent.workloads import CELLS                           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10,
                    help="N per (cell, protocol); the paper uses 10")
    ap.add_argument("--p-misjudge", type=float, default=0.05,
                    help="A3 gap: chance a notified agent misreads relevance")
    ap.add_argument("--p-task-failure", type=float, default=0.010,
                    help="A1 gap: chance an agent fails on its own")
    ap.add_argument("--retry-cap", type=int, default=5)
    ap.add_argument("--jitter", type=float, default=0.40)
    ap.add_argument("--len-sigma", type=float, default=0.40,
                    help="spread of per-agent task length within a pair")
    ap.add_argument("--redo-truncate", type=float, default=0.0,
                    help="chance an aborted agent's redo truncates early; "
                         "0 = perfect replay (the paper's strong agent), "
                         "0.22 = measured on claude-haiku-4-5")
    ap.add_argument("--out", default=str(ROOT / "results" / "matrix.csv"))
    args = ap.parse_args()

    cfg = TrialConfig(
        p_misjudge=args.p_misjudge,
        p_task_failure=args.p_task_failure,
        retry_cap=args.retry_cap,
        jitter=args.jitter,
        len_sigma=args.len_sigma,
        p_redo_truncate=args.redo_truncate,
    )
    rows = []
    for ci, (cell_name, cell) in enumerate(CELLS.items()):
        refs = reference_states(cell)
        for proto in PROTOCOL_ORDER:
            for seed in range(args.trials):
                # Offset per cell: reusing one seed set across cells would
                # correlate the trials and collapse the effective sample.
                # Alternate the launch order. Sigma is a free choice of
                # the task layer, and pinning it to [0, 1] both hides agents
                # that cannot heal and makes the same agent the deadlock
                # victim in every single trial.
                order = [0, 1] if seed % 2 == 0 else [1, 0]
                r = run_trial(
                    cell, proto, seed=1000 + 7919 * ci + seed,
                    cfg=cfg, refs=refs, order=order,
                )
                rows.append(r)

    os.makedirs(Path(args.out).parent, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "suite", "protocol", "seed", "invariant_ok",
                    "wall_clock", "tokens", "deadlocks", "aborts",
                    "notifications", "repairs", "undos", "livelocked",
                    "trajectory_match"])
        for r in rows:
            w.writerow([r.cell, CELLS[r.cell].suite, r.protocol, r.seed,
                        int(r.invariant_ok), f"{r.wall_clock:.3f}", r.tokens,
                        r.deadlocks, r.aborts, r.notifications, r.repairs,
                        r.undos, int(r.livelocked),
                        int(r.trajectory_match)])

    # per-protocol aggregates, normalized to serial per cell
    by = defaultdict(list)
    for r in rows:
        by[(r.cell, r.protocol)].append(r)
    serial_wall = {c: statistics.mean(x.wall_clock for x in by[(c, "serial")])
                   for c in CELLS}
    serial_tok = {c: statistics.mean(x.tokens for x in by[(c, "serial")])
                  for c in CELLS}

    paper = {
        "serial":   (98, 1.00, 1.00),
        "naive":    (13, 1.54, 1.06),
        "2PL-saga": (96, 1.04, 1.45),
        "OCC-saga": (93, 0.93, 1.83),
        "MTPO":     (93, 1.43, 1.15),
    }
    print(f"N={args.trials} trials x {len(CELLS)} cells "
          f"= {args.trials * len(CELLS)} runs per protocol\n")
    head = (f"{'protocol':10}{'correct%':>10}{'speedup':>9}{'cost':>7}"
            f"{'dl/trial':>10}{'ab/trial':>10}   paper (correct/speed/cost)")
    print(head)
    print("-" * len(head))
    summary = {}
    for proto in PROTOCOL_ORDER:
        all_r = [r for c in CELLS for r in by[(c, proto)]]
        corr = 100.0 * sum(r.invariant_ok for r in all_r) / len(all_r)
        # Ratio of means, not mean of ratios: 1/x is convex, so averaging
        # per-trial ratios biases speedup upward.
        sp = statistics.mean(serial_wall[r.cell] for r in all_r) / statistics.mean(
            r.wall_clock for r in all_r
        )
        cost = statistics.mean(r.tokens for r in all_r) / statistics.mean(
            serial_tok[r.cell] for r in all_r
        )
        dl = statistics.mean(r.deadlocks for r in all_r)
        ab = statistics.mean(r.aborts for r in all_r)
        p = paper[proto]
        print(f"{proto:10}{corr:9.0f}%{sp:9.2f}{cost:7.2f}{dl:10.2f}{ab:10.2f}"
              f"   {p[0]:>3}% / {p[1]:.2f} / {p[2]:.2f}")
        summary[proto] = (corr, sp, cost, dl, ab)

    # Only MTPO claims this: at quiescence the live copy equals the
    # materialization of each object's trajectory. The other protocols make
    # no such promise, so checking them against it is meaningless.
    mtpo_rows = [r for r in rows if r.protocol == "MTPO"]
    bad_traj = [r for r in mtpo_rows if not r.trajectory_match]
    print(f"\nMTPO: live state == sigma-serial state in "
          f"{len(mtpo_rows) - len(bad_traj)}/{len(mtpo_rows)} trials")
    print(f"wrote {args.out}  ({len(rows)} trials)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
