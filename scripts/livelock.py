#!/usr/bin/env python3
"""Section 5.2: unconditional broadcast livelocks; MTPO's pre-order does not."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coagent.runner import TrialConfig, reference_states, run_trial  # noqa: E402
from coagent.workloads.micro import halving_pair                    # noqa: E402


def main() -> int:
    cell = halving_pair()
    refs = reference_states(cell)
    print("task pair:  x <- y/2  ||  y <- x/2     (initial x = y = 1)")
    print("serial outcomes:",
          ", ".join(f"(x={r['x']['v']}, y={r['y']['v']})" for r in refs))
    print()
    cfg = TrialConfig(p_misjudge=0.0, p_task_failure=0.0, jitter=0.0,
                      len_sigma=0.0, time_scale=1.0)
    print(f"{'protocol':12}{'serializable':>14}{'notifications':>15}"
          f"{'repairs':>9}{'final (x, y)':>20}")
    print("-" * 72)
    for proto in ("broadcast", "MTPO"):
        r = run_trial(cell, proto, seed=1, cfg=cfg, refs=refs)
        final = (f"({r.final_state['x']['v']:.6g}, "
                 f"{r.final_state['y']['v']:.6g})")
        ok = "yes" if r.invariant_ok else "NO"
        print(f"{proto:12}{ok:>14}{r.notifications:>15}{r.repairs:>9}"
              f"{final:>20}")
    print()
    print(
        "Broadcast closes the two rw edges into a cycle: every write "
        "invalidates the\npeer's premise and every heal issues another write, "
        "so x and y walk to zero\n(Figure 2). It only stops because floats "
        "underflow, and the end state matches\nneither serial order. MTPO "
        "orients notifications low-to-high: the higher rank\nhalves once "
        "more and the pair converges on the serial outcome."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
