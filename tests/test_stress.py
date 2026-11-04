"""Stress the central claim: at quiescence the run equals the sigma-serial run.

The protocol fixes the order at launch, so MTPO must not merely match *some*
serial order -- it must match the pre-decided one, under any interleaving.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coagent.runner import TrialConfig, reference_states, run_trial  # noqa: E402
from coagent.workloads import CELLS                                 # noqa: E402
from coagent.workloads.base import states_equal                     # noqa: E402

# (jitter, len_sigma): from a fully deterministic schedule out to one where
# per-step latency and per-agent task length both swing wildly.
REGIMES = [(0.0, 0.0), (0.3, 0.3), (0.6, 0.6), (0.9, 0.9), (1.2, 0.5)]
SEEDS = 25


class TestMTPOInvariantUnderNoise(unittest.TestCase):
    def test_live_state_equals_sigma_serial_state(self):
        checked = 0
        for jitter, len_sigma in REGIMES:
            cfg = TrialConfig(p_misjudge=0.0, p_task_failure=0.0,
                              jitter=jitter, len_sigma=len_sigma)
            for name, cell in CELLS.items():
                refs = reference_states(cell)
                for seed in range(SEEDS):
                    r = run_trial(cell, "MTPO", seed, cfg, refs)
                    checked += 1
                    self.assertTrue(
                        states_equal(r.final_state, refs[0]),
                        f"{name} jitter={jitter} len_sigma={len_sigma} "
                        f"seed={seed}: live state diverged from the "
                        f"sigma-serial state",
                    )
        self.assertGreater(checked, 1000)

    def test_no_pathological_outcomes_under_noise(self):
        for jitter, len_sigma in REGIMES:
            cfg = TrialConfig(p_misjudge=0.0, p_task_failure=0.0,
                              jitter=jitter, len_sigma=len_sigma)
            for name, cell in CELLS.items():
                refs = reference_states(cell)
                for seed in range(SEEDS):
                    r = run_trial(cell, "MTPO", seed, cfg, refs)
                    self.assertFalse(r.livelocked, f"{name}: livelock")
                    self.assertEqual(r.deadlocks, 0, f"{name}: deadlock")
                    self.assertEqual(r.aborts, 0, f"{name}: abort")


if __name__ == "__main__":
    unittest.main()
