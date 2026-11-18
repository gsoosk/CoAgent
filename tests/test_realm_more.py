"""REALM-Bench J1 (job-shop) and P5 (wedding) cells.

Skipped when REALM-Bench has not been fetched (scripts/fetch_realm_bench.sh).
"""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coagent.agents.scripted import JudgmentOracle             # noqa: E402
from coagent.protocols import PROTOCOLS                        # noqa: E402
from coagent.runner import TrialConfig, reference_states, run_trial  # noqa: E402
from coagent.runtime import Runtime                            # noqa: E402
from coagent.workloads.realm_jssp import (                     # noqa: E402
    J1, JSSP_CELLS, load_cut, optimal_makespan,
)
from coagent.workloads.realm_wedding import (                  # noqa: E402
    P5_FILE, WEDDING_CELLS, double_bookings,
)

CLEAN = TrialConfig(p_misjudge=0.0, p_task_failure=0.0)
PROTOS = ("serial", "2PL-saga", "MTPO", "chopping", "chopping-prio")


def spans(cell):
    agents = cell.build_agents(JudgmentOracle(random.Random(0), 0.0),
                               random.Random(0))
    proto = PROTOCOLS["chopping-prio"]()
    rt = Runtime(cell.fresh_target(), agents, proto, random.Random(0))
    proto._ensure_plan(rt)
    return [[p.span for p in ps] for ps in proto.plan.pieces.values()]


class _Common:
    cells = {}

    def test_serial_orders_are_consistent(self):
        for cell in self.cells.values():
            refs = reference_states(cell)
            self.assertTrue(all(cell.anomaly_free(r, refs) for r in refs))
            self.assertEqual(cell.quality(refs[0])["goal_satisfaction"], 100.0)

    def test_uncoordinated_planners_double_book(self):
        for cell in self.cells.values():
            refs = reference_states(cell)
            bad = sum(not cell.anomaly_free(
                run_trial(cell, "naive", s, CLEAN, refs=refs).final_state, refs)
                for s in range(10))
            self.assertGreaterEqual(bad, 5, cell.name)

    def test_protocols_prevent_double_booking(self):
        for cell in self.cells.values():
            refs = reference_states(cell)
            for proto in PROTOS:
                for s in range(4):
                    r = run_trial(cell, proto, s, CLEAN, refs=refs)
                    self.assertTrue(cell.anomaly_free(r.final_state, refs),
                                    f"{cell.name} {proto} seed {s}")


@unittest.skipUnless(J1.exists(), "run scripts/fetch_realm_bench.sh")
class JobShop(_Common, unittest.TestCase):
    cells = JSSP_CELLS

    def test_cuts_and_optima(self):
        self.assertEqual(len(JSSP_CELLS), 3)
        for name, opt in (("abz07", 91), ("ta01", 218),
                          ("dmu-cscmax-20-15-1", 457)):
            jobs = load_cut(name)
            self.assertEqual([sorted(m for m, _ in j) for j in jobs],
                             [[0, 1, 2]] * 3)
            self.assertEqual(optimal_makespan(jobs), opt)

    def test_every_job_is_one_piece(self):
        # every job needs every machine: certification merges it whole
        self.assertEqual(spans(JSSP_CELLS["realm-jssp-abz07"]), [[7]] * 3)


@unittest.skipUnless(P5_FILE.exists(), "run scripts/fetch_realm_bench.sh")
class Wedding(_Common, unittest.TestCase):
    cells = WEDDING_CELLS

    def test_booking_is_cut_free_of_planning(self):
        self.assertEqual(spans(WEDDING_CELLS["realm-p5-wedding"]),
                         [[2, 1, 1]] * 3)

    def test_overlap_fails_the_errand(self):
        cell = WEDDING_CELLS["realm-p5-wedding"]
        final = cell.fresh_target().live
        final["wedding/car/pat"] = {"available_from": "12:00",
                                    "alex": ["12:00", "13:20"],
                                    "jamie": ["12:30", "13:50"]}
        for e, s in (("alex", "12:00"), ("jamie", "12:30")):
            final[f"wedding/plan/{e}"] = {"car": "pat", "start": s,
                                          "stops": ["B"]}
        self.assertEqual(double_bookings(final), 1)
        self.assertEqual(cell.quality(final)["goal_satisfaction"], 0.0)


if __name__ == "__main__":
    unittest.main()
