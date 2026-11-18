"""REALM-Bench P2 (campus tours) and P6/P9 (Thanksgiving) cells.

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
from coagent.workloads.realm_thanksgiving import (             # noqa: E402
    P6_FILE, THANKS_CELLS, conflicts, load,
)
from coagent.workloads.realm_tours import P2_DIR, TOURS_CELLS  # noqa: E402

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

    def test_protocols_keep_bookings_consistent(self):
        for cell in self.cells.values():
            refs = reference_states(cell)
            for proto in PROTOS:
                for s in range(3):
                    r = run_trial(cell, proto, s, CLEAN, refs=refs)
                    self.assertTrue(cell.anomaly_free(r.final_state, refs),
                                    f"{cell.name} {proto} seed {s}")


@unittest.skipUnless(P2_DIR.exists(), "run scripts/fetch_realm_bench.sh")
class CampusTours(_Common, unittest.TestCase):
    cells = TOURS_CELLS

    def test_uncoordinated_planners_double_book(self):
        for cell in TOURS_CELLS.values():
            refs = reference_states(cell)
            bad = sum(not cell.anomaly_free(
                run_trial(cell, "naive", s, CLEAN, refs=refs).final_state, refs)
                for s in range(6))
            self.assertGreaterEqual(bad, 3, cell.name)

    def test_booking_is_cut_free_of_routing(self):
        self.assertEqual(spans(TOURS_CELLS["realm-p2-tours-001"]),
                         [[2, 1, 1]] * 3)


@unittest.skipUnless(P6_FILE.exists(), "run scripts/fetch_realm_bench.sh")
class Thanksgiving(_Common, unittest.TestCase):
    cells = THANKS_CELLS

    def test_p9_delays_come_from_the_instances(self):
        self.assertEqual(load("p6"), {"dad": 14 * 60, "sister": 15 * 60 + 30})
        self.assertEqual(load("p9-001"), {"dad": 15 * 60 + 29,
                                          "sister": 15 * 60 + 44})

    def test_home_bookings_share_a_person_driving_does_not(self):
        final = {"thanks/person/mom": {"turkey": ["14:00", "18:00", "home"],
                                       "sides": ["16:00", "18:00", "home"]}}
        self.assertEqual(conflicts(final), [])
        final["thanks/person/mom"]["sister"] = ["13:30", "16:30", "drive"]
        self.assertEqual(len(conflicts(final)), 2)


if __name__ == "__main__":
    unittest.main()
