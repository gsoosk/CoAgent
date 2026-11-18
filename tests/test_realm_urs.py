"""REALM-Bench P3 ride-sharing cells: the anomaly is real, the protocols stop it.

Skipped when REALM-Bench has not been fetched (scripts/fetch_realm_bench.sh):
its instances are not redistributed with this repository.
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
from coagent.workloads.realm_urs import (                      # noqa: E402
    P3_DIR, REALM_CELLS, consistent_dispatch, route_passengers,
)

CLEAN = TrialConfig(p_misjudge=0.0, p_task_failure=0.0)


@unittest.skipUnless(P3_DIR.exists(), "run scripts/fetch_realm_bench.sh")
class RealmP3Cells(unittest.TestCase):

    def test_three_cells_of_three_vehicles(self):
        self.assertEqual(sorted(REALM_CELLS),
                         ["realm-p3-001", "realm-p3-004", "realm-p3-008"])
        for cell in REALM_CELLS.values():
            self.assertEqual(len(cell.llm_spec["tasks"]), 3)

    def test_every_serial_order_is_consistent(self):
        for cell in REALM_CELLS.values():
            refs = reference_states(cell)
            self.assertEqual(len(refs), 6)          # 3! orders
            self.assertTrue(all(cell.anomaly_free(r, refs) for r in refs))

    def test_uncoordinated_dispatch_double_books(self):
        # the anomaly is not hypothetical: nearest-first dispatchers collide
        for cell in REALM_CELLS.values():
            refs = reference_states(cell)
            bad = [not cell.anomaly_free(
                run_trial(cell, "naive", s, CLEAN, refs=refs).final_state, refs)
                for s in range(5)]
            self.assertTrue(all(bad), cell.name)

    def test_protocols_keep_dispatch_consistent(self):
        for cell in REALM_CELLS.values():
            refs = reference_states(cell)
            for proto in ("serial", "2PL-saga", "MTPO", "chopping",
                          "chopping-prio"):
                for s in range(5):
                    r = run_trial(cell, proto, s, CLEAN, refs=refs)
                    self.assertTrue(cell.anomaly_free(r.final_state, refs),
                                    f"{cell.name} {proto} seed {s}")

    def test_certification_cuts_routing_free_of_the_claim(self):
        cell = REALM_CELLS["realm-p3-001"]
        agents = cell.build_agents(JudgmentOracle(random.Random(0), 0.0),
                                   random.Random(0))
        proto = PROTOCOLS["chopping-prio"]()
        rt = Runtime(cell.fresh_target(), agents, proto, random.Random(0))
        proto._ensure_plan(rt)
        for pieces in proto.plan.pieces.values():
            # list + claim merged and held; travel times and route cut free
            self.assertEqual([p.span for p in pieces], [2, 1, 1])

    def test_route_parsing_tolerates_model_output(self):
        route = {"stops": '[{"passenger": "urs/request/p001", "action": '
                          '"pickup"}, {"passenger": "p001", "action": '
                          '"dropoff"}]'}
        self.assertEqual(route_passengers(route), ["p001"])
        final = {"urs/request/p001": {"assigned": "v002"},
                 "urs/route/v001": route}
        self.assertFalse(consistent_dispatch(final, ["v001", "v002"]))
        # routing a passenger nobody claimed is a task error, not isolation
        final["urs/request/p001"] = {"assigned": None}
        self.assertTrue(consistent_dispatch(final, ["v001", "v002"]))
        final["urs/route/v002"] = route
        self.assertFalse(consistent_dispatch(final, ["v001", "v002"]))


if __name__ == "__main__":
    unittest.main()
