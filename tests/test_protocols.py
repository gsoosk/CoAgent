"""Protocol-level tests: the claims MTPO actually makes."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coagent.protocols import PROTOCOL_ORDER                       # noqa: E402
from coagent.runner import TrialConfig, reference_states, run_trial  # noqa: E402
from coagent.workloads import CELLS, CONTENDED                     # noqa: E402
from coagent.workloads.base import states_equal                    # noqa: E402
from coagent.workloads.micro import halving_pair                   # noqa: E402

CLEAN = TrialConfig(p_misjudge=0.0, p_task_failure=0.0, jitter=0.0,
                    len_sigma=0.0)
NOISY = TrialConfig(p_misjudge=0.0, p_task_failure=0.0, jitter=0.5,
                    len_sigma=0.5)
SEEDS = range(12)


class TestMTPOSerializability(unittest.TestCase):
    """At GlobalQuiet the run must equal the sigma-serial execution."""

    def test_mtpo_matches_a_serial_order_on_every_cell(self):
        for name, cell in CELLS.items():
            refs = reference_states(cell)
            for seed in SEEDS:
                with self.subTest(cell=name, seed=seed):
                    r = run_trial(cell, "MTPO", seed, NOISY, refs)
                    self.assertTrue(
                        r.invariant_ok,
                        f"{name}/seed{seed}: {r.final_state} matched no "
                        f"serial order",
                    )

    def test_mtpo_matches_the_preorder_serial_order_specifically(self):
        """MTPO fixes the order at launch, so it must land on A . B."""
        for name, cell in CELLS.items():
            refs = reference_states(cell)
            for seed in SEEDS:
                with self.subTest(cell=name, seed=seed):
                    r = run_trial(cell, "MTPO", seed, NOISY, refs)
                    self.assertTrue(
                        states_equal(r.final_state, refs[0]),
                        f"{name}/seed{seed}: not the pre-decided order",
                    )

    def test_no_deadlocks_or_aborts_by_construction(self):
        for name, cell in CELLS.items():
            refs = reference_states(cell)
            for seed in SEEDS:
                r = run_trial(cell, "MTPO", seed, NOISY, refs)
                self.assertEqual(r.deadlocks, 0, f"{name}: MTPO deadlocked")
                self.assertEqual(r.aborts, 0, f"{name}: MTPO aborted")
                self.assertFalse(r.livelocked, f"{name}: MTPO livelocked")


class TestCellsAreContended(unittest.TestCase):
    def test_naive_violates_every_cell(self):
        """If uncoordinated execution passes, the cell is not contended."""
        for name, cell in CONTENDED.items():
            refs = reference_states(cell)
            r = run_trial(cell, "naive", 3, CLEAN, refs)
            self.assertFalse(r.invariant_ok, f"{name} is not contended")

    def test_serial_always_passes(self):
        for name, cell in CELLS.items():
            refs = reference_states(cell)
            for seed in SEEDS:
                r = run_trial(cell, "serial", seed, NOISY, refs)
                self.assertTrue(r.invariant_ok, f"{name} serial failed")


class TestClassicalProtocols(unittest.TestCase):
    def test_2pl_deadlocks_under_contention(self):
        total = sum(
            run_trial(c, "2PL-saga", s, CLEAN, reference_states(c)).deadlocks
            for c in CELLS.values() for s in range(4)
        )
        self.assertGreater(total, 0, "2PL never deadlocked on contended cells")

    def test_occ_aborts_under_contention(self):
        total = sum(
            run_trial(c, "OCC-saga", s, CLEAN, reference_states(c)).aborts
            for c in CELLS.values() for s in range(4)
        )
        self.assertGreater(total, 0, "OCC never aborted on contended cells")

    def test_classical_protocols_are_still_correct(self):
        for name, cell in CELLS.items():
            refs = reference_states(cell)
            for proto in ("2PL-saga", "OCC-saga"):
                r = run_trial(cell, proto, 5, CLEAN, refs)
                self.assertTrue(r.invariant_ok, f"{name}/{proto} violated")


class TestPreOrderIsLoadBearing(unittest.TestCase):
    """Section 5.2: notification without a pre-order is not enough."""

    def test_broadcast_livelocks_where_mtpo_converges(self):
        cell = halving_pair()
        refs = reference_states(cell)
        bc = run_trial(cell, "broadcast", 1, CLEAN, refs)
        mt = run_trial(cell, "MTPO", 1, CLEAN, refs)
        self.assertFalse(bc.invariant_ok, "broadcast should not serialize")
        self.assertGreater(bc.repairs, 100, "broadcast should thrash")
        self.assertTrue(mt.invariant_ok)
        self.assertLessEqual(mt.repairs, 2, "MTPO should converge in one heal")


class TestSelfHealingAssumption(unittest.TestCase):
    """A3 is an assumption, not a guarantee: correctness tracks it."""

    def test_correctness_degrades_with_misjudgment(self):
        cell = CELLS["k8s-canary"]
        refs = reference_states(cell)
        scores = []
        for pm in (0.0, 0.5, 1.0):
            cfg = TrialConfig(p_misjudge=pm, p_task_failure=0.0, jitter=0.3)
            ok = sum(
                run_trial(cell, "MTPO", 2000 + s, cfg, refs).invariant_ok
                for s in range(20)
            )
            scores.append(ok)
        self.assertEqual(scores[0], 20, "no misjudgment should be perfect")
        self.assertEqual(scores[2], 0, "always misjudging should always fail")
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])


if __name__ == "__main__":
    unittest.main()


class TestBaselinesAreNotStrawmanned(unittest.TestCase):
    """On a disjoint pair every protocol must recover full concurrency."""

    def test_no_overhead_without_contention(self):
        from coagent.workloads.control import disjoint_pair

        cell = disjoint_pair()
        refs = reference_states(cell)
        cfg = TrialConfig(p_misjudge=0.05, p_task_failure=0.0, jitter=0.3,
                          len_sigma=0.0)
        runs = {
            p: [run_trial(cell, p, s, cfg, refs) for s in range(15)]
            for p in PROTOCOL_ORDER
        }
        base = sum(r.wall_clock for r in runs["serial"]) / 15
        naive_sp = base / (sum(r.wall_clock for r in runs["naive"]) / 15)
        for proto in ("2PL-saga", "OCC-saga", "MTPO"):
            rs = runs[proto]
            self.assertTrue(all(r.invariant_ok for r in rs), proto)
            self.assertEqual(sum(r.deadlocks for r in rs), 0,
                             f"{proto} deadlocked without contention")
            self.assertEqual(sum(r.aborts for r in rs), 0,
                             f"{proto} aborted without contention")
            sp = base / (sum(r.wall_clock for r in rs) / 15)
            self.assertAlmostEqual(
                sp, naive_sp, delta=0.05,
                msg=f"{proto} lost concurrency on a disjoint pair",
            )
            tok = sum(r.tokens for r in rs)
            self.assertLessEqual(
                tok / sum(r.tokens for r in runs["serial"]), 1.02,
                f"{proto} added token cost without contention",
            )


class TestLossyRedo(unittest.TestCase):
    """A cleared context makes a redo a different, worse execution.

    Measured on claude-haiku-4-5 at ~15 points of lost task completeness. The
    cost is not a property of "aborting" in general -- it scales with how much
    work the victim had already done:

      OCC victimises the sweep agent, because its broad range read is what a
      concurrent insert invalidates. That agent has several writes, so a
      truncated redo loses most of them.

      2PL victimises the higher-sigma agent, which in these cells is the short
      one-write partner. Truncating after its first write removes nothing.

      MTPO never aborts, so there is no redo to lose.

    The real Haiku run showed exactly this split: 2PL aborted the canary agent
    10/10 and stayed 10/10 correct; OCC aborted the sweep agent 10/10 and fell
    to 5/10.
    """

    def _correct(self, proto, p_trunc, seeds=8):
        cfg = TrialConfig(p_misjudge=0.0, p_task_failure=0.0, jitter=0.4,
                          len_sigma=0.4, p_redo_truncate=p_trunc)
        ok = n = 0
        for name, cell in CELLS.items():
            refs = reference_states(cell)
            for seed in range(seeds):
                ok += run_trial(cell, proto, seed, cfg, refs).invariant_ok
                n += 1
        return ok / n

    def test_occ_degrades_because_it_victimises_the_sweep(self):
        clean = self._correct("OCC-saga", 0.0)
        lossy = self._correct("OCC-saga", 0.5)
        self.assertLess(
            lossy, clean,
            f"OCC should lose correctness when its redo is lossy "
            f"(clean={clean:.2f}, lossy={lossy:.2f})",
        )

    def test_mtpo_is_unaffected(self):
        self.assertEqual(
            self._correct("MTPO", 0.0), self._correct("MTPO", 0.5),
            "MTPO never aborts, so a lossy redo must cost it nothing",
        )

    def test_2pl_is_unaffected_because_its_victim_is_cheap(self):
        """Not a property of 2PL -- a property of which agent it picks."""
        self.assertEqual(
            self._correct("2PL-saga", 0.0), self._correct("2PL-saga", 0.5),
            "2PL's victim here has a single write, so truncation removes "
            "nothing; this would not hold if the victim were the sweep",
        )
