"""Bounded exhaustive verification of certified chopping.

Mirrors Table 1 of idea.pdf: every soundness claim is checked against
enumerated executions rather than argued. The oracle enumerates all
interleavings of pieces (respecting program order within a workflow, each
piece atomic) and tests conflict-serializability at *workflow* granularity.
"""

import itertools
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coagent.chopping import (                     # noqa: E402
    Access, Decision, Mode, Piece, Plan, admit, certify,
    cycle_through_committed, guided_coarsen, seal,
)

R, W = Mode.READ, Mode.WRITE


# -- brute-force ground truth ----------------------------------------------
def interleavings(plan: Plan):
    """Every order of pieces that respects program order per workflow."""
    seqs = [plan.pieces[w] for w in plan.workflows]
    counts = [len(s) for s in seqs]
    total = sum(counts)

    def walk(idx, acc):
        if len(acc) == total:
            yield list(acc)
            return
        for i, s in enumerate(seqs):
            if idx[i] < len(s):
                idx[i] += 1
                acc.append(s[idx[i] - 1])
                yield from walk(idx, acc)
                acc.pop()
                idx[i] -= 1

    yield from walk([0] * len(seqs), [])


def serializable(order) -> bool:
    """Workflow-level conflict serializability of one interleaving."""
    edges = set()
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            a, b = order[i], order[j]
            if a.workflow == b.workflow:
                continue
            if a.conflicts(b):
                edges.add((a.workflow, b.workflow))
    nodes = {p.workflow for p in order}
    colour = {n: 0 for n in nodes}

    def cyclic(n):
        colour[n] = 1
        for (u, v) in edges:
            if u != n:
                continue
            if colour[v] == 1:
                return True
            if colour[v] == 0 and cyclic(v):
                return True
        colour[n] = 2
        return False

    return not any(colour[n] == 0 and cyclic(n) for n in list(nodes))


def all_interleavings_serializable(plan: Plan) -> bool:
    return all(serializable(o) for o in interleavings(plan))


def plan_from_spec(spec) -> Plan:
    """spec: {workflow: [[(item, mode), ...], ...]} -> Plan."""
    out = {}
    for w, pieces in spec.items():
        out[w] = [
            Piece(w, i, frozenset(Access.key(k, m) for k, m in acc))
            for i, acc in enumerate(pieces)
        ]
    return Plan(out)


class TestStaticCertification(unittest.TestCase):
    """Certified plans must be serializable under every interleaving."""

    def _sweep(self, n_workflows, n_pieces, items):
        """All read/write assignments of `items` across the pieces."""
        slots = n_workflows * n_pieces * len(items)
        checked = certified = 0
        for bits in itertools.product([None, R, W], repeat=slots):
            it = iter(bits)
            spec = {}
            for w in range(n_workflows):
                spec[f"W{w}"] = [
                    [(k, m) for k in items if (m := next(it)) is not None]
                    for _ in range(n_pieces)
                ]
            if any(not any(p for p in ps) for ps in spec.values()):
                continue
            plan = plan_from_spec(spec)
            ok, _ = certify(plan)
            checked += 1
            if ok:
                certified += 1
                self.assertTrue(
                    all_interleavings_serializable(plan),
                    f"certified an unsafe plan: {spec}",
                )
        return checked, certified

    def test_two_workflows_two_items(self):
        checked, certified = self._sweep(2, 2, ["x", "y"])
        self.assertGreater(checked, 200)
        self.assertGreater(certified, 0)

    def test_three_workflows_one_item(self):
        checked, certified = self._sweep(3, 2, ["x"])
        self.assertGreater(checked, 100)
        self.assertGreater(certified, 0)


class TestRepairIsSound(unittest.TestCase):
    """A rejected plan, once coarsened, must certify AND be serializable."""

    def test_guided_coarsening(self):
        rng = random.Random(7)
        repaired = 0
        for _ in range(300):
            spec = {
                f"W{w}": [
                    [(k, rng.choice([R, W]))
                     for k in ["x", "y"] if rng.random() < 0.6] or [("x", R)]
                    for _ in range(rng.randint(2, 3))
                ]
                for w in range(2)
            }
            plan = plan_from_spec(spec)
            if certify(plan)[0]:
                continue
            frontier = {w: 0 for w in plan.workflows}
            fixed, _ = guided_coarsen(plan, frontier)
            self.assertTrue(certify(fixed)[0], "coarsening did not certify")
            self.assertTrue(all_interleavings_serializable(fixed))
            repaired += 1
        self.assertGreater(repaired, 0)


class TestLemma2(unittest.TestCase):
    """A mixed cycle survives every refinement (non-removability).

    The frontier must be non-zero for this to say anything: at frontier 0
    sealing collapses each workflow to a single piece, which has no S edges
    and so always certifies. Committed pieces are what put S edges into the
    sealed plan.
    """

    def test_cycles_persist_under_refinement(self):
        rng = random.Random(11)
        tested = refinements = 0
        for _ in range(400):
            spec = {
                f"W{w}": [[(k, rng.choice([R, W]))
                           for k in ["x", "y"] if rng.random() < 0.7]
                          or [("x", R)]
                          for _ in range(rng.randint(2, 3))]
                for w in range(2)
            }
            plan = plan_from_spec(spec)
            frontier = {w: rng.randint(1, len(plan.pieces[w]) - 1)
                        for w in plan.workflows}
            sealed = seal(plan, frontier)
            if certify(sealed)[0]:
                continue          # only count genuinely sealed-and-failing
            tested += 1
            # every future-only coarsening is a refinement of the sealing,
            # so by Lemma 2 none of them may certify
            for cand in TestTheorem2._future_only_coarsenings(plan, frontier):
                refinements += 1
                self.assertFalse(
                    certify(cand)[0],
                    "a refinement of a cyclic sealing certified, "
                    "contradicting Lemma 2",
                )
        self.assertGreater(tested, 0, "no non-vacuous scenarios generated")
        self.assertGreater(refinements, tested)


class TestTheorem2(unittest.TestCase):
    """Admissible under future-only coarsening iff sealing certifies."""

    @staticmethod
    def _future_only_coarsenings(plan: Plan, frontier):
        """Every grouping of each workflow's suffix into contiguous runs."""
        per_w = {}
        for w, ps in plan.pieces.items():
            k = frontier.get(w, 0)
            tail = ps[k:]
            options = []
            n = len(tail)
            if n == 0:
                options.append(ps[:k])
            else:
                for cuts in itertools.product([0, 1], repeat=max(n - 1, 0)):
                    groups, cur = [], [tail[0]]
                    for i, c in enumerate(cuts):
                        if c:
                            groups.append(cur)
                            cur = []
                        cur.append(tail[i + 1])
                    groups.append(cur)
                    merged = [g[0].merged_with(g[1:], k + gi)
                              for gi, g in enumerate(groups)]
                    options.append(ps[:k] + merged)
            per_w[w] = options
        keys = list(per_w)
        for combo in itertools.product(*[per_w[w] for w in keys]):
            yield Plan(dict(zip(keys, combo)))

    def test_characterization(self):
        rng = random.Random(3)
        checked = 0
        for _ in range(250):
            spec = {
                f"W{w}": [[(k, rng.choice([R, W]))
                           for k in ["x", "y"] if rng.random() < 0.7]
                          or [("x", R)]
                          for _ in range(rng.randint(2, 3))]
                for w in range(2)
            }
            plan = plan_from_spec(spec)
            frontier = {w: rng.randint(0, len(plan.pieces[w]) - 1)
                        for w in plan.workflows}
            sealed_ok = certify(seal(plan, frontier))[0]
            exists = any(certify(c)[0] for c in
                         self._future_only_coarsenings(plan, frontier))
            self.assertEqual(
                sealed_ok, exists,
                "Theorem 2 violated: sealing says "
                f"{sealed_ok} but exhaustive search says {exists}",
            )
            checked += 1
        self.assertGreater(checked, 200)


class TestAdmissionSoundness(unittest.TestCase):
    """An admitted revision must be serializable, prefix never rewritten."""

    def test_admitted_plans_are_serializable(self):
        rng = random.Random(5)
        admitted = refused = 0
        for _ in range(300):
            spec = {
                f"W{w}": [[(k, rng.choice([R, W]))
                           for k in ["x", "y"] if rng.random() < 0.6]
                          or [("x", R)]
                          for _ in range(rng.randint(2, 3))]
                for w in range(2)
            }
            plan = plan_from_spec(spec)
            frontier = {w: rng.randint(0, len(plan.pieces[w]) - 1)
                        for w in plan.workflows}
            res = admit(plan, frontier)
            if not res.ok:
                refused += 1
                continue
            admitted += 1
            self.assertTrue(certify(res.plan)[0])
            self.assertTrue(all_interleavings_serializable(res.plan))
            for w in plan.workflows:            # committed prefix untouched
                k = frontier[w]
                self.assertEqual(
                    [p.footprint for p in res.plan.pieces[w][:k]],
                    [p.footprint for p in plan.pieces[w][:k]],
                    "admission rewrote a committed piece",
                )
        self.assertGreater(admitted, 0)
        self.assertGreater(refused, 0)


class TestTheorem3(unittest.TestCase):
    """Two individually-admissible revisions can be jointly inadmissible."""

    def test_admission_must_be_serialized(self):
        base = {
            "W0": [Piece("W0", 0, frozenset({Access.key("x", R)}))],
            "W1": [Piece("W1", 0, frozenset({Access.key("y", R)}))],
        }
        frontier = {"W0": 1, "W1": 1}
        only0 = Plan({
            "W0": base["W0"] + [Piece("W0", 1, frozenset({Access.key("y", W)}))],
            "W1": list(base["W1"]),
        })
        only1 = Plan({
            "W0": list(base["W0"]),
            "W1": base["W1"] + [Piece("W1", 1, frozenset({Access.key("x", W)}))],
        })
        both = Plan({
            "W0": base["W0"] + [Piece("W0", 1, frozenset({Access.key("y", W)}))],
            "W1": base["W1"] + [Piece("W1", 1, frozenset({Access.key("x", W)}))],
        })
        self.assertTrue(admit(only0, frontier).ok, "W0 alone should admit")
        self.assertTrue(admit(only1, frontier).ok, "W1 alone should admit")
        self.assertFalse(admit(both, frontier).ok,
                         "jointly they must be refused")
        self.assertFalse(all_interleavings_serializable(both))


class TestCorollary1(unittest.TestCase):
    """No future-only repair when the cycle runs through a committed piece."""

    def test_impossibility_witness(self):
        plan = Plan({
            "W0": [Piece("W0", 0, frozenset({Access.key("x", R)})),
                   Piece("W0", 1, frozenset({Access.key("y", W)}))],
            "W1": [Piece("W1", 0, frozenset({Access.key("x", W),
                                             Access.key("y", R)}))],
        })
        frontier = {"W0": 1, "W1": 1}
        res = admit(plan, frontier)
        self.assertIs(res.decision, Decision.REFUSED)
        self.assertTrue(cycle_through_committed(res.witness, frontier))
        self.assertFalse(all_interleavings_serializable(plan),
                         "the refused plan really does admit a bad execution")
        # waiting does not help: the committed workflow keeps its edges
        self.assertFalse(certify(seal(plan, frontier))[0])


class TestRangeFootprints(unittest.TestCase):
    """A write inside a scanned range must be visible to certification."""

    def test_phantom_is_caught(self):
        scan = Piece("W0", 0, frozenset({Access.prefix("d/", R)}))
        insert = Piece("W1", 0, frozenset({Access.key("d/new", W)}))
        self.assertTrue(scan.conflicts(insert), "phantom not detected")
        outside = Piece("W1", 0, frozenset({Access.key("e/new", W)}))
        self.assertFalse(scan.conflicts(outside), "false conflict")

    def test_enforcement_is_fail_closed(self):
        p = Piece("W0", 0, frozenset({Access.key("a", W),
                                      Access.prefix("d/", R)}))
        self.assertTrue(p.permits(Mode.WRITE, "a"))
        self.assertTrue(p.permits(Mode.READ, "d/x"))
        self.assertFalse(p.permits(Mode.WRITE, "d/x"),
                         "a read range must not authorise a write")
        self.assertFalse(p.permits(Mode.READ, "zzz"))


if __name__ == "__main__":
    unittest.main()
