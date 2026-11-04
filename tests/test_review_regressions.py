"""Regressions for defects found in an adversarial review of the protocol.

Each test reproduces a concrete failure the implementation used to exhibit.
They are kept separate from the behavioural suite because they encode bugs,
not paper claims.
"""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coagent.agents.scripted import (                    # noqa: E402
    JudgmentOracle, PlanStep, ScriptedAgent,
)
from coagent.core import (                               # noqa: E402
    TargetSystem, Tool, WriteClass, blind_set, covered_by, is_descendant,
    rmw_append,
)
from coagent.protocols import PROTOCOLS                  # noqa: E402
from coagent.runtime import CallKind, Runtime, Status    # noqa: E402
from coagent.workloads.base import Cell, normalize       # noqa: E402
from coagent.runner import (                             # noqa: E402
    TrialConfig, reference_states, run_trial,
)
from coagent.workloads import CELLS                      # noqa: E402

CLEAN = TrialConfig(p_misjudge=0.0, p_task_failure=0.0, jitter=0.0,
                    len_sigma=0.0)


def _field_tool(name="set_field"):
    def _exec(v, a):
        out = dict(v or {})
        out[a["field"]] = a["value"]
        return out

    def _rev(cur, cap, a):
        out = dict(cur or {})
        f = a["field"]
        if isinstance(cap, dict) and f in cap:
            out[f] = cap[f]
        else:
            out.pop(f, None)
        return out

    return Tool(name=name, wclass=WriteClass.BLIND, exec_fn=_exec,
                reverse_fn=_rev, covers_fn=lambda a: frozenset({a["field"]}))


def _run(cell, protocol, seed=0, order=None):
    refs = reference_states(cell)
    return run_trial(cell, protocol, seed, CLEAN, refs, order=order)


class TestThomasRuleIsCoverageAware(unittest.TestCase):
    """A blind write must only shadow a write it actually covers."""

    def test_write_to_another_field_is_not_shadowed(self):
        tool = _field_tool()

        def build(oracle, rng, **kw):
            a = ScriptedAgent(
                name="A", oracle=oracle, rng=rng,
                plan=[
                    PlanStep(kind=CallKind.THINK, duration=5.0, tokens=100),
                    PlanStep(kind=CallKind.WRITE, obj="d/app", tool=tool,
                             args={"field": "replicas", "value": 7},
                             writes=("d/app",), duration=1.0, tokens=100),
                ],
            )
            b = ScriptedAgent(
                name="B", oracle=oracle, rng=rng,
                plan=[
                    PlanStep(kind=CallKind.WRITE, obj="d/app", tool=tool,
                             args={"field": "image", "value": "v2"},
                             writes=("d/app",), duration=1.0, tokens=100),
                ],
            )
            return [a, b]

        cell = Cell(name="two-fields", suite="regress",
                    anomaly="late write to a different field",
                    initial={"d/app": {"image": "v1", "replicas": 1}},
                    build_agents=build)
        r = _run(cell, "MTPO")
        self.assertTrue(r.trajectory_match,
                        "a blind write to `image` swallowed a write to "
                        "`replicas`")
        self.assertEqual(r.final_state["d/app"],
                         {"image": "v2", "replicas": 7})

    def test_same_field_still_shadows(self):
        tool = _field_tool()
        a = tool.covers({"field": "x"})
        b = tool.covers({"field": "x"})
        self.assertTrue(tool.shadows(tool, {"field": "x"}, {"field": "x"}))
        self.assertFalse(tool.shadows(tool, {"field": "x"}, {"field": "y"}))
        self.assertEqual(a, b)


class TestPremiseRespectsReadPosition(unittest.TestCase):
    """A refreshed premise must match what the reader's read would return."""

    def test_own_writes_before_the_read_are_kept(self):
        traj = TargetSystem({"log": []}).traj("log")
        t = rmw_append()
        from coagent.core import WriteOp
        traj.insert(WriteOp(obj="log", tool=t, args={"item": "B1"},
                            agent="B", sigma=1, step=1))
        traj.insert(WriteOp(obj="log", tool=t, args={"item": "A1"},
                            agent="A", sigma=0, step=1))
        # B read at step 2, after its own append at step 1
        self.assertEqual(traj.materialize_premise(1, "B", 2), ["A1", "B1"])
        # B read at step 0, before its own append
        self.assertEqual(traj.materialize_premise(1, "B", 0), ["A1"])


class TestFieldScopedUndo(unittest.TestCase):
    """A saga undo must not revert a concurrent write to another field."""

    def test_undo_preserves_peer_field(self):
        from coagent.core import WriteOp
        ts = TargetSystem({"o": {"a": 1, "b": 2}})
        t = blind_set()
        w = WriteOp(obj="o", tool=t, args={"field": "a", "value": 9},
                    agent="A", sigma=0, step=0)
        ts.apply_live(w)
        ts.live["o"]["b"] = 99          # a peer commits to another field
        ts.undo_live(w)
        self.assertEqual(ts.live["o"], {"a": 1, "b": 99})


class TestNoPhantomObjects(unittest.TestCase):
    """Reading a non-existent object must not create it."""

    def test_read_does_not_materialize(self):
        ts = TargetSystem({"d/geo": {"image": "v1"}})
        self.assertIsNone(ts.read_live("d/ghost"))
        self.assertNotIn("d/ghost", ts.live)
        self.assertEqual(ts.children("d/"), ["d/geo"])


class TestPathSegmentMatching(unittest.TestCase):
    """A sibling must not be mistaken for a child."""

    def test_sibling_is_not_a_descendant(self):
        self.assertFalse(is_descendant("d/geo-canary", "d/geo"))
        self.assertTrue(is_descendant("d/geo/sub", "d/geo"))
        self.assertTrue(covered_by("d/geo", "d/geo"))
        self.assertFalse(covered_by("d/geo-canary", "d/geo"))


class TestBothSigmaOrders(unittest.TestCase):
    """Sigma is a free choice of the task layer, so both orders must hold."""

    def test_mtpo_holds_under_either_launch_order(self):
        cfg = TrialConfig(p_misjudge=0.0, p_task_failure=0.0, jitter=0.3,
                          len_sigma=0.3)
        for name, cell in CELLS.items():
            refs = reference_states(cell)
            for order in ([0, 1], [1, 0]):
                for seed in range(8):
                    r = run_trial(cell, "MTPO", seed, cfg, refs, order=order)
                    self.assertTrue(
                        r.trajectory_match,
                        f"{name} order={order} seed={seed}: live state "
                        f"diverged from the trajectory",
                    )
                    self.assertTrue(
                        r.invariant_ok,
                        f"{name} order={order} seed={seed}: not serializable",
                    )


class TestOCCSoundness(unittest.TestCase):
    """OCC must not let an abort destroy a committed peer write."""

    def test_loser_undo_does_not_clobber_winner(self):
        tool = _field_tool()

        def build(oracle, rng, **kw):
            a = ScriptedAgent(
                name="A", oracle=oracle, rng=rng,
                plan=[
                    PlanStep(kind=CallKind.READ, obj="guard", reads=("guard",),
                             premise="g", duration=1.0, tokens=100),
                    PlanStep(kind=CallKind.WRITE, obj="p", tool=tool,
                             args={"field": "x", "value": "A"},
                             writes=("p",), duration=1.0, tokens=100),
                    PlanStep(kind=CallKind.THINK, duration=30.0, tokens=100),
                ],
            )
            b = ScriptedAgent(
                name="B", oracle=oracle, rng=rng,
                plan=[
                    PlanStep(kind=CallKind.THINK, duration=4.0, tokens=100),
                    PlanStep(kind=CallKind.WRITE, obj="p", tool=tool,
                             args={"field": "y", "value": "B"},
                             writes=("p",), duration=1.0, tokens=100),
                    PlanStep(kind=CallKind.WRITE, obj="guard", tool=tool,
                             args={"field": "x", "value": "B"},
                             writes=("guard",), duration=1.0, tokens=100),
                ],
            )
            return [a, b]

        cell = Cell(name="occ-clobber", suite="regress",
                    anomaly="loser undo destroys winner write",
                    initial={"guard": {"x": "i"}, "p": {"x": "i", "y": "i"}},
                    build_agents=build)
        r = _run(cell, "OCC-saga")
        self.assertEqual(
            r.final_state["p"], {"x": "A", "y": "B"},
            "A's committed write to p.x was destroyed by B's saga undo",
        )


class TestDeadlockDetectionWithAFinishedAgent(unittest.TestCase):
    """A two-agent cycle must be detected even if a third agent is done."""

    def test_cycle_found_via_wait_for_graph(self):
        tool = _field_tool()

        def mk(name, first, second):
            return ScriptedAgent(
                name=name, oracle=JudgmentOracle(random.Random(0), 0.0),
                rng=random.Random(0),
                plan=[
                    PlanStep(kind=CallKind.WRITE, obj=first, tool=tool,
                             args={"field": "x", "value": name},
                             writes=(first,), duration=4.0, tokens=100),
                    PlanStep(kind=CallKind.WRITE, obj=second, tool=tool,
                             args={"field": "x", "value": name},
                             writes=(second,), duration=4.0, tokens=100),
                ],
            )

        target = TargetSystem({"o1": {"x": "i"}, "o2": {"x": "i"},
                               "o3": {"x": "i"}})
        agents = [
            ScriptedAgent(
                name="A0", oracle=JudgmentOracle(random.Random(0), 0.0),
                rng=random.Random(0),
                plan=[PlanStep(kind=CallKind.WRITE, obj="o3", tool=tool,
                               args={"field": "x", "value": "a0"},
                               writes=("o3",), duration=1.0, tokens=100)],
            ),
            mk("A1", "o1", "o2"),
            mk("A2", "o2", "o1"),
        ]
        rt = Runtime(target, agents, PROTOCOLS["2PL-saga"](), random.Random(0))
        rt.jitter_sigma = 0.0
        rt.run()
        self.assertGreater(rt.deadlocks, 0,
                           "a genuine 2-agent cycle went undetected because a "
                           "third agent had already finished")
        self.assertFalse(
            any(s.status is Status.BLOCKED for s in rt.states.values()),
            "agents left permanently wedged",
        )


class TestA1RolledOncePerAgent(unittest.TestCase):
    """Restarts must not re-roll the individual-failure coin."""

    def test_failure_is_stable_across_generator_recreation(self):
        rng = random.Random(0)
        agent = ScriptedAgent(
            name="A", oracle=JudgmentOracle(rng, 0.0), rng=rng,
            p_task_failure=0.5,
            plan=[PlanStep(kind=CallKind.THINK, duration=1.0, tokens=10)],
        )
        first = agent.failed_alone
        for _ in range(10):
            list(agent.run(None, _FakeState()))
        self.assertEqual(agent.failed_alone, first)


class _FakeState:
    inbox: list = []
    repairs = 0
    notes_seen = 0
    context_tokens = 0
    step_idx = 0


if __name__ == "__main__":
    unittest.main()
