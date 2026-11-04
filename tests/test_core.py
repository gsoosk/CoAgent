"""Tests for the state substrate: trajectories, three-phase tools, undo."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coagent.core import (                      # noqa: E402
    Footprint, TargetSystem, Tool, WriteClass, WriteOp, blind_set, rmw_append,
)


def op(ts, obj, tool, args, agent, sigma, step):
    return WriteOp(obj=obj, tool=tool, args=args, agent=agent,
                   sigma=sigma, step=step)


class TestTrajectory(unittest.TestCase):
    def test_materialize_composes_rmw(self):
        """An RMW entry must see the value its predecessors left."""
        ts = TargetSystem({"log": []})
        t = rmw_append()
        for i, who in enumerate(["A", "B", "C"]):
            ts.traj("log").insert(op(ts, "log", t, {"item": who}, who, i, 0))
        self.assertEqual(ts.traj("log").materialize(), ["A", "B", "C"])

    def test_materialize_is_sigma_filtered(self):
        ts = TargetSystem({"log": []})
        t = rmw_append()
        for i, who in enumerate(["A", "B", "C"]):
            ts.traj("log").insert(op(ts, "log", t, {"item": who}, who, i, 0))
        self.assertEqual(ts.traj("log").materialize(0), ["A"])
        self.assertEqual(ts.traj("log").materialize(1), ["A", "B"])

    def test_blind_entry_shadows_only_what_it_covers(self):
        """Shadowing is coverage-based, not class-based.

        Using the same field for both writes -- the obvious way to write this
        test -- passes even when shadowing ignores coverage entirely, which is
        exactly how a lost write hides. Both cases are asserted.
        """
        ts = TargetSystem({"x": {"v": 0, "w": 0}})
        t = blind_set()
        low = op(ts, "x", t, {"field": "v", "value": 1}, "A", 0, 0)
        same = op(ts, "x", t, {"field": "v", "value": 9}, "B", 1, 0)
        ts.traj("x").insert(low)
        ts.traj("x").insert(same)
        self.assertEqual(ts.traj("x").materialize()["v"], 9)
        self.assertIs(ts.traj("x").shadowing_entry(low), same)

        ts2 = TargetSystem({"x": {"v": 0, "w": 0}})
        low2 = op(ts2, "x", t, {"field": "v", "value": 1}, "A", 0, 0)
        other = op(ts2, "x", t, {"field": "w", "value": 9}, "B", 1, 0)
        ts2.traj("x").insert(low2)
        ts2.traj("x").insert(other)
        self.assertIsNone(
            ts2.traj("x").shadowing_entry(low2),
            "a write to field `w` must not shadow a write to field `v`",
        )

    def test_materialize_premise_respects_the_read_position(self):
        """A refreshed premise reproduces the reader's read, not the live value."""
        ts = TargetSystem({"x": {"v": 0}})
        t = blind_set()
        ts.traj("x").insert(op(ts, "x", t, {"field": "v", "value": 5}, "A", 0, 0))
        ts.traj("x").insert(op(ts, "x", t, {"field": "v", "value": 7}, "B", 1, 3))
        self.assertEqual(ts.traj("x").materialize(1)["v"], 7)
        # B read at step 1, before its own write at step 3: own write excluded
        self.assertEqual(ts.traj("x").materialize_premise(1, "B", 1)["v"], 5)
        # B read at step 5, after its own write: read-your-own-writes
        self.assertEqual(ts.traj("x").materialize_premise(1, "B", 5)["v"], 7)


class TestUndo(unittest.TestCase):
    def test_blind_undo_restores_captured_value(self):
        ts = TargetSystem({"x": {"v": 1}})
        w = op(ts, "x", blind_set(), {"field": "v", "value": 2}, "A", 0, 0)
        ts.apply_live(w)
        self.assertEqual(ts.live["x"]["v"], 2)
        ts.undo_live(w)
        self.assertEqual(ts.live["x"]["v"], 1)

    def test_rmw_undo_is_element_wise(self):
        """Undoing an append must not discard a concurrent append."""
        ts = TargetSystem({"log": []})
        t = rmw_append()
        a = op(ts, "log", t, {"item": "A"}, "A", 0, 0)
        b = op(ts, "log", t, {"item": "B"}, "B", 1, 0)
        ts.apply_live(a)
        ts.apply_live(b)
        ts.undo_live(a)
        self.assertEqual(ts.live["log"], ["B"],
                         "snapshot restore would have lost B's append")

    def test_unrecoverable_tool_refuses_reverse(self):
        t = Tool(name="send_mail", wclass=WriteClass.BLIND,
                 exec_fn=lambda v, a: a["v"], reversible=False)
        with self.assertRaises(RuntimeError):
            t.reverse({}, {}, {})


class TestFootprint(unittest.TestCase):
    def test_conflict_detection(self):
        a = Footprint.of(reads=["x"], writes=["y"])
        b = Footprint.of(reads=["y"], writes=["z"])
        c = Footprint.of(reads=["p"], writes=["q"])
        self.assertTrue(a.touches(b))
        self.assertFalse(a.touches(c))


class TestVisibility(unittest.TestCase):
    def test_range_read_hides_higher_sigma_creations(self):
        """A's scan at sigma 0 must not see an object B created at sigma 1."""
        ts = TargetSystem({"d/geo": {"image": "bad"}})
        create = Tool(name="create", wclass=WriteClass.RMW,
                      exec_fn=lambda v, a: dict(a["spec"]))
        w = op(ts, "d/canary", create, {"spec": {"image": "bad"}}, "B", 1, 0)
        ts.ensure("d/canary")
        ts.apply_live(w)
        ts.traj("d/canary").insert(w)
        self.assertEqual(ts.visible_children("d/", 0), ["d/geo"])
        self.assertEqual(ts.visible_children("d/", 1), ["d/canary", "d/geo"])


if __name__ == "__main__":
    unittest.main()
