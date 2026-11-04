"""MTPO: Monotonic Trajectory Pre-Order (paper section 5.3).

Three rules over a serial rank assigned at launch:

  wr  reads pull     -- FilteredRead(j, o) = M(o, sigma_j)
  ww  writes apply speculatively in place, and a late write is either shadowed
      (Thomas write rule) or landed by undoing the suffix above it and
      reapplying that suffix in sigma order
  rw  notifications push, strictly from lower sigma to higher sigma

At GlobalQuiet every object's live copy equals the materialization of its
trajectory, i.e. live state and sigma-serial state coincide.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from ..core import WriteClass, WriteOp, covered_by
from ..runtime import AgentState, Call, Notification, ReadRecord, Runtime, Status
from .base import Outcome, Protocol


class MTPO(Protocol):
    """The paper's protocol."""

    name = "MTPO"

    def __init__(self, broadcast: bool = False):
        # ``broadcast`` turns off the pre-order, reproducing the straw-man of
        # section 5.2 that livelocks. It exists only for that experiment.
        self.broadcast = broadcast
        self.undos = 0
        self.shadowed = 0
        self.reorders = 0

    # -- wr: reads pull from the trajectory --------------------------------
    def on_read(self, rt: Runtime, st: AgentState, call: Call) -> Outcome:
        if self.broadcast:
            # No pre-order means no filtering: everyone reads the live copy.
            return Outcome(value=self.live_read(rt, st, call))
        if call.prefix is not None:
            names = rt.target.visible_children(call.prefix, st.sigma)
            value = {
                n: rt.target.traj(n).materialize(st.sigma) for n in names
            }
            st.read_set[call.prefix] = ReadRecord(
                copy.deepcopy(value), st.step_idx
            )
        else:
            value = rt.target.traj(call.obj).materialize(st.sigma)
            st.read_set[call.obj] = ReadRecord(
                copy.deepcopy(value), st.step_idx
            )
        return Outcome(value=value)

    # -- ww: writes apply speculatively ------------------------------------
    def on_write(self, rt: Runtime, st: AgentState, call: Call) -> Outcome:
        tool = call.tool
        if tool is not None and not tool.reversible:
            # Never speculate on an unrecoverable write (section 6.3).
            pending = [
                s for s in rt.others(st.name)
                if s.sigma < st.sigma and s.status is not Status.DONE
            ]
            if pending:
                return Outcome(
                    blocked=True,
                    blocked_on=f"unrecoverable:{call.obj}",
                )

        w = self.make_write(rt, st, call)
        traj = rt.target.traj(call.obj)
        traj.insert(w)
        st.writes.append(w)

        if self.broadcast:
            # Straw-man: write straight through, no trajectory repair.
            rt.target.apply_live(w)
            self._push_notifications(rt, st, call.obj)
            return Outcome(value=True)

        notified: set = set()
        applied_above = [
            x for x in traj.suffix_above(st.sigma) if x.applied_live
        ]
        if not applied_above:
            rt.target.apply_live(w)
        else:
            top_blind = traj.shadowing_entry(w)
            if top_blind is not None and top_blind.applied_live:
                # Thomas write rule: a blind entry above shadows this write.
                # It joins the trajectory but never reaches the live copy.
                self.shadowed += 1
                rt.trace(
                    f"{st.name} write {call.obj} shadowed by "
                    f"{top_blind.agent} (Thomas rule)"
                )
            else:
                notified = self._land_late_write(
                    rt, st, w, traj, applied_above
                )

        self._push_notifications(rt, st, call.obj, skip=notified)
        return Outcome(value=True)

    def _land_late_write(
        self,
        rt: Runtime,
        st: AgentState,
        w: WriteOp,
        traj,
        applied_above: List[WriteOp],
    ) -> set:
        """Undo the suffix above w in descending sigma, apply w, reapply.

        Returns the agents it notified, so the rw push does not notify the
        same reader twice -- a double notification bills the context twice and
        consumes two independent judgement rolls for one logical event.
        """
        self.reorders += 1
        rt.trace(
            f"{st.name} late write {w.obj}: reordering "
            f"{len(applied_above)} higher-sigma write(s)"
        )
        if any(not x.tool.reversible for x in applied_above):
            # Never start unwinding a suffix we cannot put back.
            rt.trace(
                f"{st.name} late write {w.obj}: suffix contains an "
                f"unrecoverable write; leaving it in place"
            )
            return set()
        for x in sorted(applied_above, key=lambda e: e.key, reverse=True):
            rt.target.undo_live(x)
            self.undos += 1
        rt.target.apply_live(w)
        for x in sorted(applied_above, key=lambda e: e.key):
            rt.target.reapply_live(x)
        # Each undone agent is told its write moved, and decides whether to act.
        notified = set()
        for x in applied_above:
            if x.agent != st.name and x.agent not in notified:
                notified.add(x.agent)
                rt.notify(
                    Notification(
                        obj=w.obj,
                        new_value=self._premise_for(rt, x.agent, w.obj),
                        from_agent=st.name,
                        to_agent=x.agent,
                        t=rt.now,
                        reason="reorder",
                    )
                )
        return notified

    # -- rw: notifications push, low sigma to high -------------------------
    def _push_notifications(
        self, rt: Runtime, st: AgentState, obj: str, skip: Optional[set] = None
    ) -> None:
        skip = skip or set()
        for other in rt.others(st.name):
            if other.status is Status.FAILED or other.name in skip:
                continue
            if not self.broadcast and other.sigma <= st.sigma:
                continue          # one-way: never notify a lower rank
            if not self._has_read(other, obj):
                continue
            value = (
                rt.target.read_live(obj) if self.broadcast
                else self._premise_for(rt, other.name, obj)
            )
            rt.notify(
                Notification(
                    obj=obj,
                    new_value=value,
                    from_agent=st.name,
                    to_agent=other.name,
                    t=rt.now,
                )
            )

    @staticmethod
    def _has_read(st: AgentState, obj: str) -> bool:
        return any(covered_by(obj, key) for key in st.read_set)

    @staticmethod
    def _premise_for(rt: Runtime, agent: str, obj: str):
        """Refresh the reader's premise at the position it actually read."""
        state = rt.state(agent)
        read_step = -1
        for key, rec in state.read_set.items():
            if covered_by(obj, key):
                read_step = max(read_step, getattr(rec, "step", -1))
        return rt.target.traj(obj).materialize_premise(
            state.sigma, agent, read_step
        )

    # -- quiescence --------------------------------------------------------
    def on_commit(self, rt: Runtime, st: AgentState) -> None:
        rt.wake_blocked()

    def on_quiescence(self, rt: Runtime) -> bool:
        """GlobalQuiet needs every delivered notification consumed."""
        reopened = False
        for s in rt.states.values():
            if s.inbox and s.status is Status.DONE:
                s.status = Status.RUNNING
                s.pending_send = None
                s.pending_call = None
                s.gen = s.agent.repair_run(rt, s)
                s.ready_at = max(s.ready_at, rt.now)
                reopened = True
        if not reopened:
            blocked = [s for s in rt.states.values() if s.status is Status.BLOCKED]
            if blocked:
                rt.wake_blocked()
                return True
        return reopened
