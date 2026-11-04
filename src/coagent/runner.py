"""Trial driver: one (cell, protocol, seed) run, and the reference oracle."""

from __future__ import annotations

import copy
import random
import zlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .agents.scripted import JudgmentOracle
from .protocols import PROTOCOLS
from .protocols.base import SerialProtocol
from .runtime import Runtime, Status, TrialResult
from .workloads.base import Cell, invariant_ok, normalize


@dataclass
class TrialConfig:
    """Knobs that stand in for the agent assumptions of section 5.1."""

    p_misjudge: float = 0.05     # A3: notified agent misreads relevance
    p_task_failure: float = 0.01  # A1: agent fails on its own (serial != 100%)
    # Chance an aborted agent's redo truncates early, losing the rest of its
    # plan. Default 0 -- a perfect replay -- because the rest of this config is
    # calibrated to the paper's model, whose 98% serial baseline implies it
    # also redoes well. Measured at 0.22 on claude-haiku-4-5 (a ~15 point loss
    # of task completeness across 20 real abort events); set it when modelling
    # a weaker agent. It only ever affects the abort-based protocols.
    p_redo_truncate: float = 0.0
    retry_cap: int = 5            # paper section 7.1
    jitter: float = 0.40
    # latency scale, calibrated so the canary cell's serial baseline is the
    # 50.2 s the paper reports for it in section 7.3
    time_scale: float = 1.103
    # spread of per-agent task length within a pair. The paper notes speedup
    # "sits around 1.5x rather than 2x because per-task latency varies, and a
    # pair's wall-clock is bounded by the slower task" (section 7.2); the pair
    # lengths themselves are not published, so this is the one workload knob
    # calibrated against their reported uncoordinated speedup of 1.54x.
    len_sigma: float = 0.40


def _build(cell: Cell, order: List[int], cfg: TrialConfig, rng) -> List[Any]:
    oracle = JudgmentOracle(rng, p_misjudge=cfg.p_misjudge)
    agents = cell.build_agents(
        oracle, rng,
        p_task_failure=cfg.p_task_failure,
        p_redo_truncate=cfg.p_redo_truncate,
    )
    return [agents[i] for i in order]


def reference_states(cell: Cell) -> List[Dict[str, Any]]:
    """Run the cell serially in both orders: the two acceptable outcomes."""
    refs = []
    clean = TrialConfig(p_misjudge=0.0, p_task_failure=0.0, jitter=0.0,
                        p_redo_truncate=0.0)
    for order in ([0, 1], [1, 0]):
        rng = random.Random(0)
        target = cell.fresh_target()
        agents = _build(cell, order, clean, rng)
        rt = Runtime(target, agents, SerialProtocol(), rng, retry_cap=5)
        rt.jitter_sigma = 0.0
        rt.time_scale = 1.0
        rt.run()
        refs.append(normalize(target.live))
    return refs


def serial_baseline(cell: Cell, cfg: TrialConfig, seed: int) -> TrialResult:
    return run_trial(cell, "serial", seed, cfg)


def run_trial(
    cell: Cell,
    protocol_name: str,
    seed: int,
    cfg: Optional[TrialConfig] = None,
    refs: Optional[List[Dict[str, Any]]] = None,
    keep_trace: bool = False,
    order: Optional[List[int]] = None,
) -> TrialResult:
    """Run one trial.

    ``order`` is the launch order, which is what fixes each agent's sigma
    rank. It is a free choice of the task layer, so both assignments must
    hold -- pinning it to [0, 1] hides any agent that cannot heal.
    """
    cfg = cfg or TrialConfig()
    refs = refs if refs is not None else reference_states(cell)
    rng = random.Random(seed)
    target = cell.fresh_target()
    order = order or [0, 1]
    agents = _build(cell, order, cfg, rng)
    protocol = PROTOCOLS[protocol_name]()
    rt = Runtime(target, agents, protocol, rng, retry_cap=cfg.retry_cap)
    rt.jitter_sigma = cfg.jitter
    rt.time_scale = cfg.time_scale
    if cfg.len_sigma > 0:
        # Task lengths are a property of the (cell, trial), not of the
        # protocol: every protocol must face the same pair of agents, or the
        # comparison is unpaired and the speedup ratio is meaningless.
        tag = f"{cell.name}:{seed}:{order}".encode()
        len_rng = random.Random(zlib.crc32(tag))
        for st in rt.states.values():
            st.speed = max(0.25, len_rng.lognormvariate(0.0, cfg.len_sigma))
    rt.run()

    wall, tokens = rt.totals()
    failed = any(s.status is Status.FAILED for s in rt.states.values())
    ok = (not rt.livelocked) and (not failed) and invariant_ok(
        target.live, refs, cell
    )
    # The protocol's own stated invariant, independent of the serial oracle:
    # at quiescence the live copy must equal the materialized trajectory.
    serial_match = normalize(target.live) == normalize(target.serial_state())
    return TrialResult(
        protocol=protocol_name,
        cell=cell.name,
        seed=seed,
        wall_clock=wall,
        tokens=tokens,
        invariant_ok=ok,
        deadlocks=rt.deadlocks,
        aborts=rt.aborts,
        notifications=rt.notifications,
        repairs=sum(s.repairs for s in rt.states.values()),
        undos=target.undo_count,
        livelocked=rt.livelocked or failed,
        final_state=normalize(target.live),
        trajectory_match=serial_match,
        note="; ".join(rt.trace_lines) if keep_trace else "",
    )
