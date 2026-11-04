# Reproduction notes: CoAgent / MTPO (arXiv 2606.15376v1)

Lyu, Zhang, Wu, Wei, Chen. *CoAgent: Concurrency Control for Multi-Agent
Systems.* Submitted to ATC 2026. 14 pages, cs.DC / cs.AI / cs.MA.

## 1. Artifact status

The authors' code is **not public**:

- the paper contains no availability statement and no artifact link;
- the arXiv abstract page (comments: "14 pages, 7 figures. Submitted to ATC
  2026") carries no code, data or project-page link;
- no repository from the authors or their group appears on GitHub.

One near-match exists, `Origamii520/mini-coagent` (created 2026-07-07, 0
stars, Peking University, with `SUBMISSION_CHECKLIST.md` and `LLM_USAGE.md`
in the tree). It is a third-party course reimplementation, not the SJTU
authors' artifact, and was not used here.

This reproduction is therefore written from the paper alone.

## 2. What is implemented faithfully

The paper specifies the protocol completely, so these are transcriptions
rather than reconstructions.

| Paper | Here |
|---|---|
| Object tree, natural + abstract objects, lazy instantiation (§6.1) | `core.TargetSystem` |
| Footprint-declared tool calls (§5.1, §6.1) | `core.Footprint`, declared per `PlanStep` |
| Write trajectory and its materialization (§5.1) | `core.Trajectory` |
| Blind vs RMW write classes, idempotence criterion (§2.1) | `core.WriteClass` |
| Three-phase tools: `prepare` / `exec` / `reverse` (§6.3) | `core.Tool` |
| `unrecoverable` tag, held until lower-σ agents commit (§6.3) | `Tool.reversible`, gated in `MTPO.on_write` |
| **wr** — filtered read `M(o, σⱼ)` (§5.3) | `MTPO.on_read` |
| **ww** — speculative in-place write, Thomas-rule shadowing, undo-suffix-and-reapply (§5.3) | `MTPO.on_write`, `_land_late_write` |
| **rw** — one-way low-σ→high-σ notification (§5.3) | `MTPO._push_notifications` |
| Re-opening a quiescent receiver; GlobalQuiet (§5.3) | `MTPO.on_quiescence` |
| Unconditional-broadcast straw-man (§5.2) | `MTPO(broadcast=True)` |
| 2PL-saga with range read locks, deadlock victim, saga undo (§7.1) | `protocols.classical.TwoPhaseLocking` |
| OCC-saga, backward validation, in-place writes, saga undo (§7.1) | `protocols.classical.OptimisticCC` |
| ToolSmith: read-only, dedup against catalog, footprint marking, bash audit (§6.4) | `toolsmith.ToolSmith` |
| Correctness rule: equivalent to agent-1·agent-2 or agent-2·agent-1 (§7.1) | `workloads.base.invariant_ok`, references derived by running serially in both orders |

Two places where the paper is silent and a decision was needed, both marked
in the source:

1. **Notification value for a receiver that has itself written the object.**
   The paper says a notification carries "the refreshed `M(o, σⱼ)`". Since a
   notification refreshes a *premise*, and the premise was a read taken
   before the receiver wrote anything, the receiver's own writes are excluded
   (`Trajectory.materialize_premise`). For the canary case the two readings
   coincide; they differ only when an agent reads and writes the same object.
2. **Repair of a step that has not executed yet.** A notification can land
   before the operation that depends on the premise runs. Re-issuing it would
   double the effect, so pending steps simply execute against the refreshed
   premise and only already-executed steps are re-issued — which is the
   "keep, rewrite, or retract its *pending* operations" of §5.3.

## 3. What is reconstructed (the main fidelity gap)

**The ten contended cells are not in the paper.** §7.1 says five tasks were
taken from each of WorkBench and AIOpsLab and "a matching agent-2"
hand-constructed for each, with hand-written invariants — but it never lists
the tasks, the partner agents, or the invariants, and there is no appendix.
Since the headline numbers are averages over exactly those ten pairs, nobody
outside the author group can reproduce the specific bar heights.

What is reconstructed here:

- **`k8s-canary`** is faithful. §2.2, Figure 1 and §7.3 specify the initial
  state, both agents' plans, the interleaving and the invariant precisely.
- **The other nine** are built to the stated recipe — five AIOpsLab-derived
  (the paper names "port misconfiguration, image rollback, permission
  revocation, and the like") and five WorkBench-derived across its five
  domains — each paired with a partner chosen to produce a textbook anomaly.
  They span four anomaly families: phantom-insert + stale read, lost update,
  read skew, and non-idempotent append over a mutating range.
- **Agent plans** are scripted with an explicit premise graph rather than
  driven by an LLM, so the paper's agent assumptions become explicit knobs
  (§4 below). `agents/llm.py` swaps in a real `deepseek-v4-flash` Worker.
- **The ToolSmith operation vocabulary** is reconstructed from kubectl /
  AIOpsLab practice.

## 4. Free parameters, and what each was calibrated against

Everything else falls out of the protocol. Four numbers were fitted, each to
a single statistic the paper reports:

| Parameter | Value | Calibrated against |
|---|---|---|
| `time_scale` | 1.103 | the canary cell's serial baseline, 50.2 s (§7.3) |
| `len_sigma` (spread of task length within a pair) | 0.40 | uncoordinated speedup, 1.54× (§7.2) |
| `max_new_per_task` (ToolSmith) | 1 | half the library by task 13 (§7.4) |
| `jitter` (per-step latency noise) | 0.40 | not fitted to a number; a round value mid-sweep |

Two more are set directly from values the paper states rather than fitted:

| Parameter | Value | Source |
|---|---|---|
| `p_misjudge` (A3 gap) | 0.05 | "misjudging a notification's relevance in only 5% of trials" (§1, §7.2) |
| `p_task_failure` (A1 gap) | 0.01 | gives serial correctness 98%, the paper's measured serial baseline |

`len_sigma` deserves a note. With two evenly-matched agents a pair's speedup
tends to 2×; the paper gets 1.54× and explains it — "per-task latency varies,
and a pair's wall-clock is bounded by the slower task". Task lengths are a
property of their ten pairs, which are unpublished, so this is the one
workload knob set from their reported number. It is drawn per (cell, trial)
and held fixed across protocols, so every protocol faces the same pair.

## 5. Results

2000 trials (10 cells × 5 protocols × 40 seeds):
`python3 scripts/run_matrix.py --trials 40`. The N=10 default gives the same
numbers to within a point.

| Protocol | Correct (paper) | Speedup (paper) | Cost (paper) |
|---|---|---|---|
| serial | 98% (98%) | 1.00 (1.00) | 1.00 (1.00) |
| naive | 8% (13%) | 1.63 (1.54) | 0.93 (1.06) |
| 2PL-saga | 98% (96%) | 1.00 (1.04) | 1.27 (1.45) |
| OCC-saga | 95% (93%) | 0.81 (0.93) | 1.49 (1.83) |
| **MTPO** | **92% (93%)** | **1.51 (1.43)** | **1.18 (1.15)** |

A non-contended control (`workloads/control.py`, two agents on disjoint
objects) confirms the baselines are not strawmanned: 2PL, OCC and MTPO all
reach the uncoordinated speedup of 1.77x with zero deadlocks, zero aborts and
no token overhead. Everything the classical protocols pay in the matrix comes
from real contention, not from the middleware.

Contention rates, which were not calibrated against anything:

| | reproduction | paper |
|---|---|---|
| 2PL deadlocks per trial | 0.79 | 0.81 |
| OCC aborts per trial | 0.98 | 0.95 |

Case study, `scripts/case_study.py` (paper Figure 6 in brackets):

serial 50.2 s [50.2] · naive 28.7 s ✗ [31.8 ✗] · 2PL 50.7 s [53.1] ·
OCC 62.9 s [63.7] · MTPO 28.7 s [32.8]. The ordering — naive ≈ MTPO <
serial ≈ 2PL < OCC — reproduces, with naive failing and the rest passing.

Section 5.2 straw-man, `scripts/livelock.py`: unconditional broadcast makes
1076 repairs and walks x and y to (0, 0), matching neither serial order, and
stops only because floats underflow. MTPO converges in one heal to (0.5,
0.25). This reproduces Figure 2 and confirms the pre-order is load-bearing.

Cold start, `scripts/toolsmith_coldstart.py`: the library converges to 25
tools — 11 snapshot reads, 2 live reads, 12 undoable writes — with half of it
present by task 13 (18% of the stream), ToolSmith time falling 26.5 → 13.4
s/task, a 95% catalog hit rate, and every write tool carrying an inverse.
**The 25/11/2/12 composition is an input, not a result**: the operation
vocabulary was sized from the paper's reported composition. The genuine
outputs are the convergence, the front-loaded shape, the hit rate, and the
inverse coverage.

## 6. Where the reproduction differs, and why

- **naive correctness 8% vs 13%.** The reconstructed pairs are more reliably
  adversarial than theirs; every one of our cells races on nearly every
  schedule. Raising latency noise lifts this to ~8% but no further without
  making the cells less contended, which would be fitting to the number.
- **2PL and OCC token cost run low (1.27 vs 1.45, 1.49 vs 1.83).** Our redo
  cost model bills a cleared context re-running a scripted plan; a real agent
  re-exploring after an abort spends more than a replay does.
- **OCC speedup 0.81 vs 0.93.** Our OCC victim restarts strictly after the
  winner commits, which is the worst case of the stated protocol.

## 7. What this reproduction cannot establish

- **The real A3 rate.** `p_misjudge = 0.05` is taken from the paper, not
  measured. Measuring it needs real Workers; `agents/llm.py` is wired for
  exactly that and needs `OPENROUTER_API_KEY`. Until then, MTPO's 93–94%
  correctness is downstream of an assumption, not evidence for it.
- **Real wall-clock and real token cost.** Time here is virtual, from a
  latency model calibrated to one reported baseline.
- **§7.4 pass rates (63/71 vs 45/71).** These need real agents against a live
  `kind` cluster running HotelReservation, SocialNetwork and Astronomy Shop.
  Docker is installed here but the daemon is stopped, and `kind` and `helm`
  are absent; `microsoft/AIOpsLab` and `olly-styles/WorkBench` are both
  public and MIT-licensed, so that path is open but was not run.

What it does establish: MTPO as specified is implementable; it is serializable
in the pre-decided order on every cell, under both launch orders, across five
noise regimes, with no deadlock, abort or livelock, and its own invariant
(live state == sigma-serial state) holds in 400/400 matrix trials
(`tests/test_stress.py`, `tests/test_review_regressions.py`);
the pre-order is necessary rather than decorative, and the protocol's
*relative* position against 2PL and OCC on correctness, concurrency retained
and token overhead reproduces closely.
