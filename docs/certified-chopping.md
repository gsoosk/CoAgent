# Certified chopping, implemented and measured

Implementation of *Certified Chopping for Agent Tool Workflows: Admitting
Plan Revisions over Committed Effects* (`idea/idea.pdf`) inside this
codebase, run against real LLM Workers beside the existing protocols.

## What is implemented

| Paper | Here |
|---|---|
| SC graph, mixed simple cycle via vertex-biconnected blocks | `chopping.sc_edges`, `_biconnected_blocks`, `mixed_block` |
| Range footprints, phantoms visible to certification | `chopping.Access` as half-open intervals |
| `seal` (Lemma 1) | `chopping.seal` |
| Theorem 2 characterization | `chopping.admit` — one test on the coarsest candidate |
| Corollary 1 (impossibility) | `chopping.cycle_through_committed` |
| Theorem 3 (serialized admission) | single plan snapshot with the frontier fixed |
| §3.4 three-step algorithm | `admit` → `seal` test → `guided_coarsen` |
| Mediated, fail-closed enforcement | `protocols.chopping`, checked per call against the live manifest |
| Whole-footprint atomic lock acquisition | `CertifiedChopping._acquire`, deadlock-free by construction |

Not implemented: durable continuation and crash recovery. This codebase's
store is in-memory, so there is no crash boundary to protect — §6.4 of the
paper has no analogue here.

## Verification (`tests/test_chopping.py`)

Your Table 1, against a brute-force oracle that enumerates every interleaving
of pieces and tests conflict-serializability at workflow granularity. All 10
pass:

static certification (2 and 3 workflows) · repair soundness · Lemma 2 over
exhaustive future-only coarsenings · Theorem 2 characterization (250
scenarios vs exhaustive search) · admission soundness with the committed
prefix never rewritten · Theorem 3 counterexample · Corollary 1 witness ·
range footprints catch phantoms · fail-closed enforcement.

## Results: real LLM Workers

`claude-haiku-4-5`, 12 cells × 6 protocols × 10 seeds = 720 trials, $10.79.
Correctness is the anomaly-free rate; two cells are excluded because serial
itself fails their predicate.

| protocol | anomaly-free | 95% CI | speedup | token cost | rounds/trial |
|---|---|---|---|---|---|
| serial | 100% | [96, 100] | 1.00 | 1.00 | 6.9 |
| naive | 56% | [46, 65] | 1.45 | 1.19 | 7.7 |
| 2PL-saga | 96% | [90, 98] | 1.05 | 1.20 | 8.5 |
| OCC-saga | 88% | [80, 93] | 0.76 | 1.91 | 12.3 |
| MTPO | 93% | [86, 97] | 1.47 | 1.22 | 7.7 |
| **certified chopping** | **96%** | **[90, 98]** | **1.23** | **1.01** | **7.1** |

Per-cell aggregates are **medians**. One 2PL trial in 722 ended with an
agent still blocked and its virtual clock ran away to 15029 s against a 16 s
cell median; a single such trial moves a mean by two orders of magnitude and
a median not at all. Using means, 2PL's speedup reads 0.93 instead of 1.05.

Certified chopping is the best safety on offer among concurrent policies —
tied with 2PL, above MTPO — and it is the only one of the three that is both
safe *and* faster than serial, at essentially serial token cost. It takes
fewer rounds than every other concurrent policy and has zero deadlocks by
construction, against 2PL's 0.79 per trial in the simulated matrix.

## Figures

| file | what it shows |
|---|---|
| `fig_chopping.png` | safety, speed, cost and the safety-vs-speed front against every other protocol |
| `fig_chopping_where.png` | per-cell speedup over whole-workflow 2PL — where chopping pays and where it correctly collapses |
| `fig_chopping_admission.png` | what certification decided, at launch and mid-flight |
| `fig6_llm.png` | the case-study timeline, now including a chopping row |

`fig6_llm.png` puts chopping on the same measured clock as the other five
protocols on `k8s-canary`: **15.5 s, PASS**, with a diamond at t=0 marking
certification merging three pieces at launch. The row then reads as
whole-workflow locking — B idle until A releases — which is why it sits
between MTPO (8.8 s) and 2PL (22.4 s) rather than winning outright on a cell
whose structure forbids chopping.

## Where it pays and where it does not

Two workloads from the paper were added as cells.

`cc-claim` (short shared piece, long private interval) is the structure
chopping exists for: **1.89× against whole-workflow 2PL's 1.00×** in the
simulated runs.

On the ten MTPO contention cells, certification merges the pieces at launch
in **100 of 120 trials** and the policy becomes whole-workflow locking —
exactly the paper's honest negative result. Every one of those cells has a
read-modify-write or scan-then-insert straddling the reasoning gap. It bought
no speed there, and it is important that it bought none rather than buying
speed unsafely.

## Admission outcomes

Of 120 chopping trials there were **10 mid-flight revisions, and all 10 were
refused** with the cycle running through a committed piece.

That 0% is not a measurement of your Table 4 range (4–37%). The only cell in
this suite that generates revisions is `cc-escalate`, which is constructed to
be inadmissible — it is Corollary 1 made executable. Ten out of ten refusals
on a cell designed to be refused is a correctness result, not an admission
rate. Measuring the real rate needs workloads whose escalations are sometimes
benign.

What it does show is that the impossibility fires against real agents, not
just in a simulation: `shared/y` was untouched in every trial, the final
state stayed serializable, and the agent's intent was simply unmet — which
the paper treats as an application-level obligation to compensate or replan.

## Two bugs the design caught in this codebase

1. Locks were not released at the reasoning gap, because `THINK` steps bypass
   dispatch. Releasing across reasoning is the entire mechanism; an
   `on_think` protocol hook was added.
2. Then the opposite: that hook released locks even for pieces certification
   had *merged*, splitting a whole-workflow piece and deadlocking both
   agents. Think steps now carry the piece index of the next I/O step.

A third surfaced only under real agents: an LLM has no `THINK` step, so its
piece never closed before it escalated. Each declared piece now closes when
its call returns — for a model-driven agent every inter-call gap *is* the
inference interval. But applying that unconditionally released merged pieces
per call and lost updates on all three lost-update cells; `Piece.span`
records how many declared pieces a live one covers, and a span > 1 holds its
locks through.
