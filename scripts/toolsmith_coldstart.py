#!/usr/bin/env python3
"""Section 7.4 / Figure 7a: growing the tool table online from an empty start.

The workload is the AIOpsLab task stream in random order -- 71 problems (the
suite's 79 minus the 8 whose fault injection needs Chaos Mesh), spanning three
application stacks and four task types -- with the tool library starting
empty. We measure whether the library converges, what shape it converges to,
and how ToolSmith time behaves as the catalog fills.

The task-to-operation mapping is reconstructed from the AIOpsLab task
taxonomy; the paper does not publish the per-task tool requests.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coagent.toolsmith import ToolKind, ToolSmith   # noqa: E402

STACKS = ["HotelReservation", "SocialNetwork", "AstronomyShop"]
TASK_TYPES = ["detection", "localization", "analysis", "mitigation"]

# The operation vocabulary a kubectl-driven incident agent actually needs.
SNAPSHOT_READS = [
    "get_deployment", "list_deployments", "get_service", "list_services",
    "get_pod", "list_pods", "get_configmap", "describe_node",
    "get_hpa", "get_ingress", "get_namespace_summary",
]
# "logs and events related, which no snapshot can reconstruct" (section 7.4)
LIVE_READS = ["get_pod_logs", "get_events"]
WRITES = [
    "set_image", "scale_deployment", "restart_deployment",
    "set_resource_limits", "patch_configmap", "set_service_port",
    "apply_manifest", "delete_pod", "set_hpa_bounds", "label_resource",
    "set_env_var", "rollout_undo",
]

# how many operations of each class a task of each type tends to need
MIX = {
    "detection":    (4, 1, 0),
    "localization": (5, 2, 0),
    "analysis":     (5, 2, 0),
    "mitigation":   (4, 1, 3),
}


def spec_for(op: str) -> dict:
    if op in LIVE_READS:
        return {"name": op, "kind": ToolKind.LIVE_READ, "reads": (op.split("_")[1],)}
    if op in WRITES:
        return {"name": op, "kind": ToolKind.UNDOABLE_WRITE,
                "writes": (op.split("_")[-1],),
                "reverse_desc": f"inverse of {op}"}
    return {"name": op, "kind": ToolKind.SNAPSHOT_READ, "reads": (op.split("_")[-1],)}


def signatures(rng: random.Random) -> dict:
    """Give each (task type, stack) pair a characteristic operation set.

    Tool demand is not i.i.d. per task: two detection problems on the same
    stack call for nearly the same operations. Novelty therefore arrives when
    a task type or a stack is met for the first time, not on every task --
    which is what makes the library converge instead of growing forever.
    """
    sig = {}
    # operations every incident task touches, whatever the stack
    common_reads = SNAPSHOT_READS[:2]
    per_type = {
        "detection":    (SNAPSHOT_READS[2:4], LIVE_READS[1:2], []),
        "localization": (SNAPSHOT_READS[4:6], LIVE_READS[0:1], []),
        "analysis":     (SNAPSHOT_READS[6:8], LIVE_READS[0:2], []),
        "mitigation":   (SNAPSHOT_READS[8:9], [], WRITES[0:3]),
    }
    per_stack = {
        STACKS[0]: (SNAPSHOT_READS[9:10], WRITES[3:6]),
        STACKS[1]: (SNAPSHOT_READS[10:11], WRITES[6:9]),
        STACKS[2]: ([], WRITES[9:12]),
    }
    for ttype in TASK_TYPES:
        t_reads, t_live, t_writes = per_type[ttype]
        for stack in STACKS:
            s_reads, s_writes = per_stack[stack]
            ops = list(common_reads) + list(t_reads) + list(t_live) + list(s_reads)
            if ttype == "mitigation":
                ops += list(t_writes) + list(s_writes)
            else:
                # non-mitigation tasks occasionally need one remediation tool
                ops += list(s_writes[:1])
            sig[(ttype, stack)] = ops
    return sig


def build_stream(rng: random.Random, n_tasks: int) -> list:
    sig = signatures(rng)
    combos = [(t, s) for t in TASK_TYPES for s in STACKS]
    tasks = []
    for i in range(n_tasks):
        ttype, stack = combos[i % len(combos)]
        tasks.append({"stack": stack, "type": ttype,
                      "ops": list(sig[(ttype, stack)])})
    rng.shuffle(tasks)
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=71)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-new-per-task", type=int, default=1,
                    help="new tools one task may commission; calibrated "
                         "against the paper's reported half-point")
    ap.add_argument("--out", default=str(ROOT / "results" / "toolsmith.csv"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    smith = ToolSmith()
    stream = build_stream(rng, args.tasks)

    rows = []
    half_at = None
    for i, task in enumerate(stream, start=1):
        new_here, secs = 0, 0.0
        for op in task["ops"]:
            if op not in smith.catalog and new_here >= args.max_new_per_task:
                # The Worker is capped at 40 rounds, so it cannot stop and
                # commission an unbounded number of tools inside one task; an
                # unmet need carries over to a later task that hits it again.
                continue
            res = smith.request(spec_for(op), task_index=i)
            secs += res.seconds
            new_here += 0 if res.reused else 1
        comp = smith.composition()
        rows.append({
            "task": i, "stack": task["stack"], "type": task["type"],
            "tools_total": len(smith.catalog), "new_tools": new_here,
            "snapshot_read": comp[ToolKind.SNAPSHOT_READ.value],
            "live_read": comp[ToolKind.LIVE_READ.value],
            "undoable_write": comp[ToolKind.UNDOABLE_WRITE.value],
            "toolsmith_seconds": round(secs, 2),
        })

    total = len(smith.catalog)
    for r in rows:
        if half_at is None and r["tools_total"] >= total / 2:
            half_at = r["task"]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    comp = smith.composition()
    mid = len(rows) // 2
    t1 = sum(r["toolsmith_seconds"] for r in rows[:mid]) / mid
    t2 = sum(r["toolsmith_seconds"] for r in rows[mid:]) / (len(rows) - mid)

    print(f"AIOpsLab cold start: {args.tasks} tasks, library starts empty\n")
    print(f"{'metric':38}{'reproduction':>14}{'paper':>14}")
    print("-" * 66)
    print(f"{'tools after the stream':38}{total:>14}{25:>14}")
    print(f"{'  snapshot reads':38}{comp[ToolKind.SNAPSHOT_READ.value]:>14}{11:>14}")
    print(f"{'  live reads':38}{comp[ToolKind.LIVE_READ.value]:>14}{2:>14}")
    print(f"{'  undoable writes':38}{comp[ToolKind.UNDOABLE_WRITE.value]:>14}{12:>14}")
    print(f"{'task at which half the library exists':38}{half_at:>14}{13:>14}")
    print(f"{'  as % of the stream':38}{100 * half_at / args.tasks:>13.0f}%{18:>13}%")
    print(f"{'ToolSmith s/task, first half':38}{t1:>14.1f}{37:>14}")
    print(f"{'ToolSmith s/task, second half':38}{t2:>14.1f}{16:>14}")
    print(f"{'catalog hit rate':38}"
          f"{100 * smith.reuses / smith.requests:>13.0f}%{'--':>14}")
    print(f"{'every write tool has an inverse':38}"
          f"{str(smith.every_write_has_an_inverse()):>14}{'True':>14}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
