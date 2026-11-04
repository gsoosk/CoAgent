"""Kubernetes-flavoured tools and the AIOpsLab-derived contended cells.

Object ids mirror the cluster tree (`k8s/deploy/<name>`), so a range read is
a prefix read and `list deployments` is a genuine range operation.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

from ..core import Tool, WriteClass
from ..runtime import CallKind
from ..agents.scripted import PlanStep, ScriptedAgent
from .base import Cell, register
from .patterns import lost_update, rmw_summary, sweep_derive

CANONICAL = "canonical:v2"
BAD = "broken:v1"


# -- tools -----------------------------------------------------------------
def _set_field(field: str) -> Tool:
    def _exec(value, args):
        out = dict(value or {})
        out[field] = args[field]
        return out

    return Tool(name=f"set_{field}", wclass=WriteClass.BLIND, exec_fn=_exec)


set_image = _set_field("image")
scale_deployment = _set_field("replicas")
set_port = _set_field("port")
set_permission = _set_field("permission")


def _create_exec(value, args):
    # Creating an object is non-idempotent: a replay would make a second one.
    return copy.deepcopy(args["spec"])


create_deployment = Tool(
    name="create_deployment",
    wclass=WriteClass.RMW,
    exec_fn=_create_exec,
    prepare_fn=lambda v, a: copy.deepcopy(v),
    reverse_fn=lambda cur, cap, a: copy.deepcopy(cap),
)

DEPLOY = "k8s/deploy/"


# -- cell 1: the canary anomaly (paper sections 2.2 and 7.3) ---------------
def _canary_agents(oracle, rng, **kw):
    """Agent A repairs every mismatched image; Agent B mirrors geo into a canary."""

    def a_fixes(view) -> List[PlanStep]:
        scope = view.get("scope", {})
        steps = []
        for name in sorted(scope):
            spec = scope[name] or {}
            if spec.get("image") != CANONICAL:
                steps.append(
                    PlanStep(
                        kind=CallKind.WRITE,
                        obj=name,
                        tool=set_image,
                        args={"image": CANONICAL},
                        writes=(name,),
                        duration=5.0,
                        tokens=300,
                        label=f"set_image({name.split('/')[-1]})",
                        depends_on=("scope",),
                    )
                )
        return steps

    agent_a = ScriptedAgent(
        name="A-repair",
        oracle=oracle,
        rng=rng,
        p_task_failure=kw.get("p_task_failure", 0.0),
        plan=[
            PlanStep(
                kind=CallKind.READ,
                prefix=DEPLOY,
                reads=(DEPLOY,),
                premise="scope",
                duration=4.0,
                tokens=600,
                label="list deployments",
            ),
            PlanStep(kind=CallKind.THINK, duration=3.0, tokens=260,
                     label="plan repairs"),
            PlanStep(kind=CallKind.WRITE, expand=a_fixes, label="apply fixes",
                     depends_on=("scope",)),
            PlanStep(
                kind=CallKind.READ,
                prefix=DEPLOY,
                reads=(DEPLOY,),
                duration=4.0,
                tokens=400,
                label="closing check",
                repairable=False,
            ),
        ],
    )

    canary = DEPLOY + "geo-canary"
    geo = DEPLOY + "geo"

    agent_b = ScriptedAgent(
        name="B-canary",
        oracle=oracle,
        rng=rng,
        p_task_failure=kw.get("p_task_failure", 0.0),
        plan=[
            PlanStep(
                kind=CallKind.READ,
                obj=geo,
                reads=(geo,),
                premise="geo",
                duration=4.5,
                tokens=400,
                label="read geo.image",
            ),
            PlanStep(kind=CallKind.THINK, duration=3.0, tokens=260,
                     label="plan canary"),
            PlanStep(
                kind=CallKind.WRITE,
                obj=canary,
                tool=create_deployment,
                args=lambda view: {
                    "spec": {
                        "image": (view.get("geo") or {}).get("image", BAD),
                        "replicas": 0,
                        "role": "canary",
                    }
                },
                writes=(canary,),
                duration=12.0,
                tokens=520,
                label="create geo-canary",
                depends_on=("geo",),
                # The repair is one set_image, not a re-create (section 7.3).
                repair_with=[
                    PlanStep(
                        kind=CallKind.WRITE,
                        obj=canary,
                        tool=set_image,
                        args=lambda view: {
                            "image": (view.get("geo") or {}).get("image", BAD)
                        },
                        writes=(canary,),
                        duration=5.0,
                        tokens=300,
                        label="repair canary image",
                        repairable=False,
                    )
                ],
            ),
        ],
    )
    return [agent_a, agent_b]


def _canary_invariant(final: Dict[str, Any]) -> bool:
    """The joint invariant spanning both agents: the canary mirrors live geo."""
    canary = final.get(DEPLOY + "geo-canary")
    geo = final.get(DEPLOY + "geo")
    if canary is None or geo is None:
        return False
    return canary.get("image") == geo.get("image") == CANONICAL


register(
    Cell(
        name="k8s-canary",
        suite="aiopslab",
        anomaly="rw antidependency cycle (stale read + missed range scan)",
        description=(
            "A repairs every deployment on a wrong image; B reads geo's image "
            "and builds a zero-replica canary mirroring it. A's scan precedes "
            "the canary, B's read precedes A's fix."
        ),
        initial={
            DEPLOY + "geo": {"image": BAD, "replicas": 3},
            DEPLOY + "profile": {"image": BAD, "replicas": 2},
            DEPLOY + "reservation": {"image": BAD, "replicas": 2},
            DEPLOY + "frontend": {"image": CANONICAL, "replicas": 2},
        },
        build_agents=_canary_agents,
        semantic_invariant=_canary_invariant,
        llm_spec={
            "tasks": [
                "A faulty rollout left an unknown subset of deployments "
                f"running a wrong container image. The canonical image is "
                f"{CANONICAL!r}. List the deployments, find every affected "
                "one, and restore its image to the canonical one. Then "
                "verify your work.",
                f"Prepare a canary release for the 'geo' service. Read the "
                f"current image of '{DEPLOY}geo', then create a deployment "
                f"'{DEPLOY}geo-canary' mirroring that image, with "
                "replicas=0 and role='canary'. The canary must end up "
                "mirroring the image geo is actually running.",
            ],
            "tools": [
                {"name": "list_deployments", "kind": "read", "prefix": DEPLOY,
                 "description": "list every deployment with its spec"},
                {"name": "get_deployment", "kind": "read",
                 "description": "read one deployment's spec"},
                {"name": "set_image", "kind": "write", "tool": set_image,
                 "args": {"image": "image string"},
                 "description": "set a deployment's image (reversible)"},
                {"name": "create_deployment", "kind": "write",
                 "tool": create_deployment,
                 "args": {"spec": "object with image, replicas, role"},
                 "description": "create a new deployment (reversible)"},
            ],
        },
    )
)


# -- cells 2-5: the rest of the AIOpsLab-derived pairs ---------------------
# AIOpsLab supplies "port misconfiguration, image rollback, permission
# revocation, and the like" (paper section 7.1). The partner agent for each is
# reconstructed, since the paper does not publish its ten pairs.

SVC = "k8s/svc/"
ROLE = "k8s/role/"

register(
    sweep_derive(
        name="k8s-port-misconfig",
        suite="aiopslab",
        prefix=SVC,
        field="port",
        good=8080,
        source=SVC + "geo",
        target=SVC + "geo-ingress",
        description=(
            "A repairs every service left on a wrong port; B reads geo's port "
            "and creates an ingress forwarding to it."
        ),
        initial={
            SVC + "geo": {"port": 9999},
            SVC + "profile": {"port": 9999},
            SVC + "search": {"port": 8080},
        },
        spec_extra={"kind": "ingress"},
    )
)

register(
    sweep_derive(
        name="k8s-permission-revoke",
        suite="aiopslab",
        prefix=ROLE,
        field="permission",
        good="read-only",
        source=ROLE + "ops",
        target=ROLE + "ops-clone",
        description=(
            "A revokes over-broad permissions across roles; B clones the ops "
            "role for a new team, mirroring the permission it read."
        ),
        initial={
            ROLE + "ops": {"permission": "cluster-admin"},
            ROLE + "ci": {"permission": "cluster-admin"},
            ROLE + "viewer": {"permission": "read-only"},
        },
        spec_extra={"team": "platform"},
    )
)

register(
    lost_update(
        name="k8s-replica-scale",
        suite="aiopslab",
        obj=DEPLOY + "frontend",
        field="replicas",
        initial={DEPLOY + "frontend": {"replicas": 4, "image": CANONICAL}},
        a_fn=lambda v: (v or 0) * 2,
        b_fn=lambda v: (v or 0) + 3,
        a_label="scale-for-load",
        b_label="add-headroom",
        a_desc="double it, to absorb a load spike",
        b_desc="add three, for headroom",
        description=(
            "A doubles replicas for a load spike while B adds three for "
            "headroom; both compute from the replica count they read."
        ),
    )
)

register(
    rmw_summary(
        name="k8s-rollback-audit",
        suite="aiopslab",
        prefix=DEPLOY,
        field="image",
        good=CANONICAL,
        log_obj="k8s/audit/log",
        description=(
            "A rolls every deployment back to the canonical image; B appends "
            "a compliance record listing the images it observed."
        ),
        initial={
            DEPLOY + "geo": {"image": BAD},
            DEPLOY + "profile": {"image": BAD},
            DEPLOY + "rate": {"image": CANONICAL},
            "k8s/audit/log": [],
        },
    )
)
