"""WorkBench-derived contended cells.

WorkBench is a simulated office-automation harness over five domains (CRM,
calendar, email, analytics, project management). Its tasks are mutually
independent by construction, so as in the paper each one is paired with a
hand-built partner chosen to make the pair race.
"""

from __future__ import annotations

from .base import register
from .patterns import lost_update, rmw_summary, sweep_derive

CRM = "wb/crm/customer/"
CAL = "wb/calendar/"
MAIL = "wb/email/msg/"
ANALYTICS = "wb/analytics/record/"
PROJ = "wb/project/member/"

register(
    sweep_derive(
        name="wb-crm-dedupe",
        suite="workbench",
        prefix=CRM,
        field="tier",
        good="gold",
        source=CRM + "acme",
        target=CRM + "acme-invoice",
        description=(
            "A normalizes customer tiers after a bad import; B reads Acme's "
            "tier and raises an invoice priced from it."
        ),
        initial={
            CRM + "acme": {"tier": "bronze"},
            CRM + "globex": {"tier": "bronze"},
            CRM + "initech": {"tier": "gold"},
        },
        spec_extra={"kind": "invoice"},
    )
)

register(
    lost_update(
        name="wb-calendar-doublebook",
        suite="workbench",
        obj=CAL + "room-a",
        field="booked",
        initial={CAL + "room-a": {"booked": 1, "capacity": 8}},
        a_fn=lambda v: (v or 0) + 2,
        b_fn=lambda v: (v or 0) + 1,
        a_label="book-offsite",
        b_label="book-standup",
        a_desc="add two bookings for the team offsite",
        b_desc="add one booking for the daily standup",
        description=(
            "Two assistants book the same room from the same occupancy read; "
            "one booking is lost unless the second recomputes."
        ),
    )
)

register(
    rmw_summary(
        name="wb-email-triage",
        suite="workbench",
        prefix=MAIL,
        field="state",
        good="archived",
        log_obj="wb/email/digest",
        description=(
            "A archives every unread message matching a filter; B appends a "
            "daily digest summarizing the states it saw."
        ),
        initial={
            MAIL + "m1": {"state": "unread"},
            MAIL + "m2": {"state": "unread"},
            MAIL + "m3": {"state": "archived"},
            "wb/email/digest": [],
        },
    )
)

register(
    sweep_derive(
        name="wb-analytics-restate",
        suite="workbench",
        prefix=ANALYTICS,
        field="region",
        good="EMEA",
        source=ANALYTICS + "r1",
        target=ANALYTICS + "r1-report",
        description=(
            "A restates mis-tagged rows to the correct region; B builds a "
            "regional report from the row it read."
        ),
        initial={
            ANALYTICS + "r1": {"region": "UNKNOWN"},
            ANALYTICS + "r2": {"region": "UNKNOWN"},
            ANALYTICS + "r3": {"region": "EMEA"},
        },
        spec_extra={"kind": "report"},
    )
)

register(
    lost_update(
        name="wb-project-rebalance",
        suite="workbench",
        obj=PROJ + "dana",
        field="workload",
        initial={PROJ + "dana": {"workload": 9, "cap": 10}},
        a_fn=lambda v: (v or 0) - 4,
        b_fn=lambda v: (v or 0) + 2,
        a_label="rebalance-off",
        b_label="assign-task",
        a_desc="subtract four, moving work off an overloaded engineer",
        b_desc="add two, assigning new tasks",
        description=(
            "A moves work off an overloaded engineer while B assigns new "
            "tasks; both compute from the workload figure they read."
        ),
    )
)
