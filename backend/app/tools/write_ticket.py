"""The gated write tool. Unlike lookup_student_account/search_kb/assess_severity
(read-only), this one has a real side effect — it creates a persisted ticket
record — so it's gated behind a severity threshold, enforced here in the
tool itself (not just suggested in the system prompt), and protected by an
idempotency key so a retried call never double-writes.
"""

import os

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from .ticket_store import create_ticket_record

SEVERITY_ORDER = ["low", "medium", "high", "critical"]
WRITE_GATE_MIN_SEVERITY = os.getenv("WRITE_GATE_MIN_SEVERITY", "medium")


def meets_write_threshold(severity: str) -> bool:
    try:
        return SEVERITY_ORDER.index(severity) >= SEVERITY_ORDER.index(WRITE_GATE_MIN_SEVERITY)
    except ValueError:
        return False  # unrecognized severity string — fail closed, no ticket


@tool
def create_ticket(category: str, description: str, severity: str, student_id: str, config: RunnableConfig) -> dict:
    """Create a follow-up ticket in the IT helpdesk system for staff to action.

    Only call this for issues that genuinely need staff follow-up beyond a
    self-service answer — e.g. an account unlock, a hardware dispatch, or
    anything you assessed as medium severity or higher via assess_severity.
    Don't call this for routine issues fully resolved by a KB article.

    category: one of password, network, hardware, account.
    severity: one of low, medium, high, critical — must match what
        assess_severity returned for this issue; tickets below the
        helpdesk's configured severity threshold are not filed.
    student_id: the student's ID if known, else an empty string.

    Returns the created ticket's id, or the id of an already-existing
    ticket if this exact (conversation, category) was already filed.
    """
    if not meets_write_threshold(severity):
        return {
            "created": False,
            "reason": (
                f"Severity '{severity}' is below this helpdesk's ticket-filing "
                f"threshold ('{WRITE_GATE_MIN_SEVERITY}') — no ticket needed."
            ),
        }

    thread_id = (config or {}).get("configurable", {}).get("thread_id") or "unknown-thread"
    return create_ticket_record(
        thread_id=thread_id,
        category=category,
        description=description,
        severity=severity,
        student_id=student_id or None,
    )
