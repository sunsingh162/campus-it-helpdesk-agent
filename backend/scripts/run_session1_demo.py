"""Manual checklist runner for Session 1.

Usage: python backend/scripts/run_session1_demo.py
Requires ANTHROPIC_API_KEY in your environment or a .env file at repo root.

Prints, for one ticket per category, which category it was classified as and
the stub handler's placeholder response — the Session 1 checklist item from
the Step-by-Step Guide.
"""

import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.graph.graph import build_graph  # noqa: E402

SAMPLE_TICKETS = [
    ("password", "I forgot my student portal password and can't log in."),
    ("network", "The Wi-Fi in the library keeps disconnecting every few minutes."),
    ("hardware", "The lab printer on the 3rd floor is jammed and won't print."),
    ("account", "I graduated last semester, please delete my student account."),
]


def main() -> None:
    graph = build_graph()
    all_correct = True
    for expected_category, ticket_text in SAMPLE_TICKETS:
        result = graph.invoke(
            {
                "thread_id": str(uuid.uuid4()),
                "ticket_text": ticket_text,
                "category": None,
                "response": None,
            }
        )
        got = result["category"]
        ok = "OK" if got == expected_category else "MISMATCH"
        all_correct &= got == expected_category
        print(f"[{ok}] expected={expected_category:<9} got={got:<9} ticket={ticket_text!r}")
        print(f"        response: {result['response']}")

    print("\nSession 1 checklist:", "PASS" if all_correct else "FAIL")


if __name__ == "__main__":
    main()
