import os
import uuid

import pytest

from app.graph.graph import build_graph

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; classify_node needs a live LLM call",
)

SAMPLE_TICKETS = {
    "password": "I forgot my student portal password and can't log in.",
    "network": "The Wi-Fi in the library keeps disconnecting every few minutes.",
    "hardware": "The lab printer on the 3rd floor is jammed and won't print.",
    "account": "I graduated last semester, please delete my student account.",
}


@pytest.mark.parametrize("expected_category,ticket_text", SAMPLE_TICKETS.items())
def test_ticket_routes_to_correct_stub(expected_category, ticket_text):
    graph = build_graph()
    result = graph.invoke(
        {
            "thread_id": str(uuid.uuid4()),
            "ticket_text": ticket_text,
            "category": None,
            "response": None,
        }
    )
    assert result["category"] == expected_category
    assert result["response"].startswith(f"[{expected_category.capitalize()} stub]")
