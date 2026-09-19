import os
import uuid

import pytest
from langchain_core.messages import ToolMessage

from app.graph.graph import build_graph

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; agent_node needs a live LLM call",
)


def _run(ticket_text: str) -> dict:
    graph = build_graph()
    return graph.invoke(
        {
            "thread_id": str(uuid.uuid4()),
            "ticket_text": ticket_text,
            "category": None,
            "response": None,
            "messages": [],
        }
    )


def test_account_lookup_triggers_crm_tool_call():
    result = _run("Can you check if student S1002's account is locked?")
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert any(m.name == "lookup_student_account" for m in tool_messages)
    assert result["response"]


def test_troubleshooting_query_triggers_kb_tool_call():
    result = _run("My Wi-Fi keeps dropping every few minutes on campus, how do I fix it?")
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert any(m.name == "search_kb" for m in tool_messages)
    assert result["response"]
