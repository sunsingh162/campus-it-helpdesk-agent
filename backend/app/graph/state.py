from typing import Annotated, Literal, Optional

from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

Category = Literal["password", "network", "hardware", "account", "unknown"]


class TicketState(TypedDict):
    """State for a single Campus IT Helpdesk ticket."""

    thread_id: str
    ticket_text: str
    category: Optional[Category]
    response: Optional[str]
    messages: Annotated[list, add_messages]
    iteration_count: int
    seen_tool_calls: list[str]
    escalated: bool
    summary: Optional[str]


def new_turn_state(thread_id: str, ticket_text: str) -> TicketState:
    """Initial/delta state for one new turn of a ticket conversation.

    Used both to start a brand-new thread and to add a follow-up turn onto
    an existing thread — in both cases classify_node re-runs on the new
    ticket_text, and the add_messages reducer appends the new HumanMessage
    (and everything the agent does after it) onto whatever history the
    checkpointer already has for this thread_id.
    """
    return {
        "thread_id": thread_id,
        "ticket_text": ticket_text,
        "category": None,
        "response": None,
        "messages": [HumanMessage(content=ticket_text)],
        "iteration_count": 0,
        "seen_tool_calls": [],
        "escalated": False,
        "summary": None,
    }
