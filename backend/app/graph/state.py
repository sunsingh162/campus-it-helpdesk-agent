from typing import Annotated, Literal, Optional

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
