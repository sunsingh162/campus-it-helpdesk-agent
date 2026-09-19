from typing import Literal, Optional

from typing_extensions import TypedDict

Category = Literal["password", "network", "hardware", "account", "unknown"]


class TicketState(TypedDict):
    """State for a single Campus IT Helpdesk ticket."""

    thread_id: str
    ticket_text: str
    category: Optional[Category]
    response: Optional[str]
