"""Entry points a server (or a script) calls to run a ticket turn.

Every call goes through the same compiled, checkpointed graph and always
passes thread_id in its config — that's what lets a conversation survive a
process restart: a new process calling run_ticket()/stream_ticket() with the
same thread_id resumes from whatever the checkpointer last persisted for it.
"""

from langchain_core.messages import HumanMessage

from .graph import build_graph
from .persistence import get_checkpointer
from .state import new_turn_state

_graph = None


def get_compiled_graph():
    global _graph
    if _graph is None:
        _graph = build_graph(checkpointer=get_checkpointer())
    return _graph


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _build_turn_input(graph, thread_id: str, ticket_text: str) -> dict:
    config = _config(thread_id)
    snapshot = graph.get_state(config)
    is_new_thread = not snapshot.values.get("messages")

    if is_new_thread:
        return new_turn_state(thread_id, ticket_text)

    # Follow-up turn: only send the delta. category/response/iteration_count/
    # seen_tool_calls/escalated reset fresh for this new question; messages
    # history is preserved by the checkpointer and add_messages appends to it.
    return {
        "ticket_text": ticket_text,
        "category": None,
        "response": None,
        "messages": [HumanMessage(content=ticket_text)],
        "iteration_count": 0,
        "seen_tool_calls": [],
        "escalated": False,
    }


def run_ticket(thread_id: str, ticket_text: str) -> dict:
    graph = get_compiled_graph()
    config = _config(thread_id)
    turn_input = _build_turn_input(graph, thread_id, ticket_text)
    return graph.invoke(turn_input, config)


def stream_ticket(thread_id: str, ticket_text: str):
    graph = get_compiled_graph()
    config = _config(thread_id)
    turn_input = _build_turn_input(graph, thread_id, ticket_text)
    yield from graph.stream(turn_input, config, stream_mode="values")
