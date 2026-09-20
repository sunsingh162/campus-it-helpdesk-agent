"""Session 4: persistence & threading.

The graph shape is unchanged from Session 3 — build_graph now optionally
takes a checkpointer so the compiled graph can persist/resume state per
thread_id. Callers that don't need persistence (most of the existing unit
tests) can still call build_graph() with no arguments.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .nodes.agent import TOOLS, agent_node, finalize_node, route_after_agent
from .nodes.classify import classify_node
from .state import TicketState


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    graph = StateGraph(TicketState)

    graph.add_node("classify", classify_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "finalize": "finalize"},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)
