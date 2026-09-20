"""Session 5: context management.

A summarize node runs once per turn, right after classify and before the
agent starts its ReAct loop, so a trimmed/summarized history is what the
(more expensive) agent model actually pays for on this turn — not just
trimmed in time for the *next* one.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .nodes.agent import TOOLS, agent_node, finalize_node, route_after_agent
from .nodes.classify import classify_node
from .nodes.context import summarize_node
from .state import TicketState


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    graph = StateGraph(TicketState)

    graph.add_node("classify", classify_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "summarize")
    graph.add_edge("summarize", "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "finalize": "finalize"},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)
