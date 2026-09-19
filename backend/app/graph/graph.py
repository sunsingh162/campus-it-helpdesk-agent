"""Session 2: tool binding.

The category-specific stub handlers from Session 1 are replaced by a single
agent_node that can call bound tools (mock CRM lookup + mock KB search) via
a LangGraph ToolNode, looping between agent and tools until it has enough
information to answer. classify_node is unchanged — it still sets the
category used to steer the agent's system prompt.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .nodes.agent import TOOLS, agent_node, finalize_node
from .nodes.classify import classify_node
from .state import TicketState


def build_graph():
    graph = StateGraph(TicketState)

    graph.add_node("classify", classify_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "agent")
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: "finalize"},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)

    return graph.compile()
