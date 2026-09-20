"""Session 3: the ReAct loop.

agent_node is now a real ReAct loop with a hard MAX_ITERATIONS cap and
duplicate-tool-call fingerprinting: instead of relying on the LLM to decide
when to stop, the graph enforces a circuit breaker and escalates. Routing
after the agent is custom (route_after_agent) rather than the prebuilt
tools_condition, since it also has to check the escalated flag.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .nodes.agent import TOOLS, agent_node, finalize_node, route_after_agent
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
        route_after_agent,
        {"tools": "tools", "finalize": "finalize"},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)

    return graph.compile()
