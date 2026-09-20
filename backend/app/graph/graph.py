"""Session 6: guardrails.

An ingress guardrail node is now the graph's entry point — it redacts PII
and checks for prompt injection before classify (or any other LLM call)
ever sees the ticket text. A detected injection routes straight to
blocked_response, spending zero LLM tokens. An egress guardrail node runs
after finalize to scan/redact the outgoing response.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .nodes.agent import TOOLS, agent_node, finalize_node, route_after_agent
from .nodes.classify import classify_node
from .nodes.context import summarize_node
from .nodes.guardrails import (
    blocked_response_node,
    egress_guardrail_node,
    ingress_guardrail_node,
    route_after_ingress,
)
from .state import TicketState


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    graph = StateGraph(TicketState)

    graph.add_node("ingress", ingress_guardrail_node)
    graph.add_node("blocked_response", blocked_response_node)
    graph.add_node("classify", classify_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("finalize", finalize_node)
    graph.add_node("egress", egress_guardrail_node)

    graph.add_edge(START, "ingress")
    graph.add_conditional_edges(
        "ingress",
        route_after_ingress,
        {"blocked_response": "blocked_response", "classify": "classify"},
    )
    graph.add_edge("blocked_response", END)
    graph.add_edge("classify", "summarize")
    graph.add_edge("summarize", "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "finalize": "finalize"},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", "egress")
    graph.add_edge("egress", END)

    return graph.compile(checkpointer=checkpointer)
