"""Session 1: routing skeleton.

classify_node categorizes an incoming ticket, then conditional routing sends
it to a category-specific stub handler. Each later session replaces one stub
(or adds a capability around this skeleton) without touching the others.
"""

from langgraph.graph import END, START, StateGraph

from .nodes.classify import classify_node
from .nodes.stubs import (
    account_stub,
    hardware_stub,
    network_stub,
    password_stub,
    unknown_stub,
)
from .state import TicketState

STUB_NODES = {
    "password": "password_stub",
    "network": "network_stub",
    "hardware": "hardware_stub",
    "account": "account_stub",
    "unknown": "unknown_stub",
}


def route_by_category(state: TicketState) -> str:
    return STUB_NODES[state["category"]]


def build_graph():
    graph = StateGraph(TicketState)

    graph.add_node("classify", classify_node)
    graph.add_node("password_stub", password_stub)
    graph.add_node("network_stub", network_stub)
    graph.add_node("hardware_stub", hardware_stub)
    graph.add_node("account_stub", account_stub)
    graph.add_node("unknown_stub", unknown_stub)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_by_category, list(STUB_NODES.values()))

    for node_name in STUB_NODES.values():
        graph.add_edge(node_name, END)

    return graph.compile()
