"""Session 7: multi-agent topologies.

The flat graph is now composed from two independently compiled subgraphs —
triage (guardrails + classify + summarize) and specialist (the ReAct loop)
— each buildable and invocable entirely on its own (see
tests/test_session7_subgraphs.py). The master graph adds each compiled
subgraph as a single node and wires them together with an egress guardrail
pass and a routing decision for the injection-blocked case.

No state field is written by more than one node yet, so there's nothing to
switch to an operator.add reducer in this session — that becomes necessary
in Session 9, when parallel specialists write findings to a shared field
concurrently.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from .nodes.guardrails import egress_guardrail_node
from .state import TicketState
from .subgraphs.specialist import build_specialist_subgraph
from .subgraphs.triage import build_triage_subgraph


def route_after_triage(state: TicketState) -> str:
    return "egress" if state.get("injection_blocked") else "specialist"


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    graph = StateGraph(TicketState)

    graph.add_node("triage", build_triage_subgraph())
    graph.add_node("specialist", build_specialist_subgraph())
    graph.add_node("egress", egress_guardrail_node)

    graph.add_edge(START, "triage")
    graph.add_conditional_edges(
        "triage",
        route_after_triage,
        {"egress": "egress", "specialist": "specialist"},
    )
    graph.add_edge("specialist", "egress")
    graph.add_edge("egress", END)

    return graph.compile(checkpointer=checkpointer)
