"""Session 9: parallel specialists.

The master graph's dispatch mechanism evolves from Session 8's sequential
supervisor loop to a real parallel fan-out: determine_categories decides the
relevant category set once, dispatch_to_specialists (a conditional-edge path
function) returns one Send per category — all of which run concurrently in
a single superstep — and a synthesizer node reads every finding once they've
all completed, filters out anything not relevant to this turn, and produces
one unified response.

specialist_findings needed a real reducer (merge_findings, in state.py) for
this to be safe: multiple parallel branches now write to it in the same
superstep, which a plain dict field can't handle without raising a
LangGraph InvalidUpdateError.

Session 8's supervisor_node/specialist_worker_node/finalize_from_findings_node
are untouched and still fully tested (tests/test_session8_supervisor.py) —
this session's master graph just wires the newer parallel path in instead,
the same way Session 8 wired around Session 7's plain specialist dispatch
without deleting the specialist subgraph it depends on.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from .nodes.dispatcher import (
    determine_categories_node,
    dispatch_to_specialists,
    parallel_specialist_worker_node,
    synthesizer_node,
)
from .nodes.guardrails import egress_guardrail_node
from .state import TicketState
from .subgraphs.triage import build_triage_subgraph


def route_after_triage(state: TicketState) -> str:
    return "egress" if state.get("injection_blocked") else "determine_categories"


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    graph = StateGraph(TicketState)

    graph.add_node("triage", build_triage_subgraph())
    graph.add_node("determine_categories", determine_categories_node)
    graph.add_node("parallel_specialist_worker", parallel_specialist_worker_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("egress", egress_guardrail_node)

    graph.add_edge(START, "triage")
    graph.add_conditional_edges(
        "triage",
        route_after_triage,
        {"egress": "egress", "determine_categories": "determine_categories"},
    )
    graph.add_conditional_edges("determine_categories", dispatch_to_specialists, ["parallel_specialist_worker"])
    graph.add_edge("parallel_specialist_worker", "synthesizer")
    graph.add_edge("synthesizer", "egress")
    graph.add_edge("egress", END)

    return graph.compile(checkpointer=checkpointer)
