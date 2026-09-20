"""Session 8: the supervisor orchestrator.

The direct triage -> specialist dispatch from Session 7 is replaced by a
supervisor loop: supervisor picks a specialist (or FINISH), specialist_worker
dispatches to the Session 7 specialist subgraph scoped to that category and
always returns to the supervisor — no worker routes directly to END. A
MAX_DELEGATIONS cap (checked before every supervisor LLM call) prevents
runaway delegation.

No state field is written by more than one node concurrently yet, so
there's still nothing to switch to an operator.add reducer — that becomes
necessary in Session 9, when parallel specialists write findings to
specialist_findings at the same time instead of one at a time.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from .nodes.guardrails import egress_guardrail_node
from .nodes.supervisor import (
    finalize_from_findings_node,
    route_after_supervisor,
    specialist_worker_node,
    supervisor_node,
)
from .state import TicketState
from .subgraphs.triage import build_triage_subgraph


def route_after_triage(state: TicketState) -> str:
    return "egress" if state.get("injection_blocked") else "supervisor"


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    graph = StateGraph(TicketState)

    graph.add_node("triage", build_triage_subgraph())
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("specialist_worker", specialist_worker_node)
    graph.add_node("finalize_from_findings", finalize_from_findings_node)
    graph.add_node("egress", egress_guardrail_node)

    graph.add_edge(START, "triage")
    graph.add_conditional_edges(
        "triage",
        route_after_triage,
        {"egress": "egress", "supervisor": "supervisor"},
    )
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"specialist_worker": "specialist_worker", "finalize": "finalize_from_findings"},
    )
    graph.add_edge("specialist_worker", "supervisor")
    graph.add_edge("finalize_from_findings", "egress")
    graph.add_edge("egress", END)

    return graph.compile(checkpointer=checkpointer)
