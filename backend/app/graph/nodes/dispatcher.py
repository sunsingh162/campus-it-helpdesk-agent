import logging
import os
import time
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send
from pydantic import BaseModel, Field

from ..state import TicketState
from ..subgraphs.specialist import build_specialist_subgraph

logger = logging.getLogger("dispatcher")

_dispatcher_llm: ChatAnthropic | None = None
_synthesizer_llm: ChatAnthropic | None = None
_specialist_graph = None


def get_dispatcher_llm() -> ChatAnthropic:
    global _dispatcher_llm
    if _dispatcher_llm is None:
        _dispatcher_llm = ChatAnthropic(model=os.getenv("AGENT_MODEL", "claude-sonnet-5"))
    return _dispatcher_llm


def get_synthesizer_llm() -> ChatAnthropic:
    global _synthesizer_llm
    if _synthesizer_llm is None:
        _synthesizer_llm = ChatAnthropic(model=os.getenv("AGENT_MODEL", "claude-sonnet-5"))
    return _synthesizer_llm


def get_specialist_graph():
    global _specialist_graph
    if _specialist_graph is None:
        _specialist_graph = build_specialist_subgraph()
    return _specialist_graph


class RelevantCategories(BaseModel):
    categories: list[Literal["password", "network", "hardware", "account"]] = Field(
        description=(
            "1 to 3 categories genuinely relevant to this ticket. Most tickets "
            "need just one; only list more than one when the ticket clearly "
            "raises multiple distinct issues."
        )
    )


DETERMINE_CATEGORIES_PROMPT = (
    "You are the dispatcher for a campus IT helpdesk multi-agent system. Read "
    "the ticket and decide which specialist categories (password, network, "
    "hardware, account) are genuinely relevant — so they can all be "
    "investigated in parallel."
)


def determine_categories_node(state: TicketState) -> dict:
    prompt = [
        SystemMessage(content=DETERMINE_CATEGORIES_PROMPT),
        HumanMessage(content=f"Ticket: {state['ticket_text']}\nInitial classification: {state.get('category')}"),
    ]
    result = get_dispatcher_llm().with_structured_output(RelevantCategories).invoke(prompt)
    logger.info("[dispatcher] relevant categories: %s", result.categories)
    return {"dispatch_categories": result.categories}


def dispatch_to_specialists(state: TicketState) -> list[Send]:
    """Conditional-edge path function: returns a Send per relevant category,
    all fanned out in a single superstep so they run concurrently."""
    categories = state.get("dispatch_categories") or []
    return [
        Send("parallel_specialist_worker", {**state, "target_category": category})
        for category in categories
    ]


def parallel_specialist_worker_node(state: TicketState) -> dict:
    """One instance of this runs per Send target, concurrently with the
    others in the same superstep. Reuses the Session 7 specialist subgraph
    (agent/tools/finalize, with its own iteration cap) scoped to this
    branch's target_category, resetting the ReAct loop's own bookkeeping so
    the circuit breaker applies per specialist call, not cumulatively.
    """
    category = state["target_category"]
    prior_messages = state.get("messages") or []
    dispatch_message = HumanMessage(
        content=(
            f"As the {category} specialist, investigate this ticket and give "
            f"your finding: {state['ticket_text']}"
        )
    )
    sub_state = {
        **state,
        "category": category,
        "iteration_count": 0,
        "seen_tool_calls": [],
        "escalated": False,
        "response": None,
        "messages": prior_messages + [dispatch_message],
    }

    start = time.monotonic()
    result = get_specialist_graph().invoke(sub_state)
    elapsed = time.monotonic() - start

    logger.info(
        "[parallel] %s specialist finding (%.2fs, concurrent dispatch): %s",
        category,
        elapsed,
        result["response"],
    )

    new_messages = result["messages"][len(prior_messages):]

    return {
        "specialist_findings": {category: result["response"]},
        "messages": new_messages,
    }


class SynthesizedResponse(BaseModel):
    response: str = Field(
        description="One unified, customer-facing response synthesizing all relevant specialist findings."
    )


SYNTHESIZER_SYSTEM_PROMPT = (
    "You are the final response synthesizer for a campus IT helpdesk. Multiple "
    "specialists investigated this ticket in parallel and each produced a "
    "finding. Read all findings, filter out anything irrelevant or redundant, "
    "and write ONE unified, coherent response to the student that addresses "
    "everything relevant without repeating the specialists' internal "
    "reasoning verbatim."
)


def synthesizer_node(state: TicketState) -> dict:
    """Filters specialist_findings down to this turn's dispatch_categories
    (dropping any stale entry from an earlier, unrelated turn — see
    state.merge_findings for why the dict itself is never reset), then
    synthesizes one unified response.
    """
    all_findings = state.get("specialist_findings") or {}
    relevant_categories = state.get("dispatch_categories") or []
    relevant_findings = {
        category: text for category, text in all_findings.items() if category in relevant_categories and text
    }

    findings_text = "\n\n".join(f"[{category}] {text}" for category, text in relevant_findings.items())
    prompt = [
        SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
        HumanMessage(content=f"Ticket: {state['ticket_text']}\n\nSpecialist findings:\n{findings_text}"),
    ]
    result = get_synthesizer_llm().with_structured_output(SynthesizedResponse).invoke(prompt)
    return {"response": result.response}
