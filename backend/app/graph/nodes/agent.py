import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from ...tools.crm import lookup_student_account
from ...tools.kb import search_kb
from ..state import TicketState

TOOLS = [lookup_student_account, search_kb]

_llm: ChatAnthropic | None = None


def get_agent_llm() -> ChatAnthropic:
    global _llm
    if _llm is None:
        # claude-sonnet-5 (and newer Claude 5 family models) reject an explicit
        # `temperature` value — deprecated for this model, so we omit it.
        _llm = ChatAnthropic(model=os.getenv("AGENT_MODEL", "claude-sonnet-5"))
    return _llm


SYSTEM_PROMPT = (
    "You are a campus IT helpdesk assistant handling a '{category}' ticket. "
    "Use lookup_student_account to check the student's account/device status "
    "when a student ID is available or relevant, and use search_kb to find "
    "troubleshooting steps before answering. Give one concise, helpful final "
    "response once you have what you need — do not call the same tool with "
    "the same arguments more than once."
)


def agent_node(state: TicketState) -> dict:
    llm_with_tools = get_agent_llm().bind_tools(TOOLS)
    existing_messages = state.get("messages") or []

    if not existing_messages:
        existing_messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(category=state["category"])),
            HumanMessage(content=state["ticket_text"]),
        ]
        ai_message = llm_with_tools.invoke(existing_messages)
        return {"messages": existing_messages + [ai_message]}

    ai_message = llm_with_tools.invoke(existing_messages)
    return {"messages": [ai_message]}


def finalize_node(state: TicketState) -> dict:
    last_message = state["messages"][-1]
    return {"response": last_message.content}
