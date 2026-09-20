from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

from .graph.runner import run_ticket  # noqa: E402

app = FastAPI(title="Campus IT Helpdesk Agent")


class TicketRequest(BaseModel):
    thread_id: str
    message: str


class TicketResponse(BaseModel):
    thread_id: str
    category: Optional[str]
    response: Optional[str]
    escalated: bool


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/tickets/run", response_model=TicketResponse)
def run(payload: TicketRequest) -> TicketResponse:
    result = run_ticket(payload.thread_id, payload.message)
    return TicketResponse(
        thread_id=payload.thread_id,
        category=result.get("category"),
        response=result.get("response"),
        escalated=result.get("escalated", False),
    )
