"""AI agent endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_agent_service, get_db
from app.schemas.agent import ChatRequest
from app.schemas.common import ok
from app.services.agent_service import AgentService

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/chat")
def agent_chat(
    payload: ChatRequest,
    db: Session | None = Depends(get_db),
    service: AgentService = Depends(get_agent_service),
) -> dict:
    result = service.chat(payload.session_id, payload.message, payload.symbol, db=db)
    return ok(result)


@router.get("/history/{session_id}")
def agent_history(
    session_id: str,
    db: Session | None = Depends(get_db),
    service: AgentService = Depends(get_agent_service),
) -> dict:
    return ok(service.history(session_id, db=db))
