"""Persistence helpers for AI agent conversations."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models.agent_conversation import AgentConversation


class ConversationRepository:
    """CRUD operations for agent_conversations."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, payload: dict[str, Any]) -> AgentConversation:
        record = AgentConversation(**payload)
        self.session.add(record)
        self.session.flush()
        return record

    def list_by_session(self, session_id: str, limit: int = 100) -> list[AgentConversation]:
        stmt = (
            select(AgentConversation)
            .where(AgentConversation.session_id == session_id)
            .order_by(AgentConversation.created_at.asc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt).all())
