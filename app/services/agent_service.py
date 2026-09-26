"""AI agent application service."""

from __future__ import annotations

import json

from sqlalchemy.orm import Session

from app.agent.agent import StockAnalysisAgent
from app.core.logging_config import get_logger
from app.database.repositories.conversation_repository import ConversationRepository
from app.database.session import check_database_connection

logger = get_logger(__name__)


class AgentService:
    def __init__(self) -> None:
        self.agent = StockAnalysisAgent()

    def chat(self, session_id: str, message: str, symbol: str | None, db: Session | None = None) -> dict:
        result = self.agent.reply(message, default_symbol=symbol)
        tools = result.get("tools") or []
        first_tool = tools[0] if tools else {}
        if db is not None and check_database_connection():
            try:
                ConversationRepository(db).create(
                    {
                        "session_id": session_id,
                        "user_message": message,
                        "assistant_message": result.get("assistant_message"),
                        "tool_name": ",".join(item.get("tool_name") or "" for item in tools) or None,
                        "tool_arguments": json.dumps(first_tool.get("tool_arguments"), default=str)
                        if first_tool
                        else None,
                        "tool_result": json.dumps([item.get("tool_result") for item in tools], default=str)
                        if tools
                        else None,
                    }
                )
            except Exception as exc:
                logger.warning("Unable to persist agent conversation: %s", exc)
        return {
            "session_id": session_id,
            "assistant_message": result.get("assistant_message"),
            "tools": tools,
        }

    def history(self, session_id: str, db: Session | None = None) -> list[dict]:
        if db is None or not check_database_connection():
            return []
        rows = ConversationRepository(db).list_by_session(session_id)
        return [
            {
                "id": row.id,
                "session_id": row.session_id,
                "user_message": row.user_message,
                "assistant_message": row.assistant_message,
                "tool_name": row.tool_name,
                "tool_arguments": row.tool_arguments,
                "tool_result": row.tool_result,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]
