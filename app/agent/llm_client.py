"""OpenAI-compatible LLM client with an offline heuristic fallback."""

from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.core.exceptions import AgentError
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class LLMClient:
    """Call an OpenAI-compatible chat API. If no key is set, return None so the local router runs."""

    def is_configured(self) -> bool:
        return bool(settings.openai_api_key.strip())

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.is_configured():
            raise AgentError("OPENAI_API_KEY is not configured.")
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                timeout=settings.openai_timeout_seconds,
            )
            response = client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2,
            )
            message = response.choices[0].message
            tool_calls = []
            if message.tool_calls:
                for call in message.tool_calls:
                    tool_calls.append(
                        {
                            "id": call.id,
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        }
                    )
            return {"content": message.content or "", "tool_calls": tool_calls}
        except Exception as exc:
            logger.exception("LLM request failed")
            raise AgentError(f"LLM request failed: {exc}") from exc
