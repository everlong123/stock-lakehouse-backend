"""Research AI agent with tool calling and an offline deterministic fallback."""

from __future__ import annotations

import json
import re
from typing import Any

from app.agent.llm_client import LLMClient
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tool_registry import execute_tool, openai_tools
from app.core.constants import SUPPORTED_SYMBOLS
from app.core.exceptions import AgentError, StockLakehouseError
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class StockAnalysisAgent:
    """Never fabricates prices, indicators, forecasts, or backtest metrics."""

    def __init__(self) -> None:
        self.llm = LLMClient()

    def reply(self, user_message: str, default_symbol: str | None = None) -> dict[str, Any]:
        if self.llm.is_configured():
            try:
                return self._llm_reply(user_message, default_symbol)
            except Exception as exc:
                logger.warning("LLM path failed (%s). Falling back to local tool router.", exc)
        return self._local_reply(user_message, default_symbol)

    def _llm_reply(self, user_message: str, default_symbol: str | None) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        first = self.llm.chat(messages, openai_tools())
        tool_trace: list[dict[str, Any]] = []
        if first.get("tool_calls"):
            messages.append(
                {
                    "role": "assistant",
                    "content": first.get("content") or "",
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {"name": call["name"], "arguments": call["arguments"]},
                        }
                        for call in first["tool_calls"]
                    ],
                }
            )
            for call in first["tool_calls"]:
                try:
                    args = json.loads(call["arguments"] or "{}")
                    if "symbol" not in args and default_symbol:
                        args["symbol"] = default_symbol
                    result = execute_tool(call["name"], args)
                    error = None
                except StockLakehouseError as exc:
                    result = {"error": exc.message}
                    error = exc.message
                    args = {}
                tool_trace.append(
                    {"tool_name": call["name"], "tool_arguments": args, "tool_result": result, "error": error}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": call["name"],
                        "content": json.dumps(result, default=str),
                    }
                )
            second = self.llm.chat(messages, openai_tools())
            content = second.get("content") or "Đã lấy dữ liệu từ tool nhưng mô hình không trả lời được."
        else:
            content = first.get("content") or ""
            if not content:
                raise AgentError("LLM returned an empty response.")
        return {"assistant_message": content, "tools": tool_trace}

    def _local_reply(self, user_message: str, default_symbol: str | None) -> dict[str, Any]:
        """Deterministic router so the thesis demo works without an API key."""
        text = user_message.lower()
        symbol = _extract_symbol(user_message, default_symbol)
        plan: list[tuple[str, dict[str, Any]]] = []
        if any(word in text for word in ["backtest", "chiến lược", "ma crossover", "rsi strategy"]):
            strategy = "rsi_strategy" if "rsi" in text and "crossover" not in text else "ma_crossover"
            plan.append(("run_backtest", {"symbol": symbol, "strategy": strategy}))
        elif any(word in text for word in ["so sánh", "compare", "mae", "rmse"]):
            plan.append(("compare_models", {"symbol": symbol}))
        elif any(word in text for word in ["dự báo", "forecast", "lstm", "arima", "linear"]):
            model = "lstm" if "lstm" in text else "arima" if "arima" in text else "linear_regression"
            plan.append(("forecast_stock", {"symbol": symbol, "model_name": model}))
        elif any(word in text for word in ["rsi", "macd", "sma", "ema", "bollinger", "chỉ báo", "kỹ thuật"]):
            plan.append(("query_stock_data", {"symbol": symbol}))
            plan.append(("calculate_indicators", {"symbol": symbol}))
        else:
            plan.append(("get_market_summary", {"symbol": symbol}))

        tools: list[dict[str, Any]] = []
        collected: dict[str, Any] = {}
        for name, args in plan:
            try:
                result = execute_tool(name, args)
                error = None
            except StockLakehouseError as exc:
                result = {"error": exc.message}
                error = exc.message
            tools.append({"tool_name": name, "tool_arguments": args, "tool_result": result, "error": error})
            collected[name] = result
        message = _render_local_answer(user_message, collected)
        return {"assistant_message": message, "tools": tools}


def _extract_symbol(text: str, default_symbol: str | None) -> str:
    upper = text.upper()
    for symbol in SUPPORTED_SYMBOLS:
        if re.search(rf"\b{symbol}\b", upper):
            return symbol
    match = re.search(r"\b([A-Z]{1,5})\b", upper)
    if match and match.group(1) not in {"RSI", "MACD", "SMA", "EMA", "LSTM", "MA"}:
        return match.group(1)
    return (default_symbol or "AAPL").upper()


def _render_local_answer(question: str, collected: dict[str, Any]) -> str:
    lines = [
        "Đây là câu trả lời dựa trên dữ liệu thật từ backend (tool calling).",
        "Kết quả chỉ phục vụ nghiên cứu học thuật, không phải khuyến nghị đầu tư.",
        "",
    ]
    for name, result in collected.items():
        if isinstance(result, dict) and result.get("error"):
            lines.append(f"Tool `{name}` thất bại: {result['error']}")
            continue
        if name == "calculate_indicators":
            lines.append(
                f"RSI={result.get('rsi')}, MACD={result.get('macd')}, "
                f"SMA20={result.get('sma_20')}, SMA50={result.get('sma_50')}."
            )
            summary = result.get("summary") or {}
            for key, value in summary.items():
                lines.append(f"{key}: {value}")
        elif name == "run_backtest":
            lines.append(
                f"Backtest {result.get('strategy')} cho {result.get('symbol')}: "
                f"Total Return={result.get('total_return'):.4f}, "
                f"Win Rate={result.get('win_rate'):.4f}, "
                f"Sharpe={result.get('sharpe_ratio'):.4f}, "
                f"Max Drawdown={result.get('maximum_drawdown'):.4f}, "
                f"Trades={result.get('number_of_trades')}."
            )
            lines.append(result.get("disclaimer", ""))
        elif name == "forecast_stock":
            metrics = result.get("metrics") or {}
            lines.append(
                f"Dự báo {result.get('model_name')} cho {result.get('symbol')}: "
                f"MAE={metrics.get('mae')}, RMSE={metrics.get('rmse')}, "
                f"MAPE={metrics.get('mape')}, Directional Accuracy={metrics.get('directional_accuracy')}."
            )
            if result.get("latest_predicted") is not None:
                lines.append(f"Giá trị dự báo gần nhất: {result['latest_predicted']:.4f}.")
            lines.append(result.get("disclaimer", ""))
        elif name == "compare_models":
            lines.append(f"So sánh mô hình cho {result.get('symbol')}:")
            for item in result.get("models") or []:
                lines.append(
                    f"- {item.get('model_name')}: MAE={item.get('mae')}, RMSE={item.get('rmse')}, "
                    f"MAPE={item.get('mape')}, DA={item.get('directional_accuracy')}"
                )
            if not result.get("models"):
                lines.append("Chưa có mô hình nào được train. Hãy train model trước.")
        elif name in {"query_stock_data", "get_market_summary"}:
            close = result.get("last_close") or result.get("close")
            change = result.get("change_pct")
            lines.append(
                f"{result.get('symbol')} close={close}, change={change}, volume={result.get('volume')}."
            )
    return "\n".join(lines)
