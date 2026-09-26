"""Event-driven research backtest engine with next-bar execution."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from app.backtesting.metrics import compute_metrics, drawdown_series
from app.backtesting.portfolio import Portfolio
from app.backtesting.strategies import STRATEGY_MAP
from app.core.exceptions import BacktestError
from app.core.logging_config import get_logger
from app.lakehouse.gold import GoldLayer
from app.lakehouse.silver import SilverLayer

logger = get_logger(__name__)


class BacktestEngine:
    """
    Signal on close of bar t, execute at open (fallback close) of bar t+1.
    This avoids look-ahead bias.
    """

    def run(
        self,
        symbol: str,
        strategy_name: str,
        start_date: datetime | str | None = None,
        end_date: datetime | str | None = None,
        initial_capital: float = 10_000.0,
        transaction_fee: float = 0.001,
        slippage: float = 0.0005,
        parameters: dict | None = None,
    ) -> dict:
        if strategy_name not in STRATEGY_MAP:
            raise BacktestError(f"Unknown strategy: {strategy_name}")
        if initial_capital <= 0:
            raise BacktestError("initial_capital must be positive.")
        frame = self._load_ohlcv(symbol, start_date, end_date)
        strategy = STRATEGY_MAP[strategy_name](parameters=parameters or {})
        signals = strategy.generate_signals(frame)
        # Shift execution by one bar: act on the next open/close.
        execution_signal = signals.shift(1).fillna(0)

        portfolio = Portfolio(initial_capital=initial_capital, cash=initial_capital)
        markers: list[dict] = []
        for index in range(len(frame)):
            row = frame.iloc[index]
            exec_price = float(row["open"]) if pd.notna(row["open"]) else float(row["close"])
            signal = float(execution_signal.iloc[index])
            timestamp = row["timestamp"]
            if signal > 0:
                before = portfolio.is_long
                portfolio.buy(timestamp, exec_price, transaction_fee, slippage)
                if portfolio.is_long and not before:
                    markers.append({"timestamp": str(timestamp), "price": exec_price, "side": "buy"})
            elif signal < 0:
                trade = portfolio.sell(timestamp, exec_price, transaction_fee, slippage)
                if trade is not None:
                    markers.append({"timestamp": str(timestamp), "price": exec_price, "side": "sell"})
            portfolio.mark_to_market(timestamp, float(row["close"]))

        if portfolio.is_long:
            last = frame.iloc[-1]
            portfolio.sell(last["timestamp"], float(last["close"]), transaction_fee, slippage)

        metrics = compute_metrics(portfolio.equity_curve, portfolio.realized_trades, initial_capital)
        result = {
            "symbol": symbol.upper(),
            "strategy": strategy_name,
            "start_date": str(frame["timestamp"].iloc[0]),
            "end_date": str(frame["timestamp"].iloc[-1]),
            "initial_capital": initial_capital,
            "final_capital": metrics["final_capital"],
            "total_return": metrics["total_return"],
            "win_rate": metrics["win_rate"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "maximum_drawdown": metrics["maximum_drawdown"],
            "number_of_trades": metrics["number_of_trades"],
            "profit_factor": metrics["profit_factor"],
            "parameters": {
                **(parameters or {}),
                "transaction_fee": transaction_fee,
                "slippage": slippage,
            },
            "equity_curve": [
                {
                    "timestamp": str(point["timestamp"]),
                    "equity": float(point["equity"]),
                    "price": float(point["price"]),
                }
                for point in portfolio.equity_curve
            ],
            "drawdown": [
                {"timestamp": str(point["timestamp"]), "drawdown": float(point["drawdown"])}
                for point in drawdown_series(portfolio.equity_curve)
            ],
            "trades": [
                {
                    "entry_time": str(trade.entry_time),
                    "exit_time": str(trade.exit_time),
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "quantity": trade.quantity,
                    "pnl": trade.pnl,
                    "return_pct": trade.return_pct,
                }
                for trade in portfolio.realized_trades
            ],
            "markers": markers,
            "disclaimer": (
                "Backtesting evaluates historical hypothetical performance only. "
                "It does not prove that a strategy will be profitable in the future."
            ),
        }
        logger.info(
            "Backtest %s %s return=%.4f trades=%s",
            symbol,
            strategy_name,
            metrics["total_return"],
            metrics["number_of_trades"],
        )
        return result

    def _load_ohlcv(
        self,
        symbol: str,
        start_date: datetime | str | None,
        end_date: datetime | str | None,
    ) -> pd.DataFrame:
        frame = GoldLayer().read(symbol)
        if frame.empty:
            frame = SilverLayer().read(symbol)
        if frame.empty:
            raise BacktestError(f"No market data available for {symbol}. Run the pipeline first.")
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        if start_date:
            frame = frame[frame["timestamp"] >= pd.Timestamp(start_date, tz="UTC")]
        if end_date:
            frame = frame[frame["timestamp"] <= pd.Timestamp(end_date, tz="UTC")]
        frame = frame.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
        if len(frame) < 60:
            raise BacktestError("Need at least 60 bars to backtest this strategy.")
        return frame
