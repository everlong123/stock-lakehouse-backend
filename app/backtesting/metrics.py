"""Backtest performance metrics. Historical only; not a future-profit guarantee."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.backtesting.trade import Trade


def compute_metrics(
    equity_curve: list[dict],
    trades: list[Trade],
    initial_capital: float,
    periods_per_year: float = 252.0,
) -> dict[str, float]:
    """Compute academic backtest statistics from an equity curve and closed trades."""
    if not equity_curve:
        return {
            "total_return": 0.0,
            "final_capital": float(initial_capital),
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
            "maximum_drawdown": 0.0,
            "number_of_trades": 0,
            "profit_factor": 0.0,
        }
    equity = pd.Series([point["equity"] for point in equity_curve], dtype=float)
    final_capital = float(equity.iloc[-1])
    total_return = (final_capital / initial_capital) - 1.0
    returns = equity.pct_change().dropna()
    sharpe = 0.0
    if not returns.empty and returns.std() > 0:
        sharpe = float(np.sqrt(periods_per_year) * returns.mean() / returns.std())
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max.replace(0, np.nan)
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0

    pnls = [trade.pnl for trade in trades]
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl < 0]
    win_rate = (len(wins) / len(pnls)) if pnls else 0.0
    gross_profit = float(sum(wins)) if wins else 0.0
    gross_loss = float(abs(sum(losses))) if losses else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    if profit_factor == float("inf"):
        profit_factor = 99.99
    return {
        "total_return": float(total_return),
        "final_capital": final_capital,
        "win_rate": float(win_rate),
        "sharpe_ratio": float(sharpe),
        "maximum_drawdown": float(max_dd),
        "number_of_trades": int(len(trades)),
        "profit_factor": float(profit_factor),
    }


def drawdown_series(equity_curve: list[dict]) -> list[dict]:
    if not equity_curve:
        return []
    equity = pd.Series([point["equity"] for point in equity_curve], dtype=float)
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max.replace(0, np.nan)
    return [
        {
            "timestamp": equity_curve[i]["timestamp"],
            "drawdown": float(dd.iloc[i]) if pd.notna(dd.iloc[i]) else 0.0,
        }
        for i in range(len(equity_curve))
    ]
