"""Walk-forward validation for robust backtesting methodology.

Walk-forward validation addresses the key weakness of simple train-test split:
- Avoids overfitting to a specific time period
- Simulates real-world model deployment where you retrain periodically
- Provides more realistic performance estimates

The methodology:
1. Divide data into rolling windows
2. Each window: train on "train_period", test on "test_period"
3. Move window forward and repeat
4. Aggregate results across all windows
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from app.backtesting.engine import BacktestEngine
from app.backtesting.metrics import compute_metrics
from app.backtesting.portfolio import Portfolio
from app.backtesting.strategies import STRATEGY_MAP
from app.core.exceptions import BacktestError
from app.core.logging_config import get_logger
from app.lakehouse.gold import GoldLayer
from app.lakehouse.silver import SilverLayer

logger = get_logger(__name__)


@dataclass
class WalkForwardWindow:
    """Single walk-forward window result."""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_return: float | None = None
    test_return: float | None = None
    test_sharpe: float | None = None
    test_max_drawdown: float | None = None
    test_win_rate: float | None = None
    test_num_trades: int = 0
    passed: bool = True
    error: str | None = None


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward validation results."""
    symbol: str
    strategy_name: str
    total_windows: int
    passed_windows: int
    failed_windows: int

    # Train period metrics
    avg_train_return: float | None = None
    train_return_std: float | None = None

    # Test period metrics (more important for evaluation)
    avg_test_return: float | None = None
    test_return_std: float | None = None
    avg_test_sharpe: float | None = None
    avg_test_max_drawdown: float | None = None
    avg_test_win_rate: float | None = None
    total_test_trades: int = 0

    # Stability metrics
    return_consistency: float | None = None  # Ratio of positive windows
    overfitting_ratio: float | None = None  # train_return / test_return

    windows: list[WalkForwardWindow] = field(default_factory=list)

    disclaimer: str = ""


class WalkForwardValidator:
    """
    Walk-forward validation engine.

    Methodology:
    - Rolling window approach with fixed train/test periods
    - For each window: train strategy parameters on train period, test on test period
    - This mimics real-world deployment where you continuously retrain
    """

    def __init__(
        self,
        train_period_days: int = 180,  # 6 months training
        test_period_days: int = 30,     # 1 month testing
        step_days: int = 20,             # Move forward 20 days each iteration
        min_train_bars: int = 60,
        min_test_bars: int = 20,
    ):
        self.train_period_days = train_period_days
        self.test_period_days = test_period_days
        self.step_days = step_days
        self.min_train_bars = min_train_bars
        self.min_test_bars = min_test_bars

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
    ) -> dict[str, Any]:
        """
        Run walk-forward validation.

        Args:
            symbol: Stock symbol
            strategy_name: Strategy name (e.g., 'ma_crossover')
            start_date: Start of overall data period
            end_date: End of overall data period
            initial_capital: Starting capital
            transaction_fee: Fee as fraction (e.g., 0.001 = 0.1%)
            slippage: Slippage as fraction
            parameters: Strategy parameters

        Returns:
            Aggregated results and per-window details
        """
        if strategy_name not in STRATEGY_MAP:
            raise BacktestError(f"Unknown strategy: {strategy_name}")

        # Load data
        frame = self._load_data(symbol, start_date, end_date)
        if frame.empty:
            raise BacktestError(f"No market data available for {symbol}.")

        frame = frame.sort_values("timestamp").reset_index(drop=True)

        # Generate walk-forward windows
        windows = self._generate_windows(frame)

        if not windows:
            raise BacktestError("Not enough data to create walk-forward windows.")

        logger.info(
            f"Walk-forward validation for {symbol}/{strategy_name}: "
            f"{len(windows)} windows"
        )

        # Run backtest for each window
        results: list[WalkForwardWindow] = []
        train_returns = []
        test_returns = []
        test_sharpes = []
        test_drawdowns = []
        test_win_rates = []
        test_trades_count = 0

        for i, window in enumerate(windows):
            try:
                # Extract train and test data
                train_data = frame[
                    (frame["timestamp"] >= window.train_start) &
                    (frame["timestamp"] <= window.train_end)
                ].copy()

                test_data = frame[
                    (frame["timestamp"] >= window.test_start) &
                    (frame["timestamp"] <= window.test_end)
                ].copy()

                if len(train_data) < self.min_train_bars or len(test_data) < self.min_test_bars:
                    window.passed = False
                    window.error = "Insufficient bars in window"
                    results.append(window)
                    continue

                # Train phase: find best parameters on train data
                train_result = self._train_on_window(
                    train_data, strategy_name, initial_capital, transaction_fee, slippage, parameters
                )
                window.train_return = train_result.get("total_return")

                if train_result is not None:
                    train_returns.append(train_result["total_return"])

                # Test phase: apply trained parameters to test data
                test_result = self._test_on_window(
                    test_data, strategy_name, initial_capital, transaction_fee, slippage, parameters
                )

                if test_result:
                    window.test_return = test_result.get("total_return")
                    window.test_sharpe = test_result.get("sharpe_ratio")
                    window.test_max_drawdown = test_result.get("maximum_drawdown")
                    window.test_win_rate = test_result.get("win_rate")
                    window.test_num_trades = test_result.get("number_of_trades", 0)

                    test_returns.append(test_result["total_return"])
                    test_sharpes.append(test_result["sharpe_ratio"])
                    test_drawdowns.append(test_result["maximum_drawdown"])
                    test_win_rates.append(test_result["win_rate"])
                    test_trades_count += test_result["number_of_trades"]
                else:
                    window.passed = False
                    window.error = "Test backtest failed"

            except Exception as exc:
                window.passed = False
                window.error = str(exc)
                logger.warning(f"Window {i} failed: {exc}")

            results.append(window)

        # Compute aggregated metrics
        passed = sum(1 for w in results if w.passed)
        failed = len(results) - passed

        # Calculate stability
        positive_test = sum(1 for r in test_returns if r is not None and r > 0)
        consistency = positive_test / len(test_returns) if test_returns else 0

        # Calculate overfitting ratio
        avg_train = sum(r for r in train_returns if r is not None) / len(train_returns) if train_returns else 0
        avg_test = sum(r for r in test_returns if r is not None) / len(test_returns) if test_returns else 0
        overfitting_ratio = avg_train / avg_test if avg_test and avg_test != 0 else None

        aggregated = WalkForwardResult(
            symbol=symbol.upper(),
            strategy_name=strategy_name,
            total_windows=len(results),
            passed_windows=passed,
            failed_windows=failed,
            avg_train_return=avg_train if train_returns else None,
            train_return_std=_std(train_returns) if train_returns else None,
            avg_test_return=avg_test if test_returns else None,
            test_return_std=_std(test_returns) if test_returns else None,
            avg_test_sharpe=sum(s for s in test_sharpes if s is not None) / len(test_sharpes) if test_sharpes else None,
            avg_test_max_drawdown=sum(d for d in test_drawdowns if d is not None) / len(test_drawdowns) if test_drawdowns else None,
            avg_test_win_rate=sum(w for w in test_win_rates if w is not None) / len(test_win_rates) if test_win_rates else None,
            total_test_trades=test_trades_count,
            return_consistency=consistency,
            overfitting_ratio=overfitting_ratio,
            windows=results,
            disclaimer=self._build_disclaimer(),
        )

        return self._to_dict(aggregated)

    def _generate_windows(self, frame: pd.DataFrame) -> list[WalkForwardWindow]:
        """Generate walk-forward window definitions."""
        windows = []

        start_ts = frame["timestamp"].min()
        end_ts = frame["timestamp"].max()

        window_id = 0
        current_train_end = start_ts + pd.Timedelta(days=self.train_period_days)

        while current_train_end < end_ts:
            train_start = start_ts
            train_end = current_train_end

            test_start = train_end + pd.Timedelta(days=1)
            test_end = test_start + pd.Timedelta(days=self.test_period_days)

            # Don't go past data end
            if test_end > end_ts:
                test_end = end_ts

            windows.append(WalkForwardWindow(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            ))

            window_id += 1
            current_train_end += pd.Timedelta(days=self.step_days)

        return windows

    def _train_on_window(
        self,
        train_data: pd.DataFrame,
        strategy_name: str,
        initial_capital: float,
        transaction_fee: float,
        slippage: float,
        parameters: dict | None,
    ) -> dict | None:
        """Run backtest on training data to evaluate strategy."""
        try:
            strategy = STRATEGY_MAP[strategy_name](parameters=parameters or {})
            signals = strategy.generate_signals(train_data)
            execution_signal = signals.shift(1).fillna(0)

            portfolio = Portfolio(initial_capital=initial_capital, cash=initial_capital)

            for index in range(len(train_data)):
                row = train_data.iloc[index]
                exec_price = float(row["open"]) if pd.notna(row["open"]) else float(row["close"])
                signal = float(execution_signal.iloc[index])

                if signal > 0:
                    portfolio.buy(row["timestamp"], exec_price, transaction_fee, slippage)
                elif signal < 0:
                    portfolio.sell(row["timestamp"], exec_price, transaction_fee, slippage)

                portfolio.mark_to_market(row["timestamp"], float(row["close"]))

            if portfolio.is_long:
                last = train_data.iloc[-1]
                portfolio.sell(last["timestamp"], float(last["close"]), transaction_fee, slippage)

            metrics = compute_metrics(portfolio.equity_curve, portfolio.realized_trades, initial_capital)
            return metrics

        except Exception as exc:
            logger.warning(f"Train phase failed: {exc}")
            return None

    def _test_on_window(
        self,
        test_data: pd.DataFrame,
        strategy_name: str,
        initial_capital: float,
        transaction_fee: float,
        slippage: float,
        parameters: dict | None,
    ) -> dict | None:
        """Run backtest on test data (unseen during training)."""
        try:
            strategy = STRATEGY_MAP[strategy_name](parameters=parameters or {})
            signals = strategy.generate_signals(test_data)
            execution_signal = signals.shift(1).fillna(0)

            portfolio = Portfolio(initial_capital=initial_capital, cash=initial_capital)

            for index in range(len(test_data)):
                row = test_data.iloc[index]
                exec_price = float(row["open"]) if pd.notna(row["open"]) else float(row["close"])
                signal = float(execution_signal.iloc[index])

                if signal > 0:
                    portfolio.buy(row["timestamp"], exec_price, transaction_fee, slippage)
                elif signal < 0:
                    portfolio.sell(row["timestamp"], exec_price, transaction_fee, slippage)

                portfolio.mark_to_market(row["timestamp"], float(row["close"]))

            if portfolio.is_long:
                last = test_data.iloc[-1]
                portfolio.sell(last["timestamp"], float(last["close"]), transaction_fee, slippage)

            metrics = compute_metrics(portfolio.equity_curve, portfolio.realized_trades, initial_capital)
            return metrics

        except Exception as exc:
            logger.warning(f"Test phase failed: {exc}")
            return None

    def _load_data(
        self,
        symbol: str,
        start_date: datetime | str | None,
        end_date: datetime | str | None,
    ) -> pd.DataFrame:
        """Load market data from Gold or Silver layer."""
        frame = GoldLayer().read(symbol)
        if frame.empty:
            frame = SilverLayer().read(symbol)

        if start_date:
            frame = frame[frame["timestamp"] >= pd.Timestamp(start_date, tz="UTC")]
        if end_date:
            frame = frame[frame["timestamp"] <= pd.Timestamp(end_date, tz="UTC")]

        frame = frame.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
        return frame

    @staticmethod
    def _build_disclaimer() -> str:
        return """
⚠️ WALK-FORWARD VALIDATION - PHƯƠNG PHÁP ĐÁNH GIÁ

Walk-forward validation là phương pháp nghiêm ngặt hơn simple backtest:
- Mỗi window có train period (huấn luyện) và test period (kiểm tra)
- Tham số chiến lược được "huấn luyện" trên train, đánh giá trên test
- Kết quả là TRUNG BÌNH của nhiều windows, giảm overfitting

Tuy nhiên:
- Kết quả vẫn dựa trên DỮ LIỆU LỊCH SỬ, không đảm bảo tương lai
- "Overfitting ratio" gần 1.0 là tốt; >2.0 có thể là overfitting
- "Return consistency" cao (>70%) cho thấy chiến lược ổn định hơn

ĐÂY CHỈ LÀ NGHIÊN CỨU HỌC THUẬT, KHÔNG PHẢI KHUYẾN NGHỊ ĐẦU TƯ.
"""

    @staticmethod
    def _to_dict(result: WalkForwardResult) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "symbol": result.symbol,
            "strategy": result.strategy_name,
            "summary": {
                "total_windows": result.total_windows,
                "passed_windows": result.passed_windows,
                "failed_windows": result.failed_windows,
                "avg_train_return": result.avg_train_return,
                "train_return_std": result.train_return_std,
                "avg_test_return": result.avg_test_return,
                "test_return_std": result.test_return_std,
                "avg_test_sharpe": result.avg_test_sharpe,
                "avg_test_max_drawdown": result.avg_test_max_drawdown,
                "avg_test_win_rate": result.avg_test_win_rate,
                "total_test_trades": result.total_test_trades,
                "return_consistency": result.return_consistency,
                "overfitting_ratio": result.overfitting_ratio,
            },
            "windows": [
                {
                    "window_id": w.window_id,
                    "train_period": f"{w.train_start.date()} to {w.train_end.date()}",
                    "test_period": f"{w.test_start.date()} to {w.test_end.date()}",
                    "train_return": w.train_return,
                    "test_return": w.test_return,
                    "test_sharpe": w.test_sharpe,
                    "test_max_drawdown": w.test_max_drawdown,
                    "test_win_rate": w.test_win_rate,
                    "test_num_trades": w.test_num_trades,
                    "passed": w.passed,
                    "error": w.error,
                }
                for w in result.windows
            ],
            "disclaimer": result.disclaimer,
        }


def _std(values: list[float]) -> float:
    """Calculate standard deviation."""
    if not values:
        return 0.0
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    return variance ** 0.5
