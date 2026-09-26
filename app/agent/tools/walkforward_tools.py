"""Walk-forward backtest tool for AI agent."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.services.backtest_service import WalkForwardService


class WalkForwardInput(BaseModel):
    symbol: str = Field(description="Ticker symbol, e.g. AAPL, VCB")
    strategy: str = Field(default="ma_crossover", description="Strategy name: ma_crossover, rsi_strategy")
    train_period_days: int = Field(default=180, ge=30, le=365, description="Training period in days")
    test_period_days: int = Field(default=30, ge=5, le=90, description="Test period in days")
    initial_capital: float = Field(default=10000, gt=0, description="Initial capital")


def run_walk_forward(
    symbol: str,
    strategy: str = "ma_crossover",
    train_period_days: int = 180,
    test_period_days: int = 30,
    initial_capital: float = 10000,
) -> dict:
    """
    Chạy walk-forward validation - phương pháp backtest nghiêm ngặt hơn.

    Walk-forward validation:
    - Chia dữ liệu thành nhiều windows, mỗi window có train và test period
    - Mô phỏng quá trình deploy thực tế: train -> test -> train -> test
    - Kết quả là trung bình của nhiều windows, giảm overfitting

    Args:
        symbol: Mã chứng khoán
        strategy: Chiến lược giao dịch
        train_period_days: Số ngày train (mặc định 180 ngày = 6 tháng)
        test_period_days: Số ngày test (mặc định 30 ngày = 1 tháng)
        initial_capital: Vốn ban đầu
    """
    service = WalkForwardService()

    result = service.run({
        "symbol": symbol.upper(),
        "strategy": strategy,
        "train_period_days": train_period_days,
        "test_period_days": test_period_days,
        "initial_capital": initial_capital,
    })

    return result
