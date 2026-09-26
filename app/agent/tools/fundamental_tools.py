"""Fundamental data tools for AI agent."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.data_sources.enhanced_fundamental_provider import EnhancedFundamentalProvider


class FundamentalMetricsInput(BaseModel):
    symbol: str = Field(description="Ticker symbol, e.g. VCB, AAPL")


class IncomeStatementInput(BaseModel):
    symbol: str = Field(description="Ticker symbol")
    years: int = Field(default=4, ge=1, le=10, description="Số năm dữ liệu")


class BalanceSheetInput(BaseModel):
    symbol: str = Field(description="Ticker symbol")
    years: int = Field(default=4, ge=1, le=10, description="Số năm dữ liệu")


class CashFlowInput(BaseModel):
    symbol: str = Field(description="Ticker symbol")
    years: int = Field(default=4, ge=1, le=10, description="Số năm dữ liệu")


class MarketIndexInput(BaseModel):
    symbol: str = Field(default="VNINDEX", description="Index symbol: VNINDEX, HNX, UPCOM, VN30")


class OrderBookInput(BaseModel):
    symbol: str = Field(description="Ticker symbol")
    depth: int = Field(default=10, ge=1, le=50, description="Số mức giá")


def get_fundamental_metrics(symbol: str) -> dict[str, Any]:
    """Lấy metrics tài chính tổng hợp cho một mã."""
    provider = EnhancedFundamentalProvider()
    metrics = provider.get_comprehensive_metrics(symbol.upper())
    return provider.metrics_to_dict(metrics)


def get_income_statement(symbol: str, years: int = 4) -> dict[str, Any]:
    """Lấy báo cáo thu nhập."""
    provider = EnhancedFundamentalProvider()
    df = provider.get_income_statement(symbol.upper(), years=years)

    if df.empty:
        return {"symbol": symbol.upper(), "data": [], "count": 0}

    return {
        "symbol": symbol.upper(),
        "data": df.to_dict("records"),
        "count": len(df),
    }


def get_balance_sheet(symbol: str, years: int = 4) -> dict[str, Any]:
    """Lấy bảng cân đối kế toán."""
    provider = EnhancedFundamentalProvider()
    df = provider.get_balance_sheet(symbol.upper(), years=years)

    if df.empty:
        return {"symbol": symbol.upper(), "data": [], "count": 0}

    return {
        "symbol": symbol.upper(),
        "data": df.to_dict("records"),
        "count": len(df),
    }


def get_cash_flow(symbol: str, years: int = 4) -> dict[str, Any]:
    """Lấy báo cáo lưu chuyển tiền tệ."""
    provider = EnhancedFundamentalProvider()
    df = provider.get_cash_flow(symbol.upper(), years=years)

    if df.empty:
        return {"symbol": symbol.upper(), "data": [], "count": 0}

    return {
        "symbol": symbol.upper(),
        "data": df.to_dict("records"),
        "count": len(df),
    }


def get_market_index(symbol: str = "VNINDEX") -> dict[str, Any]:
    """Lấy dữ liệu chỉ số thị trường."""
    from app.data_sources.market_index_provider import MarketIndexProvider

    provider = MarketIndexProvider()
    summary = provider.get_index_summary(symbol.upper())

    return {
        "index": symbol.upper(),
        "data": summary,
    }


def get_orderbook(symbol: str, depth: int = 10) -> dict[str, Any]:
    """Lấy sổ lệnh hiện tại."""
    from app.data_sources.orderbook_provider import OrderBookProvider

    provider = OrderBookProvider()
    orderbook = provider.get_order_book(symbol.upper(), depth=depth)

    if orderbook is None:
        return {
            "symbol": symbol.upper(),
            "error": "Order book không khả dụng cho mã này.",
        }

    return {
        "symbol": orderbook.symbol,
        "timestamp": orderbook.timestamp.isoformat(),
        "best_bid": orderbook.best_bid,
        "best_ask": orderbook.best_ask,
        "spread": orderbook.spread,
        "mid_price": orderbook.mid_price,
        "bid_levels": [
            {"price": b.price, "volume": b.volume, "orders": b.orders}
            for b in orderbook.bids
        ],
        "ask_levels": [
            {"price": a.price, "volume": a.volume, "orders": a.orders}
            for a in orderbook.asks
        ],
    }
