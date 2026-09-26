"""Order Book and Trades data provider - high-frequency market microstructure data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

from app.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class OrderBookEntry:
    """Single order book level."""
    price: float
    volume: int
    orders: int  # number of orders at this level


@dataclass
class OrderBook:
    """Order book snapshot."""
    symbol: str
    timestamp: datetime
    bids: list[OrderBookEntry]  # buy orders (sorted desc by price)
    asks: list[OrderBookEntry]  # sell orders (sorted asc by price)
    best_bid: float
    best_ask: float
    spread: float
    mid_price: float


@dataclass
class Trade:
    """Single trade execution."""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    side: str  # 'buy' or 'sell'
    trade_type: str  # 'regular', 'odd_lot', 'block'


class OrderBookProvider:
    """Fetch order book and trade data from VN stock exchange."""

    source_name = "orderbook"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json, text/html",
        })

    def get_order_book(self, symbol: str, depth: int = 10) -> OrderBook | None:
        """
        Get current order book for a symbol.

        Args:
            symbol: Stock ticker symbol
            depth: Number of price levels to fetch

        Returns:
            OrderBook object or None if unavailable
        """
        symbol = symbol.upper()

        # Try Cafef order book API
        orderbook = self._fetch_cafef_orderbook(symbol, depth)

        if orderbook is None:
            # Try SSI API as fallback
            orderbook = self._fetch_ssi_orderbook(symbol, depth)

        return orderbook

    def _fetch_cafef_orderbook(self, symbol: str, depth: int) -> OrderBook | None:
        """Fetch order book from Cafef."""
        try:
            url = f"https://s.cafef.vn/Ajax/Page/OrderBook/{symbol}.json"

            response = self.session.get(url, timeout=10)

            if response.status_code != 200:
                return None

            data = response.json()

            bids = []
            asks = []

            # Parse bid levels
            for i, bid in enumerate(data.get("Buy", [])[:depth]):
                bids.append(OrderBookEntry(
                    price=float(bid.get("Price", 0)),
                    volume=int(bid.get("Volume", 0)),
                    orders=int(bid.get("Orders", 1)),
                ))

            # Parse ask levels
            for i, ask in enumerate(data.get("Sell", [])[:depth]):
                asks.append(OrderBookEntry(
                    price=float(ask.get("Price", 0)),
                    volume=int(ask.get("Volume", 0)),
                    orders=int(ask.get("Orders", 1)),
                ))

            best_bid = bids[0].price if bids else 0
            best_ask = asks[0].price if asks else 0
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2

            return OrderBook(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                bids=bids,
                asks=asks,
                best_bid=best_bid,
                best_ask=best_ask,
                spread=spread,
                mid_price=mid_price,
            )

        except Exception as exc:
            logger.debug(f"Cafef orderbook fetch failed for {symbol}: {exc}")
            return None

    def _fetch_ssi_orderbook(self, symbol: str, depth: int) -> OrderBook | None:
        """Fetch order book from SSI."""
        try:
            # SSI uses a different API format
            url = f"https://www.ssi.com.vn/ajax/getorderbook/{symbol}"

            response = self.session.get(url, timeout=10)

            if response.status_code != 200:
                return None

            data = response.json()

            bids = []
            asks = []

            for bid in data.get("bids", [])[:depth]:
                bids.append(OrderBookEntry(
                    price=float(bid.get("price", 0)),
                    volume=int(bid.get("vol", 0)),
                    orders=int(bid.get("orders", 1)),
                ))

            for ask in data.get("asks", [])[:depth]:
                asks.append(OrderBookEntry(
                    price=float(ask.get("price", 0)),
                    volume=int(ask.get("vol", 0)),
                    orders=int(ask.get("orders", 1)),
                ))

            best_bid = bids[0].price if bids else 0
            best_ask = asks[0].price if asks else 0

            return OrderBook(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                bids=bids,
                asks=asks,
                best_bid=best_bid,
                best_ask=best_ask,
                spread=best_ask - best_bid,
                mid_price=(best_bid + best_ask) / 2,
            )

        except Exception as exc:
            logger.debug(f"SSI orderbook fetch failed for {symbol}: {exc}")
            return None

    def get_recent_trades(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Get recent trades for a symbol."""
        symbol = symbol.upper()
        records = []

        try:
            # Try to get from Cafef trades
            url = f"https://s.cafef.vn/Ajax/Page/TradeHistory/{symbol}.json"

            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()

                for trade in data.get("Data", [])[:limit]:
                    records.append({
                        "symbol": symbol,
                        "timestamp": self._parse_time(trade.get("Time", "")),
                        "price": float(trade.get("Price", 0)),
                        "volume": int(trade.get("Volume", 0)),
                        "side": "buy" if trade.get("Type") == "KL" else "sell",
                        "trade_type": "regular",
                    })

        except Exception as exc:
            logger.warning(f"Trades fetch failed for {symbol}: {exc}")

        frame = pd.DataFrame(records)

        if not frame.empty:
            frame["source"] = self.source_name
            frame["ingestion_time"] = pd.Timestamp.now(tz="UTC")

        return frame

    def get_trade_summary(self, symbol: str) -> dict[str, Any]:
        """Get trade summary statistics."""
        trades = self.get_recent_trades(symbol, limit=1000)

        if trades.empty:
            return {
                "symbol": symbol,
                "total_trades": 0,
                "total_volume": 0,
                "avg_price": 0,
                "buy_volume": 0,
                "sell_volume": 0,
            }

        buy_vol = trades[trades["side"] == "buy"]["volume"].sum()
        sell_vol = trades[trades["side"] == "sell"]["volume"].sum()

        return {
            "symbol": symbol,
            "total_trades": len(trades),
            "total_volume": int(trades["volume"].sum()),
            "avg_price": float(trades["price"].mean()),
            "max_price": float(trades["price"].max()),
            "min_price": float(trades["price"].min()),
            "buy_volume": int(buy_vol),
            "sell_volume": int(sell_vol),
            "buy_ratio": float(buy_vol / (buy_vol + sell_vol)) if (buy_vol + sell_vol) > 0 else 0.5,
        }

    def orderbook_to_df(self, orderbook: OrderBook) -> pd.DataFrame:
        """Convert order book to DataFrame format."""
        if orderbook is None:
            return pd.DataFrame()

        records = []

        for bid in orderbook.bids:
            records.append({
                "symbol": orderbook.symbol,
                "timestamp": orderbook.timestamp,
                "side": "bid",
                "price": bid.price,
                "volume": bid.volume,
                "orders": bid.orders,
                "cumulative_volume": sum(b.volume for b in orderbook.bids[:orderbook.bids.index(bid) + 1]),
            })

        for ask in orderbook.asks:
            records.append({
                "symbol": orderbook.symbol,
                "timestamp": orderbook.timestamp,
                "side": "ask",
                "price": ask.price,
                "volume": ask.volume,
                "orders": ask.orders,
                "cumulative_volume": sum(a.volume for a in orderbook.asks[:orderbook.asks.index(ask) + 1]),
            })

        frame = pd.DataFrame(records)
        if not frame.empty:
            frame["source"] = self.source_name
            frame["ingestion_time"] = pd.Timestamp.now(tz="UTC")
            frame["spread"] = orderbook.spread
            frame["mid_price"] = orderbook.mid_price

        return frame

    @staticmethod
    def _parse_time(time_str: str) -> datetime:
        """Parse time string to datetime."""
        if not time_str:
            return datetime.now(timezone.utc)

        formats = [
            "%H:%M:%S",
            "%H:%M",
            "%Y-%m-%d %H:%M:%S",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(time_str.strip(), fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        return datetime.now(timezone.utc)
