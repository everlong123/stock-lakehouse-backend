"""Simple long-only portfolio used by the research backtester."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.backtesting.trade import Trade


@dataclass
class Portfolio:
    """Cash plus a single-asset position."""

    initial_capital: float
    cash: float
    position_qty: float = 0.0
    avg_entry_price: float = 0.0
    entry_time: object | None = None
    realized_trades: list[Trade] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)

    @property
    def is_long(self) -> bool:
        return self.position_qty > 0

    def mark_to_market(self, timestamp, price: float) -> float:
        equity = self.cash + self.position_qty * price
        self.equity_curve.append({"timestamp": timestamp, "equity": equity, "price": price})
        return equity

    def buy(self, timestamp, price: float, fee_pct: float, slippage_pct: float) -> None:
        if self.is_long or self.cash <= 0:
            return
        fill_price = price * (1 + slippage_pct)
        spendable = self.cash / (1 + fee_pct)
        quantity = spendable / fill_price
        fees = quantity * fill_price * fee_pct
        cost = quantity * fill_price + fees
        if cost > self.cash or quantity <= 0:
            return
        self.cash -= cost
        self.position_qty = quantity
        self.avg_entry_price = fill_price
        self.entry_time = timestamp

    def sell(self, timestamp, price: float, fee_pct: float, slippage_pct: float) -> Trade | None:
        if not self.is_long:
            return None
        fill_price = price * (1 - slippage_pct)
        proceeds = self.position_qty * fill_price
        fees = proceeds * fee_pct
        net = proceeds - fees
        pnl = net - (self.position_qty * self.avg_entry_price)
        return_pct = pnl / (self.position_qty * self.avg_entry_price) if self.avg_entry_price else 0.0
        trade = Trade(
            entry_time=self.entry_time,
            exit_time=timestamp,
            entry_price=float(self.avg_entry_price),
            exit_price=float(fill_price),
            quantity=float(self.position_qty),
            side="long",
            pnl=float(pnl),
            return_pct=float(return_pct),
            fees=float(fees),
        )
        self.cash += net
        self.position_qty = 0.0
        self.avg_entry_price = 0.0
        self.entry_time = None
        self.realized_trades.append(trade)
        return trade
