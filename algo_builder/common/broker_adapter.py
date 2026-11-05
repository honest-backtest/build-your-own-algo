"""
Book 2, Chapter 2 — The seven parts of a trading system ("one engine, three modes").
Used by every live and paper chapter. All code talks to a broker ONLY through this interface.

A small, generic broker interface. Strategy and execution code never import a
broker SDK directly; backtest, paper and live differ only in which adapter is
plugged in. Write one concrete adapter per broker in your own code (not shown).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd


@dataclass
class OrderRequest:
    symbol: str                 # exchange trading symbol, e.g. an index future or option
    exchange: str               # e.g. "NSE", "NFO", "BSE", "BFO", "MCX"
    side: str                   # "BUY" | "SELL"
    qty: int
    order_type: str = "MARKET"  # "MARKET" | "LIMIT"
    price: float = 0.0          # used for LIMIT
    product: str = "INTRADAY"   # "INTRADAY" | "DELIVERY" | "CARRYFORWARD"
    tag: str = ""               # free-text tag for your own reconciliation


@dataclass
class OrderStatus:
    order_id: str
    status: str                 # normalised: "OPEN" | "COMPLETE" | "REJECTED" | "CANCELLED" | "UNKNOWN"
    filled_qty: int = 0
    avg_price: float = 0.0
    message: str = ""


@dataclass
class Position:
    symbol: str
    exchange: str
    net_qty: int                # signed: +long, -short
    avg_price: float = 0.0
    product: str = ""


class BrokerAdapter(ABC):
    """The five calls every excerpt is allowed to make."""

    @abstractmethod
    def get_candles(self, symbol: str, exchange: str, interval: str,
                    start: datetime, end: datetime) -> pd.DataFrame:
        """OHLCV with a tz-naive DatetimeIndex and columns open/high/low/close/volume."""

    @abstractmethod
    def get_quote(self, symbol: str, exchange: str) -> Optional[dict]:
        """{'ltp', 'bid', 'ask', 'bid_qty', 'ask_qty', 'lower_circuit', 'upper_circuit'}
        or None if the quote could not be read (callers must treat None as UNKNOWN)."""

    @abstractmethod
    def place_order(self, req: OrderRequest) -> Optional[str]:
        """Returns the broker order id, or None if the order was not accepted."""

    @abstractmethod
    def order_status(self, order_id: str) -> Optional[OrderStatus]:
        """None means 'could not read the order book' — NOT 'order gone'."""

    @abstractmethod
    def positions(self) -> Optional[list[Position]]:
        """None means 'could not read positions' — NOT 'flat'."""

    # Optional extension (only infra/execution.py uses it). Not one of the
    # five core calls; adapters that cannot cancel simply leave it unimplemented.
    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError


class PaperBroker(BrokerAdapter):
    """Minimal in-memory adapter for paper trading and tests.
    Market orders fill at the last quote you feed it via set_quote()."""

    def __init__(self):
        self._quotes: dict[tuple[str, str], dict] = {}
        self._orders: dict[str, OrderStatus] = {}
        self._pos: dict[tuple[str, str], Position] = {}
        self._seq = 0
        self._tags: dict[str, str] = {}

    def set_quote(self, symbol, exchange, ltp, bid=None, ask=None, **extra):
        self._quotes[(symbol, exchange)] = {"ltp": ltp, "bid": bid or ltp,
                                            "ask": ask or ltp, **extra}

    def get_candles(self, symbol, exchange, interval, start, end):
        raise NotImplementedError("feed historical data from your cache instead")

    def get_quote(self, symbol, exchange):
        return self._quotes.get((symbol, exchange))

    def place_order(self, req: OrderRequest):
        q = self.get_quote(req.symbol, req.exchange)
        if q is None:
            return None
        self._seq += 1
        oid = f"PAPER-{self._seq:06d}"
        self._tags[oid] = req.tag
        px = q["ask"] if req.side == "BUY" else q["bid"]
        if req.order_type == "LIMIT":
            marketable = px <= req.price if req.side == "BUY" else px >= req.price
            if not marketable:
                self._orders[oid] = OrderStatus(oid, "OPEN")
                return oid
        self._orders[oid] = OrderStatus(oid, "COMPLETE", req.qty, px)
        key = (req.symbol, req.exchange)
        p = self._pos.setdefault(key, Position(req.symbol, req.exchange, 0))
        p.net_qty += req.qty if req.side == "BUY" else -req.qty
        p.avg_price = px
        return oid

    def order_status(self, order_id):
        return self._orders.get(order_id)

    def orders_with_tag(self, tag):
        return [self._orders[o] for o, t in self._tags.items() if t == tag and o in self._orders]

    def cancel_order(self, order_id):
        o = self._orders.get(order_id)
        if o and o.status == "OPEN":
            o.status = "CANCELLED"
        return o is not None

    def positions(self):
        return [p for p in self._pos.values() if p.net_qty != 0]
