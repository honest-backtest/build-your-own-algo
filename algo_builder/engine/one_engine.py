"""
Book 2, Chapter 2 — The seven parts of a trading system ("one engine, three modes").
Illustrative. A deliberately simple signal, wired so that the same decision code
runs in a backtest, on paper and live. Only the data source and the order
destination change between modes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import pandas as pd

from algo_builder.common.bars import completed_bars
from algo_builder.common.broker_adapter import BrokerAdapter, OrderRequest, PaperBroker
from algo_builder.infra.candle_cache import get_candles as cached_candles

log = logging.getLogger("engine")


@dataclass(frozen=True)
class Intent:
    action: str               # "BUY" or "EXIT" (this example trades long only)
    symbol: str
    exchange: str
    lots: int
    reason: str
    ts: pd.Timestamp          # the completed bar the decision was made on
    ref_price: float          # the price the signal saw, logged beside the fill


Signal = Callable[[pd.DataFrame, int], list[Intent]]


def crossover_signal(bars: pd.DataFrame, position_lots: int, symbol: str = "IDX-FUT",
                     exchange: str = "NFO", fast: int = 5, slow: int = 20) -> list[Intent]:
    """Pure: completed bars and the current position in, intents out. No orders, no
    files, no clock. Windows are ILLUSTRATIVE; the point is the shape, not the edge."""
    if len(bars) < slow + 1:
        return []
    c = bars.close
    f, s = c.rolling(fast).mean(), c.rolling(slow).mean()
    above, was_above = f.iloc[-1] > s.iloc[-1], f.iloc[-2] > s.iloc[-2]
    ts, px = bars.index[-1], float(c.iloc[-1])
    if above and not was_above and position_lots == 0:
        return [Intent("BUY", symbol, exchange, 1, "fast crossed above slow", ts, px)]
    if was_above and not above and position_lots > 0:
        return [Intent("EXIT", symbol, exchange, position_lots, "fast crossed below slow", ts, px)]
    return []


def to_order(intent: Intent, lot_size: int, product: str = "INTRADAY") -> OrderRequest:
    """The execution adapter's first job: turn an intent into a broker order."""
    side = "BUY" if intent.action == "BUY" else "SELL"
    return OrderRequest(intent.symbol, intent.exchange, side, intent.lots * lot_size,
                        product=product, tag=f"{intent.reason} @ {intent.ts:%Y-%m-%d %H:%M}")


def step(bars: pd.DataFrame, state: dict, signal: Signal, broker: BrokerAdapter,
         lot_size: int) -> list[dict]:
    """One decision cycle, identical in every mode: completed bars in, orders out."""
    fills = []
    for it in signal(bars, state.get("lots", 0)):
        oid = broker.place_order(to_order(it, lot_size))
        st = broker.order_status(oid) if oid else None
        if st is None or st.status != "COMPLETE":
            log.warning("%s %s not confirmed (%s); state unchanged",
                        it.action, it.symbol, st.status if st else "no order id")
            continue
        state["lots"] = state.get("lots", 0) + (it.lots if it.action == "BUY" else -it.lots)
        fills.append(dict(ts=it.ts, action=it.action, signal_px=it.ref_price,
                          fill_px=st.avg_price, qty=st.filled_qty))
    return fills


def backtest(bars: pd.DataFrame, signal: Signal, symbol: str, exchange: str,
             lot_size: int) -> list[dict]:
    """Replay history one bar at a time. Each decision, taken on a completed bar,
    is filled at the NEXT bar's open: the earliest price you could really get."""
    broker, state, fills = PaperBroker(), {}, []
    for i in range(1, len(bars)):
        broker.set_quote(symbol, exchange, float(bars.open.iloc[i]))
        fills += step(bars.iloc[:i], state, signal, broker, lot_size)
    return fills


def cycle(data: BrokerAdapter, orders: BrokerAdapter, state: dict, signal: Signal,
          symbol: str, exchange: str, interval: str, lot_size: int, now: datetime,
          lookback: pd.Timedelta = pd.Timedelta(days=5)) -> list[dict]:
    """One scheduled run in paper or live mode. `data` supplies candles and quotes;
    `orders` receives the orders: a PaperBroker on paper, your real adapter live."""
    raw = cached_candles(data, symbol, exchange, interval, now - lookback, now, now=now)
    bars = completed_bars(raw, now, interval)
    if isinstance(orders, PaperBroker):
        q = data.get_quote(symbol, exchange)
        if q is None:
            return []                               # price unknown: act on nothing
        orders.set_quote(symbol, exchange, q["ltp"], q.get("bid"), q.get("ask"))
    return step(bars, state, signal, orders, lot_size)
