"""
Book 2, Chapter 5 — A backtester that doesn't lie.
Also used in Chapters 13 (opening range) and 25 (the NIFTY-50 stock ORB post-mortem).

A minimal event-driven, bar-by-bar engine for one intraday position at a time.
The instructive parts are the ORDER OF CHECKS inside a bar and the bookkeeping:

  * Range-formation guard: no entry is allowed until the range window has CLOSED.
  * The fill clock: a decision taken on a completed bar is filled at the open of a
    LATER bar (`fill_delay_bars`, default 1), with adverse slippage. Zero reproduces
    the common lie of filling at the decision bar's own close.
  * Stops are pierced intra-bar and fill at the stop, or at the open if the bar
    GAPPED through it, with adverse slippage. Targets fill at the target price.
  * Within one bar, check the STOP FIRST: with only OHLC you cannot know which was
    hit first, so assume the worse one.
  * Keep the ENTRY-TIME stop immutable (initial_sl). R is measured against it.
  * Every trade is booked net of the cost model you pass in; nothing is left open.

Parameters are ILLUSTRATIVE; signal logic is injected by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import time as dtime
from typing import Callable, Optional

import pandas as pd

from algo_builder.infra.cost_model import EQ_INTRADAY, CostRates, round_trip_cost


@dataclass
class Position:
    direction: int            # +1 long / -1 short
    entry: float              # the signal's reference price until filled, then the fill
    qty: int
    initial_sl: float         # immutable: used for R
    sl: float                 # may trail, only in the trade's favour
    target: float
    entry_ts: pd.Timestamp


@dataclass
class Trade:
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    direction: int
    entry: float
    exit: float
    qty: int
    initial_sl: float
    reason: str
    costs: CostRates = EQ_INTRADAY
    pnl_net: float = field(init=False)
    r: float = field(init=False)

    def __post_init__(self):
        gross = self.direction * (self.exit - self.entry) * self.qty
        buy, sell = (self.entry, self.exit) if self.direction > 0 else (self.exit, self.entry)
        self.pnl_net = gross - round_trip_cost(buy * self.qty, sell * self.qty, self.costs)
        risk = abs(self.entry - self.initial_sl) * self.qty
        self.r = self.pnl_net / risk if risk > 0 else 0.0


def _fill_entry(plan: Position, price: float, ts: pd.Timestamp, slip: float) -> Optional[Position]:
    """Fill a planned entry at `price` plus adverse slippage. If the fill is already
    at or beyond the stop or the target, the trade the signal planned no longer exists."""
    d = plan.direction
    fill = price * (1 + d * slip)
    if d * (fill - plan.sl) <= 0 or d * (plan.target - fill) <= 0:
        return None
    return replace(plan, entry=fill, entry_ts=ts)


def run_day(bars: pd.DataFrame,
            signal: Callable[[pd.DataFrame, pd.Timestamp], Optional[Position]],
            range_end: dtime, square_off: dtime, slip_pct: float = 0.0005,
            fill_delay_bars: int = 1, costs: CostRates = EQ_INTRADAY, max_trades: int = 1,
            trail: Optional[Callable[[Position, object], None]] = None) -> list[Trade]:
    """bars: one session of OHLC with a DatetimeIndex. `signal(history, ts)` sees bars
    up to and including the completed bar `ts` and may return a planned Position.
    `trail(pos, bar)`, if given, may move pos.sl in the trade's favour after each bar."""
    trades: list[Trade] = []
    pos: Optional[Position] = None
    plan: Optional[Position] = None
    wait = 0
    rows = list(bars.itertuples())
    for i, b in enumerate(rows):
        ts = b.Index
        if plan is not None:                                  # the fill clock
            wait -= 1
            if wait == 0:
                pos, plan = _fill_entry(plan, b.open, ts, slip_pct), None
        if pos is not None and pos.entry_ts <= ts:
            d = pos.direction
            stop_hit = (b.low <= pos.sl) if d > 0 else (b.high >= pos.sl)
            tgt_hit = (b.high >= pos.target) if d > 0 else (b.low <= pos.target)
            last_bar = ts.time() >= square_off or i == len(rows) - 1
            exit_px, reason = None, ""
            if stop_hit and not (pos.entry_ts == ts and fill_delay_bars == 0):   # worse case first
                trigger = min(b.open, pos.sl) if d > 0 else max(b.open, pos.sl)   # a gap fills at the open
                exit_px = trigger * (1 - d * slip_pct)
                reason = "SL" if pos.sl == pos.initial_sl else "TRAIL"
            elif tgt_hit and not (pos.entry_ts == ts and fill_delay_bars == 0):
                exit_px, reason = pos.target, "TARGET"
            elif last_bar:
                exit_px, reason = b.close * (1 - d * slip_pct), "EOD"
            if exit_px is not None:
                trades.append(Trade(pos.entry_ts, ts, d, pos.entry, exit_px, pos.qty,
                                    pos.initial_sl, reason, costs))
                pos = None
            elif trail is not None:
                trail(pos, b)
        if (pos is None and plan is None and len(trades) < max_trades
                and range_end <= ts.time() < square_off):      # range window already closed
            new = signal(bars.iloc[:i + 1], ts)
            if new is not None and fill_delay_bars == 0:       # the lie: fill at this close
                pos = _fill_entry(new, b.close, ts, slip_pct)
            elif new is not None:
                plan, wait = new, fill_delay_bars
    return trades
