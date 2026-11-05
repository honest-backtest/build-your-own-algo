"""
Book 2, Chapter 12 — Market Profile day types (the MCX coda).

Shows: (1) session-aware first-hour windows from a calendar you supply, and
       (2) the rolling-buffer VWAP-touch check that replaced "did THIS bar touch VWAP".

Fast trends often tag VWAP once and run; checking only the current bar never
enters them. Keep a short buffer of recent bar highs/lows and ask whether ANY of
them touched VWAP. No settings are supplied here.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, time as dtime


@dataclass(frozen=True)
class Session:
    name: str
    start: dtime
    ib_minutes: int


def session_for(ts: datetime, calendar: list[Session]) -> Session | None:
    """Pick the session containing ts. Commodity evening hours shift with US daylight
    saving — build `calendar` per date from the exchange's published timings."""
    for s in sorted(calendar, key=lambda s: s.start, reverse=True):
        if ts.time() >= s.start:
            return s
    return None


class SessionVWAP:
    """Volume-weighted average price for ONE session: make a new one, or call
    reset(), at each session's start (MCX has two)."""

    def __init__(self):
        self.pv = self.v = 0.0

    def reset(self) -> None:
        self.pv = self.v = 0.0

    def update(self, typical_price: float, volume: float) -> float:
        self.pv += typical_price * volume
        self.v += volume
        return self.pv / self.v if self.v else typical_price


class VWAPTouchBuffer:
    """The last `lookback` bars' high, low and VWAP: you choose the lookback."""

    def __init__(self, lookback: int):
        self.bars: deque[tuple[float, float, float]] = deque(maxlen=lookback)

    def push(self, high: float, low: float, vwap: float) -> None:
        self.bars.append((high, low, vwap))

    def touched_recently(self) -> bool:
        return any(lo <= vw <= hi for hi, lo, vw in self.bars)


def trend_pullback_long(close: float, vwap: float, buf: VWAPTouchBuffer,
                        one_sided_up: bool) -> bool:
    """Up-trend day + a recent VWAP touch + close back above VWAP."""
    return one_sided_up and buf.touched_recently() and close > vwap
