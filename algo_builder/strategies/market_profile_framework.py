"""
Book 2, Chapter 12 — Market Profile day types as a state machine.

GENERIC FRAMEWORK ONLY. This file deliberately contains no classification
filters, no width buckets, no thresholds, no setup rules and no session times.
It shows how the components fit together; you supply the rules.

Components (as described in the chapter):
  * InitialBalance    — builds the first-window range; nothing may fire before it closes.
  * extension         — which side(s) of the initial balance the session has broken.
  * RollingPercentile — turns today's width into a percentile of earlier days only.
  * value_area        — prior-session value area high/low and point of control.
  * DayTypeMachine    — a STATE MACHINE updated bar by bar (the label can change).
  * on_bar            — the loop body; the trade log stores the day type AS KNOWN AT ENTRY.
  * trail_from_completed — trailing only behind higher-timeframe bars that have closed.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from typing import Callable, Optional

import numpy as np
import pandas as pd

from algo_builder.common.bars import completed_bars


@dataclass
class InitialBalance:
    window_end: dtime                       # supplied by the caller
    high: float = -np.inf
    low: float = np.inf
    closed: bool = False

    def update(self, ts: pd.Timestamp, bar) -> None:
        if self.closed:
            return
        if ts.time() >= self.window_end:
            self.closed = True              # from here on, setups may be evaluated
            return
        self.high, self.low = max(self.high, bar.high), min(self.low, bar.low)

    @property
    def width(self) -> float:
        return self.high - self.low if self.closed and np.isfinite(self.high) else np.nan


def extension(ib: InitialBalance, session_high: float, session_low: float) -> str:
    """Range extension so far: 'none', 'up', 'down' or 'both'. Facts, not rules."""
    up, down = session_high > ib.high, session_low < ib.low
    return "both" if up and down else "up" if up else "down" if down else "none"


class RollingPercentile:
    """Percentile of today's value among the last `n` earlier observations.
    Returns None until enough history exists (caller decides the fallback)."""

    def __init__(self, n: int, min_history: int):
        self.hist: deque[float] = deque(maxlen=n)
        self.min_history = min_history

    def rank(self, x: float) -> Optional[float]:
        if len(self.hist) < self.min_history:
            return None
        return float((np.asarray(self.hist) < x).mean())

    def push(self, x: float) -> None:
        self.hist.append(x)

    def rank_then_push(self, x: float) -> Optional[float]:
        """Rank today against earlier days only, then add today for tomorrow."""
        r = self.rank(x)
        self.push(x)
        return r


def value_area(prices: np.ndarray, volumes: np.ndarray, tick: float,
               share: float) -> tuple[float, float, float]:
    """(VAL, VAH, POC) from a volume-at-price histogram. `share` is the value-area
    fraction (textbook convention; pass your own)."""
    bins = np.round(prices / tick) * tick
    vap = pd.Series(volumes).groupby(bins).sum().sort_index()
    poc_i = int(np.argmax(vap.values))
    lo = hi = poc_i
    total, acc = vap.sum(), vap.iloc[poc_i]
    while acc < share * total and (lo > 0 or hi < len(vap) - 1):
        up = vap.iloc[hi + 1] if hi < len(vap) - 1 else -1
        dn = vap.iloc[lo - 1] if lo > 0 else -1
        if up >= dn:
            hi += 1; acc += up
        else:
            lo -= 1; acc += dn
    return float(vap.index[lo]), float(vap.index[hi]), float(vap.index[poc_i])


@dataclass
class DayTypeMachine:
    """Holds the current day-type label and re-evaluates it on every bar.
    `rules` maps a label to a predicate(context) -> bool; they are YOUR rules,
    evaluated in priority order. None are provided here. `allowed`, if given, lists
    the labels each label may move to; a matching rule outside that list is ignored."""
    rules: list[tuple[str, Callable[[dict], bool]]]
    allowed: Optional[dict[str, set[str]]] = None
    label: str = "UNDETERMINED"
    history: list[tuple[pd.Timestamp, str]] = field(default_factory=list)

    def update(self, ts: pd.Timestamp, ctx: dict) -> str:
        for name, pred in self.rules:
            if not pred(ctx):
                continue
            if name == self.label:
                break                                   # the current label still holds
            if self.allowed is not None and name not in self.allowed.get(self.label, set()):
                continue                                # a transition you have ruled out
            self.history.append((ts, name))
            self.label = name
            break
        return self.label


def on_bar(ts, bar, ib: InitialBalance, machine: DayTypeMachine,
           setups: dict[str, Callable[[dict], Optional[dict]]], ctx: dict,
           trade_log: list, taken: set[str]) -> None:
    """Skeleton loop body: update range, update state, ask the handler for the
    CURRENT day type whether it has a setup. One handler per tradable day type;
    `taken` holds the sides already traded today (one long and one short at most)."""
    ib.update(ts, bar)
    ctx["session_high"] = max(ctx.get("session_high", -np.inf), bar.high)
    ctx["session_low"] = min(ctx.get("session_low", np.inf), bar.low)
    if not ib.closed:
        return                                              # no signal during formation
    ctx.update(ts=ts, bar=bar, ib_high=ib.high, ib_low=ib.low, ib_width=ib.width,
               extension=extension(ib, ctx["session_high"], ctx["session_low"]))
    day_type = machine.update(ts, ctx)
    handler = setups.get(day_type)
    sig = handler(ctx) if handler else None                 # errors are NOT caught here
    if sig and sig["side"] not in taken:
        sig["day_type_at_entry"] = day_type                 # no look-ahead in attribution
        taken.add(sig["side"])
        trade_log.append(sig)


def trail_from_completed(direction: int, sl: float, entry: float, best_move: float,
                         breakeven_after: float, bars_1m: pd.DataFrame, now: datetime,
                         rule: str, buffer: float) -> float:
    """New stop for a trend trade: unchanged until the trade has been `breakeven_after`
    (points) in profit, then at least breakeven, then behind the extreme of the last
    COMPLETED `rule` bar. `bars_1m` is today's completed one-minute bars, from the
    open. The stop only ever moves in the trade's favour."""
    if best_move < breakeven_after:
        return sl
    higher = (bars_1m.resample(rule, origin="start", label="left", closed="left")   # aligned to the session open
              .agg({"high": "max", "low": "min"}).dropna())
    done = completed_bars(higher, now, rule)
    level = entry
    if len(done):
        last = done.iloc[-1]
        level = (last.low - buffer) if direction > 0 else (last.high + buffer)
        level = max(level, entry) if direction > 0 else min(level, entry)
    return max(sl, level) if direction > 0 else min(sl, level)
