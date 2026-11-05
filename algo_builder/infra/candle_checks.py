"""
Book 2, Chapter 3 — The data layer: checking what came back before trusting it.
Illustrative.

Two checks every new data source should pass before any strategy reads it:
  * which moment its timestamps mean (bar START or bar END);
  * whether the payload is really candles at all, and sane ones.
"""
from __future__ import annotations

from datetime import time as dtime

import pandas as pd


def infer_bar_label(bars: pd.DataFrame, interval: str,
                    session_open: dtime = dtime(9, 15)) -> str:
    """'start' if each day's first bar is stamped at the session open, 'end' if it
    is stamped one interval later, 'unknown' otherwise. Check before you trust."""
    firsts = pd.Series(bars.index).groupby(bars.index.normalize()).min()
    offset = (firsts - firsts.dt.normalize()).mode().iloc[0]
    open_off = pd.Timedelta(hours=session_open.hour, minutes=session_open.minute)
    if offset == open_off:
        return "start"
    if offset == open_off + pd.Timedelta(interval):
        return "end"
    return "unknown"


def validate_candles(df: object, interval: str) -> list[str]:
    """Problems with a candle payload; an empty list means it passed. A response can
    succeed at the HTTP level and still be the wrong thing entirely."""
    if not isinstance(df, pd.DataFrame):
        return [f"not a table: got {type(df).__name__}"]
    need = {"open", "high", "low", "close", "volume"}
    if missing := need - set(df.columns):
        return [f"missing columns: {sorted(missing)}"]
    if df.empty:
        return ["empty"]
    problems = []
    if not isinstance(df.index, pd.DatetimeIndex):
        problems.append("index is not timestamps")
    elif not df.index.is_monotonic_increasing or df.index.has_duplicates:
        problems.append("timestamps out of order or duplicated")
    if (df.high < df[["open", "close"]].max(axis=1)).any() or \
            (df.low > df[["open", "close"]].min(axis=1)).any():
        problems.append("high/low inconsistent with open/close")
    if (df[["open", "high", "low", "close"]] <= 0).any().any() or (df.volume < 0).any():
        problems.append("non-positive prices or negative volume")
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
        step = pd.Series(df.index).diff().dropna()
        same_day = step[step < pd.Timedelta(hours=12)]
        if (same_day % pd.Timedelta(interval) != pd.Timedelta(0)).any():
            problems.append(f"bar spacing is not a multiple of {interval}")
    return problems
