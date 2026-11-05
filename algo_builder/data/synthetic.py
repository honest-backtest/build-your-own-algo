"""
Synthetic market data so every example runs without a broker or paid data.

None of this resembles any real market closely enough to test a strategy on.
It exists to exercise code paths: sessions, gaps, trends, quiet days, holidays.
"""
from __future__ import annotations

from datetime import date, time as dtime

import numpy as np
import pandas as pd

SESSION_OPEN, SESSION_CLOSE = dtime(9, 15), dtime(15, 30)


def trading_days(start: str, end: str, holidays: frozenset[date] = frozenset()) -> pd.DatetimeIndex:
    days = pd.bdate_range(start, end)
    return pd.DatetimeIndex([d for d in days if d.date() not in holidays])


def minute_bars(days: pd.DatetimeIndex, start_price: float = 25_000.0,
                daily_vol: float = 0.01, gap_vol: float = 0.004, seed: int = 0) -> pd.DataFrame:
    """One-minute OHLCV for each day, labelled by bar START, 09:15 to 15:29."""
    rng = np.random.default_rng(seed)
    minutes = pd.timedelta_range("9h15min", "15h29min", freq="1min")
    per_min = daily_vol / np.sqrt(len(minutes))
    frames, last = [], start_price
    for d in days:
        open_px = last * np.exp(rng.normal(0, gap_vol))              # overnight gap
        path = open_px * np.exp(np.cumsum(rng.normal(0, per_min, len(minutes))))
        opens = np.concatenate([[open_px], path[:-1]])
        wiggle = np.abs(rng.normal(0, per_min / 2, len(minutes))) * path
        frames.append(pd.DataFrame({
            "open": opens, "close": path,
            "high": np.maximum(opens, path) + wiggle,
            "low": np.minimum(opens, path) - wiggle,
            "volume": rng.integers(1_000, 5_000, len(minutes)),
        }, index=d.normalize() + minutes))
        last = path[-1]
    return pd.concat(frames)


def resample(bars: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Start-labelled resample within each session (no bar spans two days)."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = [g.resample(rule, label="left", closed="left", origin="start").agg(agg)
           for _, g in bars.groupby(bars.index.normalize())]
    return pd.concat(out).dropna()


def daily_panel(symbols: list[str], days: pd.DatetimeIndex, drift: float = 0.0004,
                vol: float = 0.018, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Daily open/high/low/close/volume panels (date x symbol) for a stock universe."""
    rng = np.random.default_rng(seed)
    n, k = len(days), len(symbols)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, (n, k)), axis=0))
    prev = np.vstack([close[:1], close[:-1]])
    open_ = prev * np.exp(rng.normal(0, vol / 3, (n, k)))
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, (n, k))))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, (n, k))))
    vol_ = rng.integers(100_000, 1_000_000, (n, k))
    mk = lambda a: pd.DataFrame(a, index=days, columns=symbols)
    return dict(open=mk(open_), high=mk(hi), low=mk(lo), close=mk(close), volume=mk(vol_))
