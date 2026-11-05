"""
Book 2, Chapter 9 — The one-minute test.
Also used in Chapter 11 (the default report) and Chapter 22 (buy-on-gap).

The official opening print is not a price you can trade at. Backtests that fill
at the stamped open can show an "edge" that disappears one minute later. The
test: re-run the identical signals entering at the open print, then +1, +5, +15
minutes, and see where the per-trade edge goes. Also decompose WHERE the money
is made: open -> clock-time move vs. clock-time -> exit move.

Inputs:
  minute_bars : dict[(symbol, date)] -> one-minute OHLC DataFrame (DatetimeIndex).
  signals     : DataFrame with  symbol, date, exit_px  (the strategy's exit price),
                and optionally exit_ts (no entry may be filled at or after it).
"""
from __future__ import annotations

from datetime import time as dtime
from typing import Callable, Optional

import numpy as np
import pandas as pd

from algo_builder.backtest.event_engine import Position, run_day


def entry_price(bars: pd.DataFrame, delay_min: int, session_open=dtime(9, 15)) -> float:
    """delay 0 = the opening print; otherwise the OPEN of the first bar that starts
    delay_min minutes or more after the session opens: the first price a late order could get."""
    if bars.empty:
        return np.nan
    if delay_min == 0:
        return float(bars.iloc[0].open)
    t0 = bars.index[0].normalize() + pd.Timedelta(hours=session_open.hour,
                                                  minutes=session_open.minute + delay_min)
    later = bars[bars.index >= t0]
    return float(later.iloc[0].open) if len(later) else np.nan


def _entries(minute_bars: dict, signals: pd.DataFrame, delays) -> np.ndarray:
    """One row per signal, one column per delay: the entry price, or NaN where no
    fill was possible (no bars, or the delayed entry would come at or after the exit)."""
    rows = []
    for s in signals.itertuples():
        bars = minute_bars.get((s.symbol, s.date))
        if bars is None:
            rows.append([np.nan] * len(delays))
            continue
        if hasattr(s, "exit_ts"):
            bars = bars[bars.index < s.exit_ts]
        rows.append([entry_price(bars, d) for d in delays])
    return np.array(rows, dtype=float).reshape(len(signals), len(delays))


def fill_delay_table(minute_bars: dict, signals: pd.DataFrame, side: int = +1,
                     delays=(0, 1, 5, 15), cost_rt: float = 0.0) -> pd.DataFrame:
    """Mean return per trade for the SAME trades entered at each delay. A trade that
    cannot be filled at every delay is dropped from all of them, so the rows compare
    like with like. Exits come from `signals` and do not move with the entry."""
    px = _entries(minute_bars, signals, delays)
    exit_px = signals.exit_px.to_numpy(float)
    ok = (px > 0).all(axis=1) & np.isfinite(exit_px)
    rets = side * (exit_px[ok, None] / px[ok] - 1) - cost_rt
    rows = []
    for j, d in enumerate(delays):
        r, n = rets[:, j], int(ok.sum())
        sd = r.std(ddof=1) if n > 1 else 0.0
        rows.append(dict(delay_min=d, n=n, mean_pct=100 * r.mean() if n else np.nan,
                         t_stat=r.mean() / (sd / np.sqrt(n)) if sd > 0 else np.nan))
    out = pd.DataFrame(rows)
    first = out.mean_pct.iloc[0]
    out["edge_kept"] = out.mean_pct / first if first > 0 else np.nan     # share of the first row's edge
    return out


def where_is_the_money(minute_bars: dict, signals: pd.DataFrame, clock_min: int = 1,
                       side: int = +1) -> dict:
    """Split each trade's return at `clock_min` minutes: the move from the opening
    print to the first price a late order could get, and the move from there to the exit."""
    px = _entries(minute_bars, signals, (0, clock_min))
    exit_px = signals.exit_px.to_numpy(float)
    ok = (px > 0).all(axis=1) & np.isfinite(exit_px)
    p0, p1, x = px[ok, 0], px[ok, 1], exit_px[ok]
    return dict(n=int(ok.sum()), open_to_clock_pct=100 * side * float(np.mean(p1 / p0 - 1)),
                clock_to_exit_pct=100 * side * float(np.mean(x / p1 - 1)))


def engine_delay_sweep(sessions: list[pd.DataFrame],
                       signal: Callable[[pd.DataFrame, pd.Timestamp], Optional[Position]],
                       delays=(1, 2, 5), **engine_kw) -> pd.DataFrame:
    """For strategies with stops and targets: re-run the event engine at each fill delay.
    A later fill sits nearer the target and further from the stop, and some planned
    trades no longer exist at all; a fixed exit price would hide both."""
    rows = []
    for d in delays:
        trades = [t for bars in sessions for t in run_day(bars, signal, fill_delay_bars=d, **engine_kw)]
        gross = np.array([t.direction * (t.exit - t.entry) for t in trades])
        net = np.array([t.pnl_net / t.qty for t in trades])
        rows.append(dict(fill_delay_bars=d, trades=len(trades),
                         gross_points=gross.mean() if len(trades) else np.nan,
                         net_points=net.mean() if len(trades) else np.nan))
    return pd.DataFrame(rows)
