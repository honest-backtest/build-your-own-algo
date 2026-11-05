"""
Book 2, Chapter 19 — One signal, three vehicles, two monitoring clocks.
regime_tag() is also used in Chapter 8.

Shows: a persisted oversold watchlist, a MACD-cross-above-average entry at the
       close, a stop/target/trail/time-stop exit set, regime tagging at entry,
       and the every-filter test: each configured filter must change at least one
       decision on a fixture dataset, or it is reported as dead.

Values ILLUSTRATIVE.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

P = dict(rsi_oversold=30, sma_n=50, sl_pct=8, tp_pct=20,     # ILLUSTRATIVE
         trail_activate_pct=10, trail_pct=6, max_hold_days=40)


def daily_scan(df: pd.DataFrame, watch: dict, sym: str, rsi, macd_cross_up,
               max_watch_days: int | None = None) -> bool:
    """df: completed daily bars. `watch` maps symbol -> the date it became oversold,
    and is persisted in state between runs. Returns True to BUY at today's close.
    A share oversold more than `max_watch_days` sessions ago is dropped (None: never)."""
    today = df.index[-1]
    if rsi(df.close).iloc[-1] <= P["rsi_oversold"]:
        watch.setdefault(sym, today)                         # persist in state
    if sym in watch and max_watch_days is not None and \
            len(df.loc[watch[sym]:]) - 1 > max_watch_days:
        del watch[sym]                                       # the dip is old news
    sma = df.close.rolling(P["sma_n"]).mean().iloc[-1]
    if sym in watch and macd_cross_up(df.close).iloc[-1] and df.close.iloc[-1] > sma:
        del watch[sym]
        return True
    return False


def regime_tag(index_close: pd.Series, vix: float) -> dict:
    """Regime at the moment of entry. `index_close` must END at the last close known
    before the entry; without 200 closes the trend is 'unknown', never 'down'."""
    sma = index_close.rolling(200).mean().iloc[-1]
    trend = "unknown" if pd.isna(sma) else "up" if index_close.iloc[-1] > sma else "down"
    return {"index_trend": trend, "vix_bucket": "low" if vix < 14 else "mid" if vix < 20 else "high"}


def exit_check(pos: dict, high: float, low: float, close: float, days_held: int) -> str | None:
    """One completed daily bar. The stop and target are orders at the exchange, so
    they are checked against the bar's low and high (stop first); the trail and the
    time stop are decisions taken at the close. Fill prices are the engine's job."""
    e = pos["entry"]
    pos["peak"] = max(pos.get("peak", e), high)
    if low <= e * (1 - P["sl_pct"] / 100):
        return "SL"
    if high >= e * (1 + P["tp_pct"] / 100):
        return "TP"
    if pos["peak"] >= e * (1 + P["trail_activate_pct"] / 100) and \
            close <= pos["peak"] * (1 - P["trail_pct"] / 100):
        return "TRAIL"
    if days_held >= P["max_hold_days"]:
        return "TIME"
    return None


def dead_filters(run_backtest: Callable[[object, dict], list], fixture, params: dict,
                 alternatives: dict) -> list[str]:
    """Change each filter in turn to its alternative value and re-run on a fixture
    built to exercise it. Returns the filters that changed no decision at all: a
    configured input that the code never reads looks exactly like one of these."""
    base = run_backtest(fixture, params)
    return [k for k, v in alternatives.items() if run_backtest(fixture, dict(params, **{k: v})) == base]
