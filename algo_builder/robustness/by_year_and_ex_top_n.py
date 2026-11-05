"""
Book 2, Chapter 8 — Slicing the result: by year, ex-top-N and regime tags.

Two questions every result must answer:
  * Does it work in most years, or is the total carried by one regime?
  * Does it survive removing its N best trades (or N best DAYS)? A result that
    flips sign without its top 10 trades is a lottery ticket, not an edge.

Input: trades with  entry_ts, pnl  (net).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def by_year(trades: pd.DataFrame) -> pd.DataFrame:
    g = trades.groupby(pd.to_datetime(trades.entry_ts).dt.year).pnl
    out = pd.DataFrame({"trades": g.size(), "sum": g.sum(), "mean": g.mean(),
                        "win_rate": g.apply(lambda x: (x > 0).mean())})
    out["positive"] = out["sum"] > 0
    return out


def ex_top_n(trades: pd.DataFrame, ns=(0, 1, 5, 10, 20)) -> pd.DataFrame:
    p = trades.pnl.sort_values(ascending=False).to_numpy()
    rows = []
    for n in ns:
        rest = p[n:]
        rows.append(dict(removed_top=n, total=rest.sum(), mean=rest.mean() if len(rest) else np.nan))
    return pd.DataFrame(rows)


def ex_top_days(trades: pd.DataFrame, ns=(0, 1, 3, 5, 10)) -> pd.DataFrame:
    """As ex_top_n, but removes the N best DAYS: strategies that take many trades on
    one day (a crash morning) can be carried by a few days, not a few trades."""
    daily = (trades.groupby(pd.to_datetime(trades.entry_ts).dt.normalize()).pnl.sum()
             .sort_values(ascending=False).to_numpy())
    rows = [dict(removed_top_days=n, total=daily[n:].sum()) for n in ns]
    return pd.DataFrame(rows)


def concentration_by_day(trades: pd.DataFrame, top_days: int = 10) -> dict:
    """Share of trades and of P&L that fall on the busiest days: a capacity check.
    The P&L share is reported only when the total is positive; a share of a loss
    or of nothing has no useful meaning."""
    day = pd.to_datetime(trades.entry_ts).dt.normalize()
    counts = day.value_counts()
    busiest = counts.index[:top_days]
    on = day.isin(busiest)
    total = trades.pnl.sum()
    return dict(share_of_trades=float(on.mean()),
                share_of_pnl=float(trades.pnl[on].sum() / total) if total > 0 else np.nan,
                busiest_days=[d.date().isoformat() for d in busiest])
