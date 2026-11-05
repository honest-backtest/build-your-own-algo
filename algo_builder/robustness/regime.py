"""
Book 2, Chapter 8 — regime tags, as known at entry. Illustrative.

Tag every trade with the market's state as it was BEFORE the trade was entered,
then break results down by tag. A label computed from the entry day's own close
(or the end-of-day day type) knows how the day turned out, and quietly turns the
breakdown into a forecast of the past.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VIX_EDGES = (14.0, 20.0)                 # ILLUSTRATIVE: low below 14, high from 20


def daily_regimes(index_close: pd.Series, vix: pd.Series, sma_n: int = 200) -> pd.DataFrame:
    """One row per trading day: the regime implied by THAT day's close."""
    d = pd.DataFrame({"close": index_close, "vix": vix}).sort_index()
    sma = d.close.rolling(sma_n).mean()
    d["index_trend"] = np.where(sma.isna(), "unknown", np.where(d.close > sma, "up", "down"))
    lo, hi = VIX_EDGES
    d["vix_bucket"] = np.select([d.vix < lo, d.vix < hi], ["low", "mid"], "high")
    return d[["index_trend", "vix_bucket"]]


def tag_at_entry(trades: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    """Attach the regime from the last trading day STRICTLY before each entry's day."""
    t = trades.copy()
    t["_day"] = pd.to_datetime(t.entry_ts).dt.normalize()
    t["_order"] = np.arange(len(t))
    r = regimes.copy()
    r.index = pd.to_datetime(r.index).normalize()
    out = pd.merge_asof(t.sort_values("_day"), r.rename_axis("_known").reset_index(),
                        left_on="_day", right_on="_known", allow_exact_matches=False)
    out[["index_trend", "vix_bucket"]] = out[["index_trend", "vix_bucket"]].fillna("unknown")
    return out.sort_values("_order").drop(columns=["_day", "_order", "_known"]).set_index(trades.index)


def by_regime(trades: pd.DataFrame, col: str) -> pd.DataFrame:
    """Results by any tag column: regime, weekday, days to expiry, setup."""
    def pf(p: pd.Series) -> float:
        losses = -p[p < 0].sum()
        return p[p > 0].sum() / losses if losses else np.inf

    g = trades.groupby(col, observed=True).pnl
    out = pd.DataFrame({"trades": g.size(), "total": g.sum(), "mean": g.mean(),
                        "profit_factor": g.apply(pf)})
    total = trades.pnl.sum()
    out["share_of_total"] = out.total / total if total > 0 else np.nan
    return out
