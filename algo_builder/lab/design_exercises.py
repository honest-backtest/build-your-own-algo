"""
Book 2, Chapter 24 — Design exercises from the indicator-soup years.

Small tools for turning a chart-based design into something that can be tested:
an exact definition of "touched", higher-timeframe bars aligned by completion
time, a gate-ablation loop, per-recipe accounts for "any one fires" systems, and
the spread of results across short windows. ILLUSTRATIVE.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def cpr(high: float, low: float, close: float) -> tuple[float, float]:
    """Central pivot range for the next session, as (lower, upper) edges. The top
    central level mirrors the bottom one around the pivot, so it can fall below it."""
    pivot = (high + low + close) / 3
    bc = (high + low) / 2
    tc = 2 * pivot - bc
    return min(bc, tc), max(bc, tc)


def touched(bars: pd.DataFrame, level: float, rule: Literal["trade", "close"]) -> pd.Series:
    """Two testable meanings of "the price touched the level".
    'trade': the bar's range included it, even for an instant.
    'close': the bar CLOSED on the other side of it from the previous close."""
    if rule == "trade":
        return (bars.low <= level) & (bars.high >= level)
    side = np.sign(bars.close - level)
    return (side != side.shift()) & side.shift().notna() & (side != 0)


def align_completed(lower: pd.DataFrame, lower_interval: str,
                    higher: pd.DataFrame, higher_interval: str) -> pd.DataFrame:
    """Attach to each lower-timeframe bar the latest higher-timeframe bar that had
    FINISHED when the lower bar finished. Both inputs are labelled by bar start.
    Columns from the higher timeframe get the suffix '_hi'; before the first
    completed higher bar they are NaN."""
    lo = lower.copy()
    lo["_end"] = lo.index + pd.Timedelta(lower_interval)
    hi = higher.add_suffix("_hi")
    hi["_end"] = higher.index + pd.Timedelta(higher_interval)
    out = pd.merge_asof(lo.reset_index().sort_values("_end"), hi.sort_values("_end"),
                        on="_end", direction="backward", allow_exact_matches=True)
    return out.set_index(lower.index.name or "index").drop(columns="_end")


def ablate(gates: pd.DataFrame, outcome: pd.Series) -> pd.DataFrame:
    """For each gate (a boolean column; a trade needs all of them), the trades it
    removes and how those trades did. A gate earns its place only if the trades it
    removes are clearly worse than the ones it keeps. `outcome` is the net result
    of taking a trade at each row."""
    keep = gates.all(axis=1)
    rows = []
    for g in gates.columns:
        removed = gates.drop(columns=g).all(axis=1) & ~keep
        others = gates.drop(columns=g).astype(float)
        rows.append(dict(gate=g, kept=int(keep.sum()), kept_mean=outcome[keep].mean(),
                         removed=int(removed.sum()), removed_mean=outcome[removed].mean(),
                         max_corr=others.corrwith(gates[g].astype(float)).abs().max()))
    return pd.DataFrame(rows).set_index("gate")


def by_recipe(fired: pd.DataFrame, pnl: pd.Series) -> pd.DataFrame:
    """Accounts for a system that trades when ANY recipe fires. Each trade is
    credited to every recipe that fired on it, and separately to the recipe alone
    when no other fired. `shared` is the share of a recipe's trades on which at
    least one other recipe also fired."""
    many = fired.sum(axis=1) > 1
    rows = {}
    for r in fired.columns:
        on, alone = fired[r], fired[r] & ~many
        rows[r] = dict(fired=int(on.sum()), mean_pnl=pnl[on].mean(),
                       alone=int(alone.sum()), alone_mean=pnl[alone].mean(),
                       shared=float(many[on].mean()) if on.any() else np.nan)
    return pd.DataFrame.from_dict(rows, orient="index")


def window_pf(pnl: pd.Series, months: int) -> pd.DataFrame:
    """Profit factor and trade count in consecutive windows of `months` months,
    from a date-indexed series of trade results. Shows how far a short test can
    land from the long-run answer."""
    key = (pnl.index.year * 12 + pnl.index.month - 1) // months
    def pf(x):
        loss = -x[x < 0].sum()
        return x[x > 0].sum() / loss if loss else np.inf
    g = pnl.groupby(key)
    return pd.DataFrame({"start": g.apply(lambda x: x.index.min().date()),
                         "trades": g.size(), "pf": g.apply(pf)}).reset_index(drop=True)
