"""
Book 2, Chapter 4 — Ten traps that quietly corrupt your data. Illustrative.

Small, separate defences for the traps that are not specific to option files:
marking positions at the right price, lot-size history, a second-source check
for derived series, an overnight-gap guard for corporate actions, and
point-in-time index membership.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from algo_builder.infra.option_bhavcopy_traps import leg_exit_price


def mark_price(row: pd.Series, is_expiry: bool, underlying_settle: float) -> float:
    """The price to value one option row at. Outside expiry day, the settle price:
    an untraded strike's 'close' can be a trade from days ago. On expiry day, the
    close if it is genuine, else intrinsic value against the underlying's settlement."""
    if not is_expiry:
        return float(row.settle)
    is_call = row.opt_type == "CE"
    return leg_exit_price(float(row.close), underlying_settle, float(row.strike), is_call)


def lot_size_on(dates: pd.Series, history: pd.DataFrame) -> pd.Series:
    """Lot size in force on each date. `history`: columns effective_from, lot_size,
    one row per change, taken from exchange circulars or dated instrument masters."""
    h = history.sort_values("effective_from")
    d = pd.DataFrame({"date": pd.to_datetime(dates).values}).reset_index()
    out = pd.merge_asof(d.sort_values("date"), h, left_on="date", right_on="effective_from")
    return out.sort_values("index").set_index("index").lot_size.rename(None)


def drift_check(derived: pd.Series, reference: pd.Series, tol_pct: float = 0.5) -> pd.DataFrame:
    """Compare a derived or cached series with an independent one, day by day, and
    summarise by year. A healthy series stays inside `tol_pct` almost every day."""
    both = pd.concat({"derived": derived, "reference": reference}, axis=1).dropna()
    gap = 100 * (both.derived / both.reference - 1)
    by_year = gap.groupby(gap.index.year)
    return pd.DataFrame({"days": by_year.size(), "median_gap_pct": by_year.median(),
                         "worst_gap_pct": by_year.apply(lambda g: g.abs().max()),
                         "days_outside_tol": by_year.apply(lambda g: int((g.abs() > tol_pct).sum()))})


def suspicious_overnight_gaps(open_: pd.DataFrame, close: pd.DataFrame,
                              low: float = 0.75, high: float = 1.40) -> pd.DataFrame:
    """True where today's open is implausibly far from yesterday's close for a normal
    day: the signature of an unadjusted split, bonus or demerger, or a bad tick.
    Such rows are checked against corporate-action records, never traded blindly."""
    ratio = open_ / close.shift(1)
    return (ratio <= low) | (ratio >= high)


def members_on(day: pd.Timestamp, snapshots: pd.DataFrame) -> set[str]:
    """Index members as they were on `day`, from dated snapshots you have kept
    (columns: as_of, symbol). Today's list run backwards is survivorship bias."""
    known = snapshots[snapshots.as_of <= day]
    if known.empty:
        return set()
    latest = known.as_of.max()
    return set(known[known.as_of == latest].symbol)


def to_points(pnl_rupees: pd.Series, lot_size: pd.Series, lots: pd.Series | int = 1) -> pd.Series:
    """Rupee P&L back to index points, using the lot size in force at each trade."""
    return pnl_rupees / (np.asarray(lot_size) * np.asarray(lots))
