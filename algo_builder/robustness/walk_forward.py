"""
Book 2, Chapter 10 — Walk-forward: tuning without fooling yourself.

Yearly walk-forward: for each test year, pick the parameter combo with the best
TRAILING training result (by default mean P&L per trade, in rupees or points, not
R, which is easy to corrupt), require a minimum trade count so a 3-trade fluke
can't win, then record ONLY the out-of-sample year. The stitched OOS series is
the honest result. Training trades that were still open when the test year began
are left out of the training window.

`run_combo(combo) -> trades DataFrame` with columns  entry_ts, pnl  (and exit_ts,
if trades can span a year end). Each combo runs once over the full history and is
sliced by year afterwards.

Also here: the in-sample/out-of-sample rank correlation and the plateau check.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def mean_pnl(trades: pd.DataFrame) -> float:
    return float(trades.pnl.mean())


def walk_forward(combos: list[dict], run_combo: Callable[[dict], pd.DataFrame],
                 first_test_year: int, last_test_year: int,
                 train_years: int | None = None, min_trades: int = 30,
                 score: Callable[[pd.DataFrame], float] = mean_pnl) -> pd.DataFrame:
    """One row per test year: the combo chosen on the training window alone, its
    training score, and what it then did in the test year. train_years=None trains
    on every earlier year (anchored); a number trains on that many (rolling)."""
    cache = {i: run_combo(c).assign(year=lambda t: pd.to_datetime(t.entry_ts).dt.year)
             for i, c in enumerate(combos)}
    rows = []
    for test_year in range(first_test_year, last_test_year + 1):
        lo = -10**9 if train_years is None else test_year - train_years
        start = pd.Timestamp(year=test_year, month=1, day=1)
        best, best_score = None, -np.inf
        for i, t in cache.items():
            train = t[(t.year >= lo) & (t.year < test_year)]
            if "exit_ts" in train:
                train = train[pd.to_datetime(train.exit_ts) < start]   # no peeking into the test year
            if len(train) < min_trades:
                continue
            s = score(train)
            if s > best_score:
                best, best_score = i, s
        if best is None:
            continue
        oos = cache[best][cache[best].year == test_year]
        rows.append(dict(test_year=test_year, chosen=combos[best], train_score=best_score,
                         oos_trades=len(oos), oos_pnl=oos.pnl.sum(),
                         oos_mean=oos.pnl.mean() if len(oos) else np.nan))
    return pd.DataFrame(rows)


def rank_correlation(combos: list[dict], run_combo: Callable[[dict], pd.DataFrame],
                     in_years: range, out_years: range,
                     score: Callable[[pd.DataFrame], float] = mean_pnl) -> tuple[float, pd.DataFrame]:
    """Score every combo on the in-sample years and again on the out-of-sample years,
    and return the rank correlation between the two. Near zero: the ranking is noise.
    Negative: the optimiser is choosing the combos that are about to do worst."""
    rows = []
    for c in combos:
        t = run_combo(c)
        yr = pd.to_datetime(t.entry_ts).dt.year
        rows.append(dict(combo=c, in_sample=score(t[yr.isin(in_years)]),
                         out_of_sample=score(t[yr.isin(out_years)])))
    table = pd.DataFrame(rows)
    rho = table.in_sample.rank().corr(table.out_of_sample.rank())
    return float(rho), table


def plateau_share(results: pd.DataFrame, params: list[str], metric: str = "pnl") -> pd.Series:
    """For each row of a grid of results (one row per combo, a column per parameter),
    the share of its immediate grid neighbours whose `metric` is above zero. A peak
    with losing neighbours is noise; a plateau of winners is more likely to be real."""
    pos = {p: {v: k for k, v in enumerate(sorted(results[p].unique()))} for p in params}
    idx = results[params].apply(lambda col: col.map(pos[col.name])).to_numpy()
    win = (results[metric] > 0).to_numpy()
    out = []
    for i in range(len(results)):
        near = np.abs(idx - idx[i]).max(axis=1) == 1                  # one step away in any direction
        out.append(win[near].mean() if near.any() else np.nan)
    return pd.Series(out, index=results.index, name="neighbours_positive")
