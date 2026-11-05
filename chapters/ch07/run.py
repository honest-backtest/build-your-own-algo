"""
Chapter 7 — Benchmarks that fight back.

A SYNTHETIC market of 40 stocks that drift upwards, with a few sharp market-wide
falls followed by rebounds. Three tests:
  1. a dip-buyer with no skill: positive returns, no excess over random entries;
  2. a 'crash buyer' (like the insider study): large excess over random days in the
     same stocks, none over random stocks on the same days. Timing, not selection;
  3. a swing strategy on a list against simply holding the list.

    python chapters/ch07/run.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from algo_builder.robustness.benchmarks import hold_the_list, same_day_means
from algo_builder.robustness.drift_benchmark import drift_report, random_entry_means

N_STOCKS, N_DAYS, HOLD = 40, 1500, 40


def market(seed: int = 3) -> tuple[pd.DataFrame, list[int]]:
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0.0004, 0.009, N_DAYS)
    crash_ends = [300, 700, 1100]
    for e in crash_ends:                                     # a fall, then a rebound
        mkt[e - 15:e] -= 0.012
        mkt[e:e + 30] += 0.007
    beta = rng.uniform(0.7, 1.3, N_STOCKS)
    idio = rng.normal(0.0, 0.015, (N_DAYS, N_STOCKS))
    rets = mkt[:, None] * beta + idio
    days = pd.bdate_range("2019-01-01", periods=N_DAYS)
    closes = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=days,
                          columns=[f"S{i:02d}" for i in range(N_STOCKS)])
    return closes, crash_ends


def trades_at(closes: pd.DataFrame, picks: list[tuple[int, str]], hold: int) -> pd.DataFrame:
    c = closes.to_numpy()
    rows = [dict(symbol=s, entry_idx=i, hold=hold,
                 ret=c[i + 1 + hold, closes.columns.get_loc(s)] / c[i + 1, closes.columns.get_loc(s)] - 1)
            for i, s in picks if i + 1 + hold < len(c)]
    return pd.DataFrame(rows)


def report(name: str, closes: pd.DataFrame, t: pd.DataFrame) -> None:
    r = drift_report(t, random_entry_means(closes, t, n_draws=500))
    d = drift_report(t, same_day_means(closes, t, n_draws=500))
    print(f"{name}: {len(t)} trades, mean {r['strategy_mean_pct']:+.2f}%")
    print(f"   vs random days, same stocks: random {r['random_mean_pct']:+.2f}%, "
          f"excess {r['excess_pct']:+.2f}%, p = {r['p_value']:.3f}")
    print(f"   vs random stocks, same days: random {d['random_mean_pct']:+.2f}%, "
          f"excess {d['excess_pct']:+.2f}%, p = {d['p_value']:.3f}")


def main() -> None:
    closes, crash_ends = market()
    rng = np.random.default_rng(9)

    five_day = closes.pct_change(5)
    dips = [(i, s) for i in range(100, N_DAYS - HOLD - 2, 7) for s in closes.columns
            if five_day.iloc[i][s] < -0.05]                   # a 5% fall in a week: no skill in it
    report("1. Dip-buyer", closes, trades_at(closes, dips, HOLD))

    crash = [(e + int(rng.integers(0, 3)), s) for e in crash_ends
             for s in rng.choice(closes.columns, 15, replace=False)]
    report("2. Crash buyer", closes, trades_at(closes, crash, HOLD))

    held = hold_the_list(closes, list(closes.columns[:10]), closes.index[0], closes.index[-1])
    in_market = closes.iloc[:, :10].pct_change().where(five_day.iloc[:, :10].shift(1) < 0)
    swing = float((1 + in_market.mean(axis=1).fillna(0)).prod() - 1)
    print(f"3. On a list of 10 names: hold the list {held['total_return_pct']:+.0f}%, "
          f"swing (in only after a down week) {100 * swing:+.0f}%")


if __name__ == "__main__":
    main()
