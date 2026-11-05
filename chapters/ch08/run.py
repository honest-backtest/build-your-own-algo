"""
Chapter 8 — Slicing the result: by year, ex-top-N and regime tags.

Four SYNTHETIC demonstrations:
  1. a 'lottery ticket': about 670 trades, slightly positive, carried by its 10 best;
  2. a crash-day strategy carried by a few busy days;
  3. a regime specialist, by year;
  4. the same trades broken down by VIX bucket, tagged two ways: with the regime
     known before entry, and with the entry day's own closing VIX (look-ahead).

    python chapters/ch08/run.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from algo_builder.robustness.by_year_and_ex_top_n import (by_year, concentration_by_day, ex_top_days,
                                                          ex_top_n)
from algo_builder.robustness.regime import by_regime, daily_regimes, tag_at_entry


def lottery_ticket(seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pnl = rng.normal(-2.0, 20.0, 670)                          # points; a small, steady bleed
    pnl[rng.choice(670, 10, replace=False)] = rng.uniform(90, 160, 10)    # ten big days
    pnl += (40.0 - pnl.sum()) / 670                           # total of about +40 points
    days = pd.bdate_range("2021-01-01", periods=670)
    return pd.DataFrame({"entry_ts": days, "pnl": pnl})


def crash_day_strategy(seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2019-01-01", "2025-12-31")
    rows = [(d, rng.normal(-0.05, 1.0)) for d in rng.choice(days, 260, replace=False)]
    for d in rng.choice(days, 6, replace=False):                    # six crash mornings
        rows += [(d, rng.normal(1.2, 1.5)) for _ in range(28)]       # many signals, each a winner on average
    return pd.DataFrame(rows, columns=["entry_ts", "pnl"])          # per-trade % returns


def regime_specialist(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-01", "2025-12-31")
    picks = np.sort(rng.choice(days, 600, replace=False))
    mean = np.where(pd.DatetimeIndex(picks).year == 2020, 2.5, -0.15)    # 2020 rebound carries it
    return pd.DataFrame({"entry_ts": picks, "pnl": rng.normal(mean, 4.0)})


def market(seed: int = 4) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2017-01-01", "2025-12-31")
    r = rng.normal(0.0004, 0.011, len(days))
    vix = np.empty(len(days))
    vix[0] = 15.0
    for i in range(1, len(days)):                                  # VIX jumps on falling days
        vix[i] = max(9.0, 0.93 * vix[i - 1] + 0.07 * 15.0 - 250 * r[i])
    return pd.Series(10_000 * np.exp(np.cumsum(r)), days), pd.Series(vix, days)


def main() -> None:
    t = lottery_ticket()
    print("1. A lottery ticket (synthetic, index points)")
    print(ex_top_n(t, ns=(0, 1, 5, 10, 20)).round(1).to_string(index=False))

    c = crash_day_strategy()
    con = concentration_by_day(c, top_days=6)
    print("\n2. A crash-day strategy (synthetic, % per trade)")
    print(f"   {con['share_of_trades']:.0%} of trades and {con['share_of_pnl']:.0%} of the profit "
          f"fall on its 6 busiest days")
    print(ex_top_days(c, ns=(0, 1, 3, 6)).round(1).to_string(index=False))

    print("\n3. A regime specialist, by year (synthetic, % per trade)")
    print(by_year(regime_specialist()).round(2).to_string())

    close, vix = market()
    trades = pd.DataFrame({"entry_ts": close.index[1:], "pnl": np.diff(np.log(close.to_numpy())) * 100})
    known = tag_at_entry(trades, daily_regimes(close, vix))
    leaked = trades.assign(vix_bucket=daily_regimes(close, vix).vix_bucket.iloc[1:].to_numpy())
    print("\n4. One long trade a day, open to close (synthetic), by VIX bucket")
    print("   tagged with the regime known BEFORE entry:")
    print(by_regime(known, "vix_bucket")[["trades", "mean", "profit_factor"]].round(2).to_string())
    print("   tagged with the entry day's OWN closing VIX (look-ahead):")
    print(by_regime(leaked, "vix_bucket")[["trades", "mean", "profit_factor"]].round(2).to_string())


if __name__ == "__main__":
    main()
