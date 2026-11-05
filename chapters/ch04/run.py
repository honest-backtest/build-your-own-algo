"""
Chapter 4 — Ten traps that quietly corrupt your data.

Three of the traps, reproduced on synthetic data:
  1. the expiry-day overwrite, on one short straddle;
  2. a cached index file that drifts from reality, caught by rebuilding the index
     from the option chain by put-call parity;
  3. lot-size history: the same rupee rule meaning different things.

    python chapters/ch04/run.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from algo_builder.infra.data_traps import drift_check, lot_size_on, mark_price, to_points
from algo_builder.infra.option_bhavcopy_traps import settlement_from_parity
from algo_builder.strategies.opening_range_options import bs_price


def expiry_overwrite() -> None:
    settle, strike, sold_for = 72_180.0, 72_000, 120.0          # illustrative SENSEX-like level
    overwritten = [pd.Series(dict(opt_type=t, strike=strike, close=settle, settle=settle)) for t in ("CE", "PE")]
    naive = sold_for - sum(r.close for r in overwritten)
    honest = sold_for - sum(mark_price(r, True, settle) for r in overwritten)
    print("1. Expiry-day overwrite, one short straddle sold for 120 points")
    print(f"   taking the file's closes at face value: {naive:,.0f} points "
          f"({naive / sold_for:,.0f} times the premium)")
    print(f"   settling at intrinsic value instead:     {honest:,.0f} points")


def chain_for(level: float, days_left: float = 3, sigma: float = 0.14) -> pd.DataFrame:
    T = days_left / 365
    k0 = round(level / 50) * 50
    rows = [dict(strike=k, opt_type=t, close=round(bs_price(level, k, T, sigma, t == "CE"), 2), contracts=500)
            for k in range(k0 - 500, k0 + 550, 50) for t in ("CE", "PE")]
    return pd.DataFrame(rows)


def corrupt_index_file() -> None:
    days = pd.bdate_range("2019-01-01", "2024-12-31")
    rng = np.random.default_rng(4)
    truth = pd.Series(11_000 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(days)))), index=days)
    cached = truth.copy()
    cached[(cached.index.year >= 2021) & (cached.index.year <= 2022)] *= 1.10   # a silently broken era
    sample = truth.iloc[::21]                                                   # one day a month
    rebuilt = pd.Series([settlement_from_parity(chain_for(v)) for v in sample], index=sample.index)
    print("2. A cached index file checked against the index rebuilt from option prices")
    print(drift_check(cached.loc[rebuilt.index], rebuilt).round(2).to_string())


def lot_history() -> None:
    hist = pd.DataFrame({"effective_from": pd.to_datetime(["2019-01-01", "2021-07-01", "2024-11-20"]),
                         "lot_size": [75, 50, 75]})                            # ILLUSTRATIVE history
    dates = pd.Series(pd.to_datetime(["2020-03-02", "2022-01-03", "2025-02-03"]))
    lots = lot_size_on(dates, hist)
    print("3. A 'stop at Rs 3,000' rule under three lot sizes (illustrative)")
    for d, lot in zip(dates, lots):
        print(f"   {d:%b %Y}: lot {lot} -> the rule allows {3000 / lot:.0f} points")
    print(f"   the same 60-point loss, back in points: {list(to_points(pd.Series([-4500.0, -3000.0, -4500.0]), lots))}")


if __name__ == "__main__":
    expiry_overwrite()
    corrupt_index_file()
    lot_history()
