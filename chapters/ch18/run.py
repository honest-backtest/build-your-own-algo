"""
Chapter 18 — A quarterly cycle on real prices: the DIY principal-protected note.

32 SYNTHETIC quarters: an index path with one sharp crash, a safe yield that dips
for a while, and a chain of quarterly calls per quarter priced by the model at a
modest implied volatility (synthetic data has no option prices; the method in the
book uses real ones). Strikes far from the money are listed but untraded.

  1. The structure against holding the index: CAGR and maximum drawdown.
  2. The same structure at three sizes of capital: whole lots decide participation.

Every setting is ILLUSTRATIVE.

    python chapters/ch18/run.py
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from algo_builder.strategies.opening_range_options import bs_price
from algo_builder.strategies.ppn_quarterly_call import Cycle, run

LOT = 75                                                       # ILLUSTRATIVE


def quarters(seed: int = 18) -> list[Cycle]:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2018-04-01", periods=33, freq="QS").date
    rets = rng.normal(0.025, 0.07, 32)
    rets[8] = -0.27                                             # one sharp crash
    rets[9] = 0.18                                              # and a rebound
    yields = np.full(32, 0.068)
    yields[8:14] = 0.036                                        # a period of low rates
    s, out = 10_000.0, []
    for i in range(32):
        T = (days[i + 1] - days[i]).days / 365
        rows = []
        for k in np.arange(round(s * 0.8 / 50) * 50, s * 1.2, 50):
            close = bs_price(s, k, T, 0.14, True)
            traded = abs(k / s - 1) < 0.10                        # far strikes listed, untraded
            rows.append(dict(strike=k, opt_type="CE", close=close, contracts=500 if traded else 0))
        st = s * (1 + rets[i])
        out.append(Cycle(days[i], days[i + 1] - timedelta(days=1), pd.DataFrame(rows), s, st, yields[i], LOT))
        s = st
    return out


def stats(values: pd.Series, years: float) -> tuple[float, float]:
    cagr = (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1
    dd = (values / values.cummax() - 1).min()
    return cagr, dd


def main() -> None:
    cyc = quarters()
    years = (cyc[-1].expiry - cyc[0].day).days / 365.25
    idx = pd.Series([cyc[0].S0] + [c.ST for c in cyc])
    print("1. 32 synthetic quarters")
    c, d = stats(idx, years)
    print(f"   hold the index:            CAGR {c:6.1%}, max drawdown {d:6.1%}")
    for capital, label in ((5_00_000, "5 lakh"), (30_00_000, "30 lakh"), (1_00_00_000, "1 crore")):
        r = run(cyc, capital)
        eq = pd.Series([capital] + list(r.fund))
        c, d = stats(eq, years)
        print(f"   structure, Rs {label:>7}:    CAGR {c:6.1%}, max drawdown {d:6.1%}, "
              f"quarters with no lot {int((r.lots == 0).sum()):2d}, killed {int((r.note != '').sum()):2d}, "
              f"average participation {r.participation.mean():.0%}")


if __name__ == "__main__":
    main()
