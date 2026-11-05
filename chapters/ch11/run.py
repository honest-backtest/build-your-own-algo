"""
Chapter 11 — The default report.

Two SYNTHETIC intraday index-futures strategies, about 670 trades each over
2021-2026, one lot of 50, 15 points of cost a round trip, ₹5 lakh of capital:
  1. 'steady': a small edge spread evenly across the trades;
  2. 'lottery ticket': the same total, carried by ten big days (as in Chapter 8).
Both get the same default report; the run is stamped and saved.

    python chapters/ch11/run.py
"""
from __future__ import annotations

import tempfile
from datetime import datetime

import numpy as np
import pandas as pd

from algo_builder.robustness.report import robustness_report, run_stamp, save_run

LOT, COST_PTS, CAPITAL = 50, 15.0, 500_000


def strategy(kind: str, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 670
    if kind == "steady":
        gross = rng.normal(19.0, 45.0, n)
    else:
        gross = rng.normal(13.0, 45.0, n)
        gross[rng.choice(n, 10, replace=False)] = rng.uniform(250, 400, 10)
    gross += (19.0 * n - gross.sum()) / n                         # the same gross total for both
    days = pd.bdate_range("2021-01-01", "2026-06-30")
    entry = pd.to_datetime(np.sort(rng.choice(days, n, replace=False))) + pd.Timedelta("10h")
    return pd.DataFrame({"entry_ts": entry, "exit_ts": entry + pd.Timedelta("4h"),
                         "gross_pts": gross, "pnl": (gross - COST_PTS) * LOT})


def main() -> None:
    pd.set_option("display.width", 160, "display.max_colwidth", 90)
    for kind in ("steady", "lottery ticket"):
        t = strategy(kind)
        rep = robustness_report(t, CAPITAL, cost_pts=COST_PTS)
        print(f"\n{kind}: {len(t)} trades, net total ₹{t.pnl.sum():,.0f} (synthetic)")
        print(rep["checks"].to_string(index=False))
    stamp = run_stamp("Backtest", dict(strategy="steady", lot=LOT, cost_pts=COST_PTS),
                      "2021-01-01", "2026-06-30", "FUTURES 2026", now=datetime(2026, 10, 30, 10, 0))
    with tempfile.TemporaryDirectory() as d:
        t = strategy("steady")
        out = save_run(d, stamp, robustness_report(t, CAPITAL, cost_pts=COST_PTS), t)
        print("\nsaved:", sorted(p.name for p in out.iterdir()))
    print({k: v for k, v in stamp.items() if k != "params"})


if __name__ == "__main__":
    main()
