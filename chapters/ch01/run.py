"""
Chapter 1 — The trading you need to know, in one chapter.

Reproduces Listing 1.5: scoring the chapter's ten illustrative trades with the
report-card code (Listings 1.3 and 1.4), and demonstrates Listings 1.1 and 1.2.

    python chapters/ch01/run.py
"""
from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from algo_builder.backtest.metrics import max_drawdown, realised_equity, trade_stats
from algo_builder.common.bars import completed_bars
from algo_builder.common.contract import Contract


def ten_trades() -> pd.DataFrame:
    pnl = [-1000, -1000, 2500, -1000, -600, 3200, -1000, -1000, 1800, -400]
    days = pd.bdate_range("2026-01-05", periods=10)          # illustrative dates
    t = pd.DataFrame({"entry_ts": days + pd.Timedelta("9h30min"),
                      "exit_ts": days + pd.Timedelta("15h"), "pnl": pnl})
    t["r"] = t.pnl / 1000                                     # 1R = Rs 1,000 at entry
    return t


def main() -> None:
    trades = ten_trades()
    print("Report card:", {k: round(v, 2) for k, v in trade_stats(trades).items()})
    print(f"Max drawdown: {max_drawdown(realised_equity(trades, 100_000)):.1%}")

    bars = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
                        index=pd.date_range("2026-01-05 09:15", periods=4, freq="15min"))
    kept = completed_bars(bars, datetime(2026, 1, 5, 9, 50), "15min")
    print("Completed at 09:50:", list(kept.index.strftime("%H:%M")))

    call = Contract("ILLUSTRATIVE-CE", "NIFTY", "CE", date(2026, 1, 27), 50, 25_000)
    print("100 points on one lot:", call.rupees(100), "| intrinsic at 25,250:", call.intrinsic(25_250))


if __name__ == "__main__":
    main()
