from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from algo_builder.strategies.expiry_straddle_framework import (daily_straddle, legs_on_one_clock,
                                                               record_quotes, spread_stats)


def test_daily_straddle_settles_overwritten_closes():
    chain = pd.DataFrame({"strike": [25_000.0, 25_000.0], "opt_type": ["CE", "PE"],
                          "open": [60.0, 60.0], "close": [25_040.0, 25_040.0]})   # both overwritten
    assert np.isclose(daily_straddle(chain, 25_000.0, underlying_settle=25_040.0), 100 * (120 - 40) / 120)
    assert daily_straddle(chain[chain.opt_type == "CE"], 25_000.0, 25_040.0) is None


class _Broker:
    def __init__(self):
        self.n = 0

    def get_quote(self, symbol, exchange):
        self.n += 1
        return None if self.n == 3 else dict(bid=10.0, ask=11.0, bid_qty=75, ask_qty=75, ltp=10.5)


def test_recorder_writes_gaps_and_uses_the_given_clock(tmp_path):
    out, loops, t0 = tmp_path / "s.csv", iter(range(3)), datetime(2026, 1, 13, 9, 15)
    times = iter([t0 + timedelta(seconds=5 * i) for i in range(3)])
    record_quotes(_Broker(), [("A", "NFO"), ("B", "NFO")], str(out), every_s=5,
                  until=lambda: next(loops, None) is None, clock=lambda: next(times), sleep=lambda s: None)
    rec = pd.read_csv(out)
    assert len(rec) == 6 and rec.bid.isna().sum() == 1 and rec.ts.iloc[0] == "2026-01-13T09:15:00"


def test_legs_on_one_clock_refuses_stale_quotes():
    rec = pd.DataFrame({"ts": ["2026-01-13T10:00:00", "2026-01-13T10:00:00", "2026-01-13T10:00:05",
                               "2026-01-13T10:00:30"],
                        "symbol": ["C", "P", "C", "C"], "bid": [10.0, 9.0, 11.0, 12.0],
                        "ask": [11.0, 10.0, 12.0, 13.0]})
    both = legs_on_one_clock(rec, "C", "P", max_stale="10s")
    assert both.straddle.iloc[0] == 20.0 and both.straddle.iloc[1] == 21.0
    assert np.isnan(both.straddle.iloc[2])                 # the put was last seen 30 seconds ago
    assert spread_stats(rec).loc["C", "median"] == 1.0
