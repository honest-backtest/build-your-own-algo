import numpy as np
import pandas as pd

from algo_builder.strategies.hourly_signal_to_options import shadow_row, stop_fill
from algo_builder.strategies.rsi_macd_dipbuy import daily_scan, dead_filters


def _bars(o, h, lo, c):
    idx = pd.date_range("2026-01-05 09:15", periods=len(c), freq="h")
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c}, index=idx)


def test_stop_fills_differ_by_clock():
    b = _bars([100, 99, 96, 97], [100, 99, 97, 98], [98, 95, 94, 96], [99, 98, 96.5, 97.5])
    assert stop_fill(b, 96.0, "intrabar") == (b.index[1], 96.0)          # wick touches: filled at the stop
    assert stop_fill(b, 96.0, "close") is None                           # no close below 96
    gap = _bars([100, 94, 95], [100, 95, 96], [99, 93, 94], [99.5, 94, 95.5])
    assert stop_fill(gap, 96.0, "intrabar") == (gap.index[1], 94.0)      # gapped through: the open
    assert stop_fill(gap, 96.0, "close") == (gap.index[2], 95.0)         # next bar's open
    row = shadow_row("T1", gap, 96.0, live_mode="close")
    assert row["shadow_mode"] == "intrabar" and row["shadow_stop_px"] == 94.0


def test_dead_filters_names_the_unused_input():
    data = list(range(20))
    run = lambda d, p: [x for x in d if x < p["limit"]]
    assert dead_filters(run, data, dict(limit=10, adx=20), dict(limit=5, adx=40)) == ["adx"]


def test_watchlist_entries_expire():
    idx = pd.bdate_range("2026-01-01", periods=60)
    df = pd.DataFrame({"close": np.linspace(100, 130, 60)}, index=idx)
    watch = {"A": idx[10]}
    never = lambda s: pd.Series(False, index=s.index)
    rsi = lambda s: pd.Series(50.0, index=s.index)
    assert not daily_scan(df, watch, "A", rsi, never, max_watch_days=20) and "A" not in watch
    watch = {"A": idx[10]}
    daily_scan(df, watch, "A", rsi, never)
    assert "A" in watch                                                  # no expiry unless asked
