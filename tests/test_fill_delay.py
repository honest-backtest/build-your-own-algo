from datetime import time

import numpy as np
import pandas as pd

from algo_builder.backtest.event_engine import Position
from algo_builder.robustness.fill_delay_test import (engine_delay_sweep, fill_delay_table,
                                                     where_is_the_money)


def _day(d, opens):
    idx = pd.date_range(pd.Timestamp(d) + pd.Timedelta("9h15min"), periods=len(opens), freq="1min")
    o = np.asarray(opens, float)
    return pd.DataFrame({"open": o, "high": o, "low": o, "close": o}, index=idx)


def test_every_delay_uses_the_same_trades():
    bars = {("A", 1): _day("2026-01-05", [100] + [101] * 29),
            ("B", 2): _day("2026-01-06", [100, 101, 101])}          # no bar 5 or 15 minutes in
    sig = pd.DataFrame({"symbol": ["A", "B"], "date": [1, 2], "exit_px": [102.0, 102.0]})
    tab = fill_delay_table(bars, sig, delays=(0, 1, 5, 15))
    assert (tab.n == 1).all()                                        # B dropped from every row


def test_no_entry_at_or_after_the_exit():
    bars = {("A", 1): _day("2026-01-05", [100] + [101] * 29)}
    exit_ts = pd.Timestamp("2026-01-05 09:20")
    sig = pd.DataFrame({"symbol": ["A"], "date": [1], "exit_px": [102.0], "exit_ts": [exit_ts]})
    assert (fill_delay_table(bars, sig, delays=(0, 1, 5)).n == 0).all()
    assert (fill_delay_table(bars, sig, delays=(0, 1, 4)).n == 1).all()


def test_edge_kept_needs_a_positive_first_row():
    bars = {("A", 1): _day("2026-01-05", [100] + [99] * 29)}
    sig = pd.DataFrame({"symbol": ["A"], "date": [1], "exit_px": [98.0]})
    assert fill_delay_table(bars, sig, delays=(0, 1)).edge_kept.isna().all()


def test_where_is_the_money_respects_the_side():
    bars = {("I", 1): _day("2026-01-05", [100.5] + [100.0] * 29)}   # a print that is too high
    sig = pd.DataFrame({"symbol": ["I"], "date": [1], "exit_px": [100.0]})
    w = where_is_the_money(bars, sig, clock_min=1, side=-1)
    assert w["open_to_clock_pct"] > 0.49 and abs(w["clock_to_exit_pct"]) < 1e-12


def test_engine_sweep_charges_a_late_breakout():
    prices = [100.0] * 31 + [101, 102, 103, 104, 105, 106] + [106.0] * 20
    bars = _day("2026-01-05", prices)
    bars["high"], bars["low"] = bars.open + 0.1, bars.open - 0.1

    def sig(h, ts):
        if len(h) > 31 and h.close.iloc[-1] > 100.5 and h.close.iloc[-2] <= 100.5:
            px = float(h.close.iloc[-1])
            return Position(+1, px, 1, px - 3, px - 3, px + 4, ts)
        return None

    t = engine_delay_sweep([bars], sig, delays=(1, 3), range_end=time(9, 45),
                           square_off=time(10, 10), slip_pct=0.0).set_index("fill_delay_bars")
    assert t.loc[1, "gross_points"] > t.loc[3, "gross_points"]
