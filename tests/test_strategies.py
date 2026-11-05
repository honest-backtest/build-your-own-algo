import numpy as np
import pandas as pd

from algo_builder.data.synthetic import daily_panel, minute_bars, resample, trading_days
from algo_builder.strategies.buy_on_gap import live_qty, scan
from algo_builder.strategies.compounder_screen import cache_parse_result, table_kind
from algo_builder.strategies.covered_call_framework import beta_matched_lots, expected_move
from algo_builder.strategies.market_profile_framework import (DayTypeMachine, InitialBalance,
                                                                 value_area)
from algo_builder.strategies.pullback_in_uptrend import prep
from algo_builder.strategies.rsi_macd_futures import Pos, check_exit
from algo_builder.strategies.supertrend_rsi import Signaller


def test_initial_balance_is_blind_until_closed():
    from datetime import time
    bars = resample(minute_bars(trading_days("2026-01-05", "2026-01-05")), "15min")
    ib = InitialBalance(window_end=time(10, 15))
    for ts, b in bars.iterrows():
        ib.update(ts, b)
        if ts.time() < time(10, 15):
            assert np.isnan(ib.width)
    assert ib.closed and ib.width > 0


def test_day_type_machine_records_changes():
    m = DayTypeMachine(rules=[("TREND", lambda c: c["x"] > 1), ("NORMAL", lambda c: True)])
    m.update(pd.Timestamp("2026-01-05 10:30"), {"x": 0})
    m.update(pd.Timestamp("2026-01-05 11:30"), {"x": 2})
    assert [lab for _, lab in m.history] == ["NORMAL", "TREND"]


def test_value_area_contains_poc():
    px = np.array([100, 100, 101, 101, 101, 102, 103])
    val, vah, poc = value_area(px, np.ones(7), tick=1, share=0.7)
    assert val <= poc <= vah and poc == 101


def test_trailing_stop_only_ratchets_one_way():
    cfg = dict(trail_activate_pts=10, trail_pts=5)
    p = Pos(+1, entry=100, sl=90, tp=200, peak=100)
    check_exit(p, 115, cfg)
    sl_high = p.sl
    check_exit(p, 112, cfg)
    assert p.sl == sl_high == 110


def test_candle_counter_allows_one_signal_per_bar():
    idx = pd.date_range("2026-01-05 09:15", periods=60, freq="5min")
    c = np.r_[np.linspace(100, 90, 30), np.linspace(90, 110, 30)]
    bars = pd.DataFrame({"high": c + 0.5, "low": c - 0.5, "close": c}, index=idx)
    s = Signaller()
    rsi = pd.Series(60.0, index=idx)
    first = [s.on_completed_bar(bars.iloc[:i], rsi.iloc[:i]) for i in range(10, 61)]
    assert sum(abs(x) for x in first) >= 1
    assert s.on_completed_bar(bars, rsi) == 0          # same last bar again: no second entry


def test_pullback_levels_use_only_past_bars():
    days = trading_days("2025-01-01", "2025-06-30")
    p = daily_panel(["A"], days)
    d = pd.DataFrame({k: v["A"] for k, v in p.items()})
    a = prep(d)
    d2 = d.copy()
    d2.iloc[-1, d2.columns.get_loc("low")] = 1.0      # change only TODAY's low
    b = prep(d2)
    assert a.low_prior.iloc[-1] == b.low_prior.iloc[-1]   # yesterday's level unaffected


def test_gap_scan_runs_and_fat_finger_cap_holds():
    days = trading_days("2024-01-01", "2024-12-31")
    out = scan(daily_panel([f"S{i}" for i in range(20)], days))
    assert {"ret", "picks"} <= set(out.columns)
    assert live_qty(price=4_000.0) == 1 and live_qty(price=6_000.0) == 0


def test_expected_move_and_beta_lots():
    assert round(expected_move(60, 60), 1) == 150.4
    assert beta_matched_lots(3_500_000, 1.0, 25_000, 50) == 2          # 2.8 lots covered: never round up


def test_annual_vs_quarterly_by_spacing_and_partial_parse_is_failure():
    q = list(pd.date_range("2024-03-31", periods=6, freq="QE"))
    a = list(pd.date_range("2019-12-31", periods=5, freq="YE"))
    assert table_kind(q) == "quarterly" and table_kind(a) == "annual"
    cache = {}
    cache_parse_result(cache, "X", {"sales": 1, "profit": None}, ["sales", "profit"])
    assert cache["X"]["status"] == "failed"
