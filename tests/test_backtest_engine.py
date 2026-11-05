from datetime import time

import pandas as pd

from algo_builder.backtest.event_engine import Position, run_day
from datetime import date

import pytest

from algo_builder.backtest.sizing import SizingError, check_sane, lots_for_risk
from algo_builder.infra.cost_model import (EQ_INTRADAY, FUTURES, OPTIONS, leg_cost, rates_on,
                                         round_trip_cost)


def _bars(rows):
    idx = pd.date_range("2026-01-05 09:15", periods=len(rows), freq="15min")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def test_no_entry_before_range_closes_and_stop_checked_first():
    bars = _bars([(100, 101, 99, 100), (100, 101, 99, 100), (100, 100, 100, 100), (100, 106, 94, 100)])
    seen = []

    def signal(hist, ts):
        seen.append(ts.time())
        return Position(+1, 100.0, 10, initial_sl=95.0, sl=95.0, target=105.0, entry_ts=ts)

    trades = run_day(bars, signal, range_end=time(9, 45), square_off=time(15, 15), slip_pct=0)
    assert min(seen) >= time(9, 45)                 # range guard
    assert trades[0].reason == "SL"                  # stop and target in one bar: worse case


def test_r_uses_the_initial_stop_not_the_trailed_one():
    bars = _bars([(100, 100, 100, 100), (100, 100, 100, 100), (100, 110, 101, 110)])

    def signal(hist, ts):
        return Position(+1, 100.0, 10, initial_sl=95.0, sl=99.0, target=110.0, entry_ts=ts)

    t = run_day(bars, signal, range_end=time(9, 30), square_off=time(15, 15), slip_pct=0)[0]
    assert t.reason == "TARGET"
    assert 1.8 < t.r < 2.0                           # ~2R less costs, not infinity


def test_stt_base_differs_by_instrument():
    v = 1_000_000
    assert leg_cost(v, "SELL", FUTURES) > leg_cost(v, "BUY", FUTURES)       # STT on sell side
    assert leg_cost(7_500, "SELL", OPTIONS) < leg_cost(v, "SELL", FUTURES)   # premium base
    assert leg_cost(v, "BUY", EQ_INTRADAY) > 0


def _always_long(stop_pts=5.0, target_pts=20.0):
    def signal(hist, ts):
        px = float(hist.close.iloc[-1])
        return Position(+1, px, 10, initial_sl=px - stop_pts, sl=px - stop_pts,
                        target=px + target_pts, entry_ts=ts)
    return signal


def test_fill_clock_delays_the_entry_to_a_later_open():
    bars = _bars([(100, 100, 100, 100), (100, 101, 99, 101), (103, 104, 102, 103), (103, 103, 103, 103)])
    late = run_day(bars, _always_long(), time(9, 30), time(15, 15), slip_pct=0, fill_delay_bars=1)[0]
    lie = run_day(bars, _always_long(), time(9, 30), time(15, 15), slip_pct=0, fill_delay_bars=0)[0]
    assert lie.entry == 101 and late.entry == 103          # decision close vs next open


def test_a_gap_through_the_stop_fills_at_the_open():
    bars = _bars([(100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100), (90, 91, 89, 90)])
    t = run_day(bars, _always_long(stop_pts=5), time(9, 30), time(15, 15), slip_pct=0)[0]
    assert t.reason == "SL" and t.exit == 90               # not 95: the market opened at 90


def test_a_position_open_when_the_data_ends_is_booked():
    bars = _bars([(100, 100, 100, 100), (100, 100, 100, 100), (100, 101, 99, 100)])
    t = run_day(bars, _always_long(), time(9, 30), time(15, 15), slip_pct=0)
    assert len(t) == 1 and t[0].reason == "EOD"


def test_trail_marks_the_exit_as_a_trail():
    bars = _bars([(100, 100, 100, 100), (100, 100, 100, 100), (100, 108, 100, 108),
                  (108, 108, 102, 103)])

    def trail(pos, bar):
        pos.sl = max(pos.sl, bar.close - 4)

    t = run_day(bars, _always_long(), time(9, 30), time(15, 15), slip_pct=0, trail=trail)[0]
    assert t.reason == "TRAIL" and t.exit == 104


def test_cost_profiles_reproduce_the_2026_round_trips():
    fut = round_trip_cost(1_250_000, 1_250_000, FUTURES)     # one lot of 50 at 25,000
    opt = round_trip_cost(7_500, 7_500, OPTIONS)              # 50 x 150 premium
    assert 740 < fut < 770 and 60 < opt < 70


def test_stt_history_is_applied_by_date():
    assert rates_on("FUTURES", date(2024, 6, 3)).stt_sell_pct == 0.000125
    assert rates_on("FUTURES", date(2025, 6, 2)).stt_sell_pct == 0.0002
    assert rates_on("FUTURES", date(2026, 6, 1)).stt_sell_pct == 0.0005
    assert rates_on("EQ_DELIVERY", date(2020, 1, 1)).stt_sell_pct == 0.001


def test_sizing_is_capped_and_impossible_states_fail_loudly():
    assert lots_for_risk(500_000, 0.01, 25_000, 24_900, 50, max_notional_mult=3) == 1
    assert lots_for_risk(500_000, 0.50, 25_000, 24_990, 50, max_notional_mult=3) == 1   # cap wins
    idx = pd.bdate_range("2026-01-05", periods=3)
    with pytest.raises(SizingError):
        check_sane(pd.Series([1e5, 1e5, 1e5], idx), pd.Series([1e5, 5e5, 1e5], idx), 3)
    with pytest.raises(SizingError):
        check_sane(pd.Series([1e5, -1.0, 1e5], idx), pd.Series([0.0, 0.0, 0.0], idx), 3)
