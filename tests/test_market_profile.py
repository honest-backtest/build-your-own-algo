from datetime import datetime, time

import numpy as np
import pandas as pd

from algo_builder.strategies.market_profile_framework import (DayTypeMachine, InitialBalance,
                                                                 RollingPercentile, extension, on_bar,
                                                                 trail_from_completed)
from algo_builder.strategies.mcx_session_vwap import SessionVWAP, VWAPTouchBuffer, trend_pullback_long


def _bars(prices, start="2026-01-05 09:15"):
    idx = pd.date_range(start, periods=len(prices), freq="1min")
    p = np.asarray(prices, float)
    return pd.DataFrame({"open": p, "high": p + 1, "low": p - 1, "close": p}, index=idx)


def test_extension_reports_facts():
    ib = InitialBalance(time(10, 15), high=110.0, low=100.0, closed=True)
    assert [extension(ib, h, lo) for h, lo in [(105, 101), (111, 101), (105, 99), (111, 99)]] == \
           ["none", "up", "down", "both"]


def test_today_is_ranked_against_earlier_days_only():
    rp = RollingPercentile(n=10, min_history=3)
    for w in (1.0, 2.0, 3.0):
        rp.push(w)
    assert rp.rank_then_push(10.0) == 1.0 and rp.rank(10.0) == 0.75


def test_ruled_out_transitions_are_ignored():
    rules = [("TWO_SIDED", lambda c: c["x"] == 2), ("TREND", lambda c: c["x"] == 1),
             ("INSIDE", lambda c: c["x"] == 0)]
    m = DayTypeMachine(rules, allowed={"UNDETERMINED": {"INSIDE", "TREND", "TWO_SIDED"},
                                       "INSIDE": {"TREND", "TWO_SIDED"}, "TREND": {"TWO_SIDED"},
                                       "TWO_SIDED": set()})
    labels = [m.update(pd.Timestamp("2026-01-05 10:15") + pd.Timedelta(minutes=i), {"x": x})
              for i, x in enumerate([0, 1, 0, 2, 1])]
    assert labels == ["INSIDE", "TREND", "TREND", "TWO_SIDED", "TWO_SIDED"]


def test_no_setup_before_the_first_hour_and_one_trade_per_side():
    bars = _bars([100.0] * 60 + [105.0] * 10)
    ib, log, taken, ctx = InitialBalance(time(10, 15)), [], set(), {}
    m = DayTypeMachine([("ANY", lambda c: True)])
    for ts, b in bars.iterrows():
        on_bar(ts, b, ib, m, {"ANY": lambda c: dict(side="long", ts=c["ts"])}, ctx, log, taken)
    assert len(log) == 1 and log[0]["ts"].time() == time(10, 15)
    assert log[0]["day_type_at_entry"] == "ANY"


def test_trail_uses_only_completed_higher_bars():
    prices = list(np.linspace(100, 130, 30)) + list(np.linspace(130, 160, 30)) + [161.0] * 5
    bars = _bars(prices)
    now = datetime(2026, 1, 5, 10, 20)                     # 09:15 and 09:45 bars complete; 10:15 forming
    done = bars[bars.index + pd.Timedelta("1min") <= now]
    sl = trail_from_completed(+1, 95.0, 100.0, best_move=60, breakeven_after=20, bars_1m=done,
                              now=now, rule="30min", buffer=1.0)
    assert np.isclose(sl, 130.0 - 1 - 1)                  # low of the 09:45 bar (129) minus the buffer
    early = trail_from_completed(+1, 95.0, 100.0, best_move=10, breakeven_after=20, bars_1m=done,
                                 now=now, rule="30min", buffer=1.0)
    assert early == 95.0


def test_vwap_buffer_catches_a_one_bar_touch():
    buf = VWAPTouchBuffer(lookback=3)
    buf.push(101, 99, 100)                                  # the touch
    buf.push(120, 110, 101)                                 # runs away
    assert trend_pullback_long(118, 101, buf, one_sided_up=True)
    vw = SessionVWAP()
    vw.update(100, 1)
    vw.reset()
    assert vw.update(50, 1) == 50
