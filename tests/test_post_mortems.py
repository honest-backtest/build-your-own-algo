"""Chapter 25: post-mortem checks."""
import numpy as np
import pandas as pd

from algo_builder.infra.cost_model import EQ_DELIVERY, EQ_INTRADAY, round_trip_cost
from algo_builder.lab.post_mortems import (both_sides, by_horizon, event_study, iv_percentile,
                                           pair_round_trip_pct, turnover_drag)


def test_costs_scale_with_turnover_and_legs():
    assert turnover_drag(0, EQ_INTRADAY, 0.0003) == 0
    assert turnover_drag(1.0, EQ_INTRADAY, 0.0003) > 3.9 * turnover_drag(0.25, EQ_INTRADAY, 0.0003)
    # delivery STT both sides: at least 4 x 0.1% across the four orders
    assert pair_round_trip_pct(500_000, EQ_DELIVERY) >= 0.004
    assert pair_round_trip_pct(500_000, EQ_DELIVERY) == 2 * round_trip_cost(5e5, 5e5, EQ_DELIVERY) / 5e5


def test_event_study_separates_drift_and_timing():
    days = pd.bdate_range("2020-01-01", periods=300)
    mkt = pd.Series(np.linspace(100, 130, 300), index=days)      # steady drift
    closes = pd.DataFrame({"A": mkt.to_numpy()}, index=days)     # share = market
    ev = event_study(closes, pd.DataFrame({"date": [days[10], days[295]], "symbol": ["A", "A"]}),
                     horizon=20, market=mkt, n_draws=50)
    assert len(ev) == 1                                          # the late event has no future to measure
    assert abs(ev.vs_market.iloc[0]) < 1e-12
    assert ev.ret.iloc[0] > 0 and ev.vs_random.iloc[0] > 0       # early in a linear path, returns are larger


def test_horizon_sides_and_iv_percentile():
    h = by_horizon(pd.Series([100.0, 100.0]), pd.DataFrame({"15min": [99.0, 101.0], "close": [98.0, 97.0]}))
    assert h.loc["15min", "win_rate"] == 0.5 and h.loc["close", "win_rate"] == 0.0
    b = both_sides(pd.Series([0.0, 0.0]), cost_rt=0.01)
    assert b["long_mean"] == b["short_mean"] == -0.01            # a zero trade loses on both sides
    iv = pd.Series([10.0, 20, 30, 15, 40])
    p = iv_percentile(iv, window=3)
    assert p.isna().iloc[:3].all() and p.iloc[3] == 1 / 3 and p.iloc[4] == 1.0


def test_skip_rule_effect_counts_worst_periods():
    from algo_builder.lab.post_mortems import skip_rule_effect
    pnl = pd.Series([1.0, 1.0, -5.0, 1.0, -0.5])
    e = skip_rule_effect(pnl, pd.Series([True, False, False, True, True]), worst=1)
    assert e["skipped"] == 3 and e["income_given_up"] == 2.0 and e["losses_avoided"] == 0.5
    assert e["worst_skipped"] == 0 and e["total_after"] == -4.0
