"""Chapter 24: design-exercise tools."""
import numpy as np
import pandas as pd
import pytest

from algo_builder.lab.design_exercises import ablate, align_completed, by_recipe, cpr, touched, window_pf


def test_cpr_and_two_meanings_of_touched():
    assert cpr(25_100, 24_900, 25_050) == pytest.approx((25_000, 25_033.33), abs=0.01)
    bars = pd.DataFrame({"high": [101, 101, 102], "low": [99.5, 99.0, 100.5],
                         "close": [100.5, 100.8, 101.5]})
    # level 100: traded through on bars 0 and 1, but never closed below it
    assert touched(bars, 100, "trade").tolist() == [True, True, False]
    assert not touched(bars, 100, "close").any()
    assert touched(pd.DataFrame({"close": [99.0, 101.0]}), 100, "close").tolist() == [False, True]


def test_higher_bar_only_after_it_finishes():
    t15 = pd.date_range("2026-01-05 09:15", periods=8, freq="15min")
    lo = pd.DataFrame({"close": range(8)}, index=t15).rename_axis("ts")
    hi = pd.DataFrame({"close": [100.0, 200.0]},
                      index=pd.DatetimeIndex(["2026-01-05 09:15", "2026-01-05 10:15"])).rename_axis("ts")
    out = align_completed(lo, "15min", hi, "60min")
    # 15-min bars ending 09:30..10:00 have no finished hour; the bar ending 10:15 sees 09:15's hour
    assert out.close_hi.iloc[:3].isna().all()
    assert out.close_hi.iloc[3:7].tolist() == [100.0] * 4
    assert out.close_hi.iloc[7] == 200.0          # 10:15 hour finished at 11:15


def test_ablation_and_recipes():
    rng = np.random.default_rng(1)
    n = 20_000
    good = rng.normal(size=n)
    outcome = pd.Series(0.2 * good + rng.normal(size=n))
    gates = pd.DataFrame({"useful": good > 0, "useless": rng.random(n) > 0.5})
    a = ablate(gates, outcome)
    assert a.loc["useful", "removed_mean"] < a.loc["useful", "kept_mean"] - 0.1
    assert abs(a.loc["useless", "removed_mean"] - a.loc["useless", "kept_mean"]) < 0.05

    fired = pd.DataFrame({"a": [True, True, False], "b": [False, True, True]})
    r = by_recipe(fired, pd.Series([1.0, 2.0, -1.0]))
    assert r.loc["a", "alone"] == 1 and r.loc["a", "alone_mean"] == 1.0
    assert r.loc["b", "shared"] == 0.5


def test_window_pf_counts_trades():
    idx = pd.to_datetime(["2025-01-10", "2025-02-10", "2025-06-10", "2025-07-10"])
    w = window_pf(pd.Series([1.0, -1.0, 2.0, -0.5], index=idx), 5)
    assert w.trades.tolist() == [2, 2] and w.pf.tolist() == [1.0, 4.0]
