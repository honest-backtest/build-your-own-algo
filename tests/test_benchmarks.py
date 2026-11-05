import numpy as np
import pandas as pd

from algo_builder.robustness.benchmarks import hold_the_list, same_day_means
from algo_builder.robustness.drift_benchmark import drift_report, random_entry_means
from chapters.ch07.run import market, trades_at


def test_crash_buying_is_timing_not_selection():
    closes, ends = market()
    rng = np.random.default_rng(9)
    picks = [(e, s) for e in ends for s in rng.choice(closes.columns, 15, replace=False)]
    t = trades_at(closes, picks, 40)
    vs_days = drift_report(t, random_entry_means(closes, t, n_draws=300))
    vs_stocks = drift_report(t, same_day_means(closes, t, n_draws=300))
    assert vs_days["excess_pct"] > 5 and vs_days["p_value"] < 0.05
    assert abs(vs_stocks["excess_pct"]) < 3 and vs_stocks["p_value"] > 0.1


def test_each_trade_keeps_its_own_holding_period():
    days = pd.bdate_range("2020-01-01", periods=400)
    closes = pd.DataFrame({"A": np.exp(np.linspace(0, 1, 400))}, index=days)   # smooth rise
    short = pd.DataFrame({"symbol": ["A"] * 20, "hold": 5, "ret": 0.0})
    long_ = pd.DataFrame({"symbol": ["A"] * 20, "hold": 100, "ret": 0.0})
    assert random_entry_means(closes, long_, n_draws=50).mean() > \
        10 * random_entry_means(closes, short, n_draws=50).mean()


def test_p_value_is_never_zero():
    r = drift_report(pd.DataFrame({"ret": [1.0]}), np.zeros(999))
    assert r["p_value"] == 1 / 1000


def test_hold_the_list():
    days = pd.bdate_range("2020-01-01", periods=253)
    closes = pd.DataFrame({"A": np.linspace(100, 200, 253), "B": np.linspace(100, 100, 253)}, index=days)
    h = hold_the_list(closes, ["A", "B"], days[0], days[-1])
    assert h["names"] == 2 and abs(h["total_return_pct"] - 50) < 1e-9
