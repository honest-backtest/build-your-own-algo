import pandas as pd

from chapters.ch01.run import ten_trades
from algo_builder.backtest.metrics import (max_drawdown, monte_carlo, realised_equity,
                                              trade_stats)
from algo_builder.common.bars import completed_bars
from algo_builder.common.contract import Contract


def test_report_card_matches_the_book():
    s = trade_stats(ten_trades())
    assert s["n"] == 10 and s["win_rate"] == 0.3
    assert round(s["profit_factor"], 2) == 1.25
    assert s["mean_pnl"] == 150 and round(s["expectancy_r"], 2) == 0.15
    assert round(s["t_stat"], 2) == 0.28


def test_equity_starts_from_capital_so_day_one_loss_counts():
    t = pd.DataFrame({"exit_ts": pd.to_datetime(["2026-01-05 15:00", "2026-01-06 15:00"]),
                      "pnl": [-5000, 1000]})
    eq = realised_equity(t, 100_000)
    assert eq.iloc[0] == 100_000
    assert round(max_drawdown(eq), 4) == -0.05


def test_equity_is_booked_on_exit_day():
    t = pd.DataFrame({"entry_ts": pd.to_datetime(["2026-01-05 10:00"]),
                      "exit_ts": pd.to_datetime(["2026-01-07 15:00"]), "pnl": [2000]})
    eq = realised_equity(t, 100_000)
    assert eq.index[-1] == pd.Timestamp("2026-01-07")


def test_ten_trades_drawdown_is_two_percent():
    assert round(max_drawdown(realised_equity(ten_trades(), 100_000)), 3) == -0.02


def test_monte_carlo_returns_percentiles():
    mc = monte_carlo(ten_trades(), 100_000, n_sim=200)
    assert mc["worst_5pct_max_dd"] <= mc["median_max_dd"] <= 0
    assert mc["worst_5pct_streak"] >= mc["median_streak"] >= 1


def test_monte_carlo_counts_a_loss_on_the_first_trade():
    t = pd.DataFrame({"pnl": [-1000.0] * 5})
    mc = monte_carlo(t, 100_000, n_sim=20)
    assert abs(mc["median_max_dd"] + 0.05) < 1e-9 and mc["median_streak"] == 5


def test_monte_carlo_blocks_keep_clusters_together():
    t = pd.DataFrame({"pnl": [-100.0] * 10 + [60.0] * 90})
    one = monte_carlo(t, 10_000, n_sim=400, block=1, seed=3)
    blocks = monte_carlo(t, 10_000, n_sim=400, block=10, seed=3)
    assert blocks["worst_5pct_streak"] > one["worst_5pct_streak"]


def test_completed_bars_boundary():
    idx = pd.date_range("2026-01-05 09:15", periods=3, freq="15min")
    bars = pd.DataFrame({"close": [1, 2, 3]}, index=idx)
    assert len(completed_bars(bars, pd.Timestamp("2026-01-05 09:45:00"), "15min")) == 2
    assert len(completed_bars(bars, pd.Timestamp("2026-01-05 09:44:59"), "15min")) == 1


def test_contract_arithmetic():
    c = Contract("X", "NIFTY", "PE", pd.Timestamp("2026-01-27").date(), 50, 25_000)
    assert c.rupees(-60) == -3000
    assert c.intrinsic(24_900) == 100 and c.intrinsic(25_100) == 0
