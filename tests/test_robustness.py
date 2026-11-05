import numpy as np
import pandas as pd

from algo_builder.robustness.by_year_and_ex_top_n import by_year, ex_top_n
from algo_builder.robustness.drift_benchmark import drift_report, random_entry_means
from algo_builder.robustness.fill_delay_test import fill_delay_table
from algo_builder.robustness.friction_sweep import break_even_bps, friction_sweep
from algo_builder.robustness.walk_forward import walk_forward


def test_friction_sweep_crosses_zero_at_break_even():
    t = pd.DataFrame({"gross_ret": np.full(100, 0.002)})       # 20 bps gross per trade
    assert abs(break_even_bps(t) - 10) < 1e-9
    sweep = friction_sweep(t, bps_per_side=(0, 10, 20)).set_index("bps_per_side")
    assert sweep.loc[0, "mean_net_pct"] > 0 and abs(sweep.loc[10, "mean_net_pct"]) < 1e-9
    assert sweep.loc[20, "mean_net_pct"] < 0


def test_ex_top_n_flips_a_lottery_ticket():
    t = pd.DataFrame({"entry_ts": pd.date_range("2024-01-01", periods=101, freq="W"),
                      "pnl": [1000] + [-5] * 100})
    table = ex_top_n(t, ns=(0, 1)).set_index("removed_top")
    assert table.loc[0, "total"] > 0 > table.loc[1, "total"]
    assert set(by_year(t).index) == {2024, 2025}


def test_drift_benchmark_sees_through_a_rising_market():
    days = pd.bdate_range("2020-01-01", periods=600)
    closes = pd.DataFrame({"A": 100 * np.exp(np.linspace(0, 0.6, 600))}, index=days)
    trades = pd.DataFrame({"symbol": ["A"] * 50, "entry_idx": range(100, 150), "ret": 0.0})
    trades["ret"] = [closes.A.iloc[i + 11] / closes.A.iloc[i + 1] - 1 for i in trades.entry_idx]
    rnd = random_entry_means(closes, trades, hold=10, cost_rt=0.0, n_draws=200)
    rep = drift_report(trades, rnd)
    assert abs(rep["excess_pct"]) < 0.05 and rep["p_value"] > 0.05      # no edge beyond drift


def test_fill_delay_kills_an_open_print_edge():
    bars = {}
    for i, d in enumerate(pd.bdate_range("2026-01-05", periods=20)):
        idx = pd.date_range(d + pd.Timedelta("9h15min"), periods=30, freq="1min")
        px = np.full(30, 101.0) + np.random.default_rng(i).normal(0, 0.001, 30)
        px[0] = 100.0                                              # the print is the only cheap price
        bars[("S", d.date())] = pd.DataFrame({"open": px}, index=idx)
    sig = pd.DataFrame({"symbol": "S", "date": [k[1] for k in bars], "exit_px": 101.0})
    tab = fill_delay_table(bars, sig, delays=(0, 1)).set_index("delay_min")
    assert tab.loc[0, "mean_pct"] > 0.9 and abs(tab.loc[1, "mean_pct"]) < 0.01


def test_walk_forward_reports_out_of_sample_only():
    def run(c):
        yrs = np.repeat(np.arange(2019, 2025), 40)
        pnl = np.where(yrs < 2022, c["good_early"], -c["good_early"]).astype(float)
        return pd.DataFrame({"entry_ts": pd.to_datetime([f"{y}-06-01" for y in yrs]), "pnl": pnl})

    wf = walk_forward([{"good_early": 1}, {"good_early": -1}], run, 2022, 2024, min_trades=30)
    assert (wf.oos_pnl < 0).all()           # in-sample winner fails out of sample, and we see it
