import numpy as np
import pandas as pd

from algo_builder.robustness.by_year_and_ex_top_n import concentration_by_day, ex_top_days
from algo_builder.robustness.regime import by_regime, daily_regimes, tag_at_entry
from algo_builder.strategies.rsi_macd_dipbuy import regime_tag
from chapters.ch08.run import crash_day_strategy, lottery_ticket, market


def test_lottery_ticket_flips_without_its_best_trades():
    from algo_builder.robustness.by_year_and_ex_top_n import ex_top_n
    tab = ex_top_n(lottery_ticket(), ns=(0, 10)).set_index("removed_top")
    assert tab.loc[0, "total"] > 0 > tab.loc[10, "total"]


def test_crash_days_carry_the_result():
    c = crash_day_strategy()
    tab = ex_top_days(c, ns=(0, 6)).set_index("removed_top_days")
    assert tab.loc[0, "total"] > 0 > tab.loc[6, "total"]
    assert concentration_by_day(c.assign(pnl=-c.pnl.abs()))["share_of_pnl"] is np.nan or \
        np.isnan(concentration_by_day(c.assign(pnl=-c.pnl.abs()))["share_of_pnl"])


def test_regime_tag_says_unknown_without_enough_history():
    short = pd.Series(np.linspace(100, 120, 50))
    assert regime_tag(short, vix=12)["index_trend"] == "unknown"
    long_ = pd.Series(np.linspace(100, 200, 300))
    assert regime_tag(long_, vix=25) == {"index_trend": "up", "vix_bucket": "high"}


def test_tags_use_only_the_day_before_entry():
    close, vix = market()
    regimes = daily_regimes(close, vix)
    trades = pd.DataFrame({"entry_ts": close.index[[300, 301]] + pd.Timedelta("9h20min"),
                           "pnl": [1.0, -1.0]})
    tagged = tag_at_entry(trades, regimes)
    assert list(tagged.vix_bucket) == list(regimes.vix_bucket.iloc[[299, 300]])


def test_look_ahead_tags_invent_a_filter():
    close, vix = market()
    trades = pd.DataFrame({"entry_ts": close.index[1:], "pnl": np.diff(np.log(close.to_numpy())) * 100})
    known = by_regime(tag_at_entry(trades, daily_regimes(close, vix)), "vix_bucket")
    leaked = by_regime(trades.assign(vix_bucket=daily_regimes(close, vix).vix_bucket.iloc[1:].to_numpy()),
                       "vix_bucket")
    assert leaked.loc["low", "profit_factor"] > 3 and leaked.loc["high", "profit_factor"] < 0.5
    assert known.profit_factor.max() / known.profit_factor.min() < 1.5
