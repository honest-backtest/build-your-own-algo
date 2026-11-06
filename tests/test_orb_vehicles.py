from datetime import datetime

import numpy as np
import pandas as pd

from algo_builder.backtest.event_engine import Trade
from algo_builder.infra.cost_model import FUTURES
from algo_builder.strategies.opening_range_options import (PARAMS, option_trade, two_vehicles,
                                                           years_to_expiry)


def test_trading_time_skips_nights_and_weekends():
    fri = datetime(2026, 1, 9, 15, 30)
    mon = datetime(2026, 1, 12, 9, 15)
    assert years_to_expiry(fri, mon) == 0.0
    assert years_to_expiry(fri, mon, basis="calendar") > 2.5 / 365
    one_session = years_to_expiry(datetime(2026, 1, 12, 9, 15), datetime(2026, 1, 12, 15, 30))
    assert np.isclose(one_session, 1 / 252)


def test_option_charges_are_on_premium():
    kw = dict(direction=1, S_entry=25_000, S_exit=25_000, t_entry_yrs=2 / 252, t_exit_yrs=2 / 252,
              rv=0.15, qty=75)
    new = option_trade(**kw)
    assert 0 < new["charges_pts"] < 5                               # points per unit, on premium
    assert np.isclose(new["pts"], new["sell"] - new["buy"] - new["charges_pts"])
    assert new["buy"] - new["sell"] == PARAMS["spread_pts"]          # same price, same vol: only the spread


def test_same_trade_two_vehicles():
    ts = pd.Timestamp("2026-01-12 10:00")
    t = Trade(ts, ts + pd.Timedelta("3h"), +1, 25_000.0, 25_100.0, 1, 24_950.0, "EOD", FUTURES)
    v = two_vehicles([t], lot_size=50, expiry_for=lambda _: datetime(2026, 1, 15, 15, 30),
                     rv_for=lambda _: 0.15)
    row = v.iloc[0]
    assert row.index_pts == 100 and row.futures < 100 * 50 and row.days_to_expiry == 3
    assert row.options > 0
