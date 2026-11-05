from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from algo_builder.strategies.option_chart_momentum import (AtmTracker, atm_contract, entry_ok, exit_reason,
                                                           nearest_expiry, premium_features)
from algo_builder.strategies.supertrend_rsi import Signaller


def _master():
    rows = []
    for e, lot in ((date(2026, 1, 13), 75), (date(2026, 1, 20), 65)):
        for k in (24_950.0, 25_000.0, 25_050.0, 25_100.0):
            rows.append(dict(underlying="NIFTY", kind="CE", expiry=e, strike=k,
                             symbol=f"N{e:%d}{int(k)}CE", lot_size=lot))
    return pd.DataFrame(rows)


def test_expiry_and_contract_come_from_the_master():
    m = _master()
    assert nearest_expiry(m, "NIFTY", date(2026, 1, 14)) == date(2026, 1, 20)
    c = atm_contract(m, "NIFTY", date(2026, 1, 20), spot=25_062.0, kind="CE")
    assert c.strike == 25_050.0 and c.lot_size == 65 and c.symbol == "N2025050CE"
    with pytest.raises(LookupError):
        nearest_expiry(m, "NIFTY", date(2026, 2, 1))


def test_tracker_does_not_flap_near_the_midpoint():
    strikes = np.array([24_950.0, 25_000.0, 25_050.0])
    t = AtmTracker(hysteresis=10)
    seen = [t.update(s, strikes) for s in (25_020, 25_028, 25_022, 25_029, 25_040)]
    assert seen == [25_000.0] * 4 + [25_050.0]


def test_no_entry_before_warm_up_and_time_exit():
    idx = pd.date_range("2026-01-12 09:15", periods=40, freq="5min")
    c = pd.DataFrame({"open": 100.0, "high": np.linspace(101, 140, 40), "low": 99.0,
                      "close": np.linspace(100, 139, 40), "volume": 1_000.0}, index=idx)
    f = premium_features(c)
    assert not entry_ok(f.iloc[5], underlying_adx=30, vix=15)
    assert entry_ok(f.iloc[-1], underlying_adx=30, vix=15)
    assert exit_reason(f.iloc[-1], 100.0, pd.Timestamp("2026-01-12 15:10"), time(15, 0)) == "time exit"


def test_signaller_refuses_misaligned_rsi():
    idx = pd.date_range("2026-01-05 09:15", periods=10, freq="5min")
    bars = pd.DataFrame({"high": 101.0, "low": 99.0, "close": 100.0}, index=idx)
    with pytest.raises(ValueError):
        Signaller().on_completed_bar(bars, pd.Series(60.0, index=idx[:-1]))
