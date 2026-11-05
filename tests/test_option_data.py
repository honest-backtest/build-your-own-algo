import numpy as np
import pandas as pd

from algo_builder.infra.option_bhavcopy_traps import (leg_exit_price, liquid_strikes,
                                                         settlement_from_parity)
from algo_builder.strategies.opening_range_options import bs_price


def test_expiry_overwrite_is_detected_and_settled_at_intrinsic():
    assert leg_exit_price(close=25_180.0, underlying=25_180.0, strike=25_000, is_call=True) == 180
    assert leg_exit_price(close=25_180.0, underlying=25_180.0, strike=25_000, is_call=False) == 0
    assert leg_exit_price(close=42.5, underlying=25_180.0, strike=25_200, is_call=True) == 42.5


def _chain(S, T=0.02, sigma=0.15):
    rows = []
    for k in range(24_500, 25_550, 50):
        for t, call in (("CE", True), ("PE", False)):
            rows.append(dict(strike=k, opt_type=t, close=bs_price(S, k, T, sigma, call), contracts=100))
    return pd.DataFrame(rows)


def test_parity_recovers_the_underlying():
    assert abs(settlement_from_parity(_chain(25_123.0)) - 25_123.0) < 0.5


def test_parity_needs_enough_pairs():
    assert np.isnan(settlement_from_parity(_chain(25_000.0).head(4)))


def test_liquidity_gate():
    ch = _chain(25_000.0)
    ch.loc[0, "contracts"] = 0
    assert len(liquid_strikes(ch, min_contracts=1, min_premium=0.0)) == len(ch) - 1


def test_parity_refuses_an_overwritten_chain():
    S = 72_000.0
    rows = [dict(strike=k, opt_type=t, close=S, contracts=10)
            for k in range(70_500, 72_050, 100) for t in ("CE", "PE")]
    assert np.isnan(settlement_from_parity(pd.DataFrame(rows)))      # was 71,250
