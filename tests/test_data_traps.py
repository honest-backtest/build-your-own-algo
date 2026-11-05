import numpy as np
import pandas as pd

from algo_builder.infra.data_traps import (drift_check, lot_size_on, mark_price, members_on,
                                           suspicious_overnight_gaps, to_points)


def test_mark_price_uses_settle_off_expiry_and_intrinsic_on_overwrite():
    row = pd.Series(dict(opt_type="CE", strike=25_000, close=4.0, settle=11.5))
    assert mark_price(row, is_expiry=False, underlying_settle=np.nan) == 11.5    # stale close ignored
    over = pd.Series(dict(opt_type="CE", strike=25_000, close=25_180.0, settle=25_180.0))
    assert mark_price(over, is_expiry=True, underlying_settle=25_180.0) == 180


def test_lot_size_history_and_points():
    hist = pd.DataFrame({"effective_from": pd.to_datetime(["2019-01-01", "2021-07-01", "2024-11-20"]),
                         "lot_size": [75, 50, 75]})
    dates = pd.Series(pd.to_datetime(["2020-03-02", "2022-01-03", "2025-02-03"]))
    lots = lot_size_on(dates, hist)
    assert list(lots) == [75, 50, 75]
    pts = to_points(pd.Series([-4500.0, -3000.0, -4500.0]), lots)
    assert list(pts) == [-60.0, -60.0, -60.0]           # three different rupee losses, one rule


def test_drift_check_flags_a_corrupt_era():
    days = pd.bdate_range("2019-01-01", "2021-12-31")
    ref = pd.Series(np.linspace(11_000, 17_000, len(days)), index=days)
    bad = ref.copy()
    bad[bad.index.year == 2020] *= 1.12
    table = drift_check(bad, ref)
    assert table.loc[2019, "days_outside_tol"] == 0 and table.loc[2020, "days_outside_tol"] > 200


def test_overnight_gap_guard_and_point_in_time_members():
    idx = pd.bdate_range("2026-01-05", periods=3)
    close = pd.DataFrame({"A": [100.0, 101.0, 50.0]}, index=idx)
    open_ = pd.DataFrame({"A": [100.0, 100.5, 50.5]}, index=idx)
    open_.iloc[2, 0] = 51.0                              # a 1:1 bonus, unadjusted
    assert bool(suspicious_overnight_gaps(open_, close).iloc[2, 0])
    snaps = pd.DataFrame({"as_of": pd.to_datetime(["2018-01-01", "2018-01-01", "2024-01-01"]),
                          "symbol": ["OLD", "KEEP", "KEEP"]})
    assert members_on(pd.Timestamp("2019-06-01"), snaps) == {"OLD", "KEEP"}
    assert members_on(pd.Timestamp("2025-06-01"), snaps) == {"KEEP"}
