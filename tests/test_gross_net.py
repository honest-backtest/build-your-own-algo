import numpy as np
import pandas as pd

from algo_builder.robustness.gross_net import cost_or_signal, friction_sweep_points, round_trip_points
from algo_builder.strategies.rsi_macd_futures import Pos, check_exit, stitch_continuous


def test_round_trip_points_at_current_rates():
    pts = round_trip_points(25_000, 50, slippage_pts=1.0)
    assert 16.5 < pts < 17.5                                 # about 17 points


def test_points_do_not_depend_on_the_lot_size():
    a = round_trip_points(25_000, 50, slippage_pts=0)
    b = round_trip_points(25_000, 75, slippage_pts=0)
    assert abs(a - b) < 0.5                                  # only the flat brokerage (and its GST) differs


def test_sweep_and_diagnosis():
    gross = pd.Series([30.0, -20.0] * 50 + [6.0])            # mean just over 5 points
    sweep = friction_sweep_points(gross, cost_pts=(0, 10)).set_index("cost_pts")
    assert sweep.loc[0, "mean_net_pts"] > 0 > sweep.loc[10, "mean_net_pts"]
    assert cost_or_signal(gross, 17).startswith("cost problem")
    assert cost_or_signal(gross - 10, 17).startswith("signal problem")
    assert cost_or_signal(gross + 20, 17).startswith("survives")


def test_stitched_series_has_no_roll_jump():
    t = pd.date_range("2026-01-01", periods=6, freq="D")
    near = pd.Series([100.0, 101, 102, 103, np.nan, np.nan], t)
    far = pd.Series([np.nan, np.nan, 110, 111, 112, 113], t)
    s = stitch_continuous([near, far], [t[3]])
    assert list(s.diff().dropna()) == [1.0, 1.0, 1.0, 1.0, 1.0]   # moves kept, the 8-point gap removed


def test_trail_activates_and_exits_as_trail():
    cfg = dict(trail_activate_pts=60, trail_pts=40)
    p = Pos(+1, entry=25_000, sl=24_940, tp=25_120, peak=25_000)
    assert check_exit(p, 25_070, cfg) is None and p.trail_on and p.sl == 25_030
    assert check_exit(p, 25_025, cfg) == "TRAIL"
