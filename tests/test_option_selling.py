import pandas as pd

from algo_builder.strategies.expiry_mispricing import deviation_pct
from algo_builder.strategies.option_selling_adx_switch import P, Leg, ShortPosition, manage, stop_slip_sweep


def _pos():
    return ShortPosition([Leg("C", "CE", 100.0, 1), Leg("P", "PE", 100.0, 1),
                          Leg("WC", "CE", 5.0, -1), Leg("WP", "PE", 5.0, -1)])


def test_credit_nets_out_the_wings():
    assert _pos().credit() == 190.0


def test_stop_fills_pay_extra_and_fire_once():
    pos, p = _pos(), dict(P, leg_sl_pct=30, combined_sl_pct=20, stop_slip_pts=2.0, exit_slip_pts=0.5)
    px = {"C": 300.0, "P": 10.0, "WC": 20.0, "WP": 1.0}
    ev = manage(pos, px, adx15=0, spot_dir=0, p=p)
    assert ev[0] == "leg stop CE" and pos.legs[0].exit == 302.5
    assert "combined stop" in ev and pos.legs[2].exit == 20.0 - 2.5           # a wing sells lower
    assert manage(pos, px, adx15=0, spot_dir=0, p=p) == []                   # nothing left to fire


def test_switch_closes_only_the_threatened_sold_leg():
    pos = _pos()
    ev = manage(pos, {"C": 110.0, "P": 90.0, "WC": 6.0, "WP": 4.0}, adx15=40, spot_dir=+1)
    assert ev == ["trend: closed CE"] and not pos.legs[0].open and pos.legs[2].open


def test_slip_sweep_reports_each_level():
    def run(p):
        return pd.DataFrame({"pnl_pts": [5.0 - p["stop_slip_pts"]] * 3, "stops": [1, 1, 1]})
    s = stop_slip_sweep(run, slips=(0, 5))
    assert list(s.mean_pnl_pts) == [5.0, 0.0]


def test_no_signal_when_fair_value_is_tiny():
    assert deviation_pct(mid=1.0, vwap_spot=24_000, K=25_000, T=1 / 252 / 375, iv=0.12, is_call=True) == 0.0
