"""
Chapter 13 — One signal, two vehicles: the opening-range breakout.

A SYNTHETIC index on 5-minute bars, 2024-2025, in which some days trend after the
first half-hour. One breakout signal (30-minute range, a trend filter, the range
band against yesterday's ATR, one shot per side), run through the Chapter 5 engine
on the INDEX, then expressed two ways: one lot of futures, and two lots of the ATM
weekly option, PRICED BY THE MODEL (synthetic data has no option prices).

  1. Futures vs options, and the futures friction sweep.
  2. Stressing the model: volatility multiplier, spread, and a fall in volatility
     while the trade is open.
  3. Results by days to expiry.

Every setting is ILLUSTRATIVE. Weekly expiry here is a synthetic Thursday.

    python chapters/ch13/run.py
"""
from __future__ import annotations

from datetime import datetime, time as dtime, timedelta

import numpy as np
import pandas as pd

from algo_builder.backtest.event_engine import Position, run_day
from algo_builder.infra.cost_model import FUTURES
from algo_builder.robustness.gross_net import friction_sweep_points
from algo_builder.strategies.opening_range_options import (PARAMS, OneShotPerSide, range_ok,
                                                           realised_vol, two_vehicles, wilder_atr)

LOT = 50                                                   # ILLUSTRATIVE lot size
OR_END, SQUARE_OFF = dtime(9, 45), dtime(15, 0)


def market(seed: int = 13) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-01-01", "2025-12-31")
    level, sessions, daily = 22_000.0, {}, []
    for d in days:
        n = 75                                             # 09:15 ... 15:25, 5-minute bars
        steps = rng.normal(0, 0.0011, n)
        if rng.random() < 0.3:                             # a trend day after the first half-hour
            steps[6:] += rng.choice([1, -1]) * rng.uniform(0.0001, 0.0002)
        close = level * np.exp(np.cumsum(steps))
        open_ = np.concatenate([[level], close[:-1]])
        wig = np.abs(rng.normal(0, 0.0005, n)) * close
        idx = pd.date_range(d + pd.Timedelta("9h15min"), periods=n, freq="5min")
        b = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + wig,
                          "low": np.minimum(open_, close) - wig, "close": close}, index=idx)
        sessions[d.date()] = b
        daily.append(dict(date=d, high=b.high.max(), low=b.low.min(), close=close[-1]))
        level = close[-1]
    return sessions, pd.DataFrame(daily).set_index("date")


def make_signal(atr_today: float):
    shots = OneShotPerSide()

    def signal(h: pd.DataFrame, ts: pd.Timestamp):
        rng = h.between_time("09:15", "09:44")
        if not range_ok(rng.high.max(), rng.low.min(), atr_today):
            return None
        c, trend = h.close.iloc[-1], np.sign(h.close.iloc[-1] - h.close.rolling(12).mean().iloc[-1])
        for d, broke in ((+1, c > rng.high.max()), (-1, c < rng.low.min())):
            if broke and trend == d and shots.allow(d):
                stop = c - d * 0.4 * atr_today                       # ILLUSTRATIVE
                return Position(d, float(c), 1, stop, stop, c + d * 1.5 * atr_today, ts)
        return None
    return signal


def expiry_for(ts: pd.Timestamp) -> datetime:
    d = ts.date() + timedelta(days=(3 - ts.weekday()) % 7)        # this week's Thursday
    return datetime.combine(d, dtime(15, 30))


def main() -> None:
    pd.set_option("display.width", 140)
    sessions, daily = market()
    atr_prev = wilder_atr(daily).shift(1)                          # known before the day
    trades = []
    for d, bars in sessions.items():
        a = atr_prev.get(pd.Timestamp(d))
        if a is None or not np.isfinite(a):
            continue
        trades += run_day(bars, make_signal(float(a)), OR_END, SQUARE_OFF, slip_pct=0.0,
                          costs=FUTURES, max_trades=2)
    closes = daily.close
    rv_for = lambda day: realised_vol(closes[closes.index < pd.Timestamp(day)])
    v = two_vehicles(trades, LOT, expiry_for, rv_for)

    print(f"1. {len(v)} trades, 2024-2025 (synthetic index; options priced by the model, "
          f"vol x{PARAMS['vol_mult']}, trading-time basis)")
    print(f"   index points a trade, before costs: {v.index_pts.mean():+.1f}")
    print(pd.DataFrame({"total_rupees": v[["futures", "options"]].sum(),
                        "mean_rupees": v[["futures", "options"]].mean()}).round(0).to_string())
    print("   futures, friction sweep (index points a round trip):")
    print(friction_sweep_points(v.index_pts, cost_pts=(0, 5, 10, 15, 20)).round(2).to_string(index=False))

    print("\n2. Stressing the model: total option result, rupees (rows: vol multiplier; columns: spread)")
    for basis in ("trading", "calendar"):
        rows = []
        for vm in (0.9, 1.0, 1.1, 1.2, 1.4):
            for sp in (0.5, 1.0, 2.0, 4.0):
                p = dict(PARAMS, vol_mult=vm, spread_pts=sp)
                tot = two_vehicles(trades, LOT, expiry_for, rv_for, p, basis=basis).options.sum()
                rows.append(dict(vol_mult=vm, spread=sp, total=tot))
        print(f"   time basis: {basis}")
        print(pd.DataFrame(rows).pivot(index="vol_mult", columns="spread", values="total").round(-3).to_string())
    for drop in (0.9, 0.8):
        x = two_vehicles(trades, LOT, expiry_for, rv_for, dict(PARAMS, vol_mult=1.0), exit_vol_mult=drop).options.sum()
        print(f"   vol x1.0 at entry, x{drop} at exit: {x:,.0f}")

    print("\n3. By days to expiry (rupees; options at vol x1.0)")
    v1 = two_vehicles(trades, LOT, expiry_for, rv_for, dict(PARAMS, vol_mult=1.0))
    g = v1.groupby("days_to_expiry")[["futures", "options"]].agg(["size", "sum"])
    print(g.round(0).to_string())


if __name__ == "__main__":
    main()
