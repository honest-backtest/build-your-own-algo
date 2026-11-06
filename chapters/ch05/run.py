"""
Chapter 5 — A backtester that doesn't lie.

Four small demonstrations on constructed and synthetic data:
  1. the fill clock: the same signal filled at the decision bar's close and at
     the next bar's open;
  2. a gap through the stop, filled at the open;
  3. the cost model: 2026 round trips, and one futures trade costed by date;
  4. a sizing bug that runs away, and the cap that stops it.

    python chapters/ch05/run.py
"""
from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

from algo_builder.backtest.event_engine import Position, run_day
from algo_builder.backtest.sizing import SizingError, check_sane, lots_for_risk
from algo_builder.infra.cost_model import FUTURES, OPTIONS, round_trip_cost


def bars(rows):
    idx = pd.date_range("2026-01-05 09:15", periods=len(rows), freq="15min")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def breakout_long(hist, ts):
    px = float(hist.close.iloc[-1])
    return Position(+1, px, 50, initial_sl=px - 40, sl=px - 40, target=px + 80, entry_ts=ts)


def fill_clock() -> None:
    day = bars([(25000, 25010, 24990, 25000), (25000, 25060, 24995, 25055),
                (25075, 25140, 25070, 25130), (25130, 25150, 25120, 25140)])
    for delay in (0, 1):
        t = run_day(day, breakout_long, time(9, 30), time(15, 15), slip_pct=0, fill_delay_bars=delay)[0]
        print(f"   fill_delay_bars={delay}: entry {t.entry:,.0f}, exit {t.exit:,.0f} ({t.reason}), "
              f"{t.direction * (t.exit - t.entry):+.0f} points")


def gap_through_stop() -> None:
    day = bars([(25000, 25010, 24990, 25000), (25000, 25005, 24995, 25000),
                (25000, 25005, 24995, 25000), (24900, 24920, 24880, 24910)])
    t = run_day(day, breakout_long, time(9, 30), time(15, 15), slip_pct=0)[0]
    print(f"   stop at 24,960; the bar opened at 24,900: filled at {t.exit:,.0f} ({t.reason})")


def costs() -> None:
    print(f"   one lot of futures (50 x 25,000), round trip: Rs {round_trip_cost(1_250_000, 1_250_000, FUTURES):,.0f}")
    print(f"   one lot of an ATM option at 150, round trip:   Rs {round_trip_cost(7_500, 7_500, OPTIONS):,.0f}")
    c = round_trip_cost(1_250_000, 1_250_000, FUTURES)
    print(f"   the same futures round trip in index points: {c / 50:.1f}")


def runaway_sizing() -> None:
    rng = np.random.default_rng(2)
    days = pd.bdate_range("2025-01-01", periods=250)
    stop_pts = np.clip(rng.lognormal(np.log(40), 0.8, len(days)), 0.5, None)   # sometimes tiny
    win = rng.random(len(days)) < 0.55
    for label, cap in (("no cap", 1e9), ("3x notional cap", 3.0)):
        equity, eq, notional = 500_000.0, [], []
        for s, w in zip(stop_pts, win):
            lots = lots_for_risk(equity, 0.01, 25_000, 25_000 - s, 50, max_notional_mult=cap)
            notional.append(lots * 50 * 25_000)
            equity += lots * 50 * (s if w else -s)
            eq.append(equity)
        eq_s, not_s = pd.Series(eq, days), pd.Series(notional, days)
        try:
            check_sane(eq_s, not_s, max_notional_mult=3.0)
            verdict = "passes the sanity check"
        except SizingError as e:
            verdict = f"fails: {e}"
        print(f"   {label:<16} largest position {not_s.max() / 1e7:,.1f} crore; {verdict}")


if __name__ == "__main__":
    print("1. The fill clock"); fill_clock()
    print("2. A gap through the stop"); gap_through_stop()
    print("3. The cost model"); costs()
    print("4. Sizing from a stop that is sometimes tiny"); runaway_sizing()
