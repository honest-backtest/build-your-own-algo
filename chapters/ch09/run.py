"""
Chapter 9 — The one-minute test.

Two SYNTHETIC demonstrations on one-minute bars:
  1. a gap buyer whose opening print overshoots and bounces within the first minute;
     its edge at the print is real only at the print;
  2. an 'index falls from its open' claim, built from an opening value that is
     slightly stale, tested on the index print, with delays, and on a tradable future;
plus 3. the engine sweep for a breakout whose stop and target move with the fill.

    python chapters/ch09/run.py
"""
from __future__ import annotations

from datetime import time as dtime

import numpy as np
import pandas as pd

from algo_builder.backtest.event_engine import Position
from algo_builder.infra.cost_model import FUTURES
from algo_builder.robustness.fill_delay_test import (engine_delay_sweep, fill_delay_table,
                                                     where_is_the_money)

DELAYS = (0, 1, 5, 15)


def session(day: pd.Timestamp, prices: np.ndarray) -> pd.DataFrame:
    """One-minute bars from 09:15: each bar opens at the previous price and closes at its own."""
    idx = pd.date_range(day + pd.Timedelta("9h15min"), periods=len(prices) - 1, freq="1min")
    o, c = prices[:-1], prices[1:]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c),
                         "close": c}, index=idx)


def gap_buyer(n: int = 600, seed: int = 9) -> tuple[dict, pd.DataFrame]:
    """The auction clears too low; most of the bounce happens in the first minute."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2023-01-02", periods=n)
    bars, sig = {}, []
    for d in days:
        p0 = 100.0
        bounce = rng.normal(0.0023, 0.0020)                       # the print's overshoot, repaid at once
        steps = rng.normal(0.00002, 0.0008, 30)                   # then an almost driftless half-hour
        path = p0 * np.exp(np.concatenate([[0.0, bounce], bounce + np.cumsum(steps)]))
        b = session(d, path)                                       # 09:15 ... 09:45
        bars[("GAP", d.date())] = b
        sig.append(("GAP", d.date(), float(b.close.iloc[-1]), b.index[-1]))
    return bars, pd.DataFrame(sig, columns=["symbol", "date", "exit_px", "exit_ts"])


def index_and_future(n: int = 1000, seed: int = 4) -> tuple[dict, dict, pd.DataFrame]:
    """The index's first value is slightly stale on the high side; the future is not."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-01-01", periods=n)
    idx_bars, fut_bars, sig = {}, {}, []
    for d in days:
        true = 25_000 * np.exp(np.cumsum(rng.normal(0.0, 0.0003, 376)))
        printed = true.copy()
        printed[0] = true[1] * (1 + rng.normal(0.0012, 0.0006))   # first value: assembled from stale prices
        fut = true * 1.002                                          # a tradable future, small basis
        i, f = session(d, printed), session(d, fut)
        idx_bars[("NIFTY", d.date())], fut_bars[("NIFTY", d.date())] = i, f
        sig.append(("NIFTY", d.date(), float(i.close.iloc[-1]), float(f.close.iloc[-1])))
    s = pd.DataFrame(sig, columns=["symbol", "date", "exit_px", "fut_exit_px"])
    return idx_bars, fut_bars, s


def breakout_sessions(n: int = 300, seed: int = 11) -> list[pd.DataFrame]:
    rng = np.random.default_rng(seed)
    out = []
    for d in pd.bdate_range("2025-01-01", periods=n):
        steps = rng.normal(0, 0.0006, 120)
        steps[30:36] += rng.choice([0.0008, -0.0003])              # a burst just after the range
        out.append(session(d, 25_000 * np.exp(np.concatenate([[0.0], np.cumsum(steps)]))))
    return out


def breakout(history: pd.DataFrame, ts: pd.Timestamp):
    rng_hi = history.high.iloc[:30].max()
    last = history.iloc[-1]
    if len(history) > 30 and last.close > rng_hi and history.close.iloc[-2] <= rng_hi:
        px = float(last.close)
        return Position(+1, px, 50, px - 20, px - 20, px + 30, ts)     # ILLUSTRATIVE stop and target
    return None


def main() -> None:
    pd.set_option("display.width", 120)
    bars, sig = gap_buyer()
    print("1. Gap buyer, exit at 09:45, cost 0.05% a round trip (synthetic)")
    print(fill_delay_table(bars, sig, delays=DELAYS, cost_rt=0.0005).round(3).to_string(index=False))
    print(where_is_the_money(bars, sig, clock_min=1))

    ib, fb, s = index_and_future()
    print("\n2. 'The index falls from its open': short at the open, cover at 15:29 (synthetic)")
    print(fill_delay_table(ib, s, side=-1, delays=(0, 1, 5)).round(3).to_string(index=False))
    fs = s.assign(exit_px=s.fut_exit_px)
    print("   on the future:")
    print(fill_delay_table(fb, fs, side=-1, delays=(0, 5)).round(3).to_string(index=False))

    print("\n3. Breakout with stop and target, re-run in the engine (synthetic)")
    print(engine_delay_sweep(breakout_sessions(), breakout, delays=(1, 2, 5),
                             range_end=dtime(9, 45), square_off=dtime(11, 10),
                             slip_pct=0.0, costs=FUTURES).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
