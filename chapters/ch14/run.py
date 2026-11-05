"""
Chapter 14 — Working with option contracts in code.

Three SYNTHETIC demonstrations:
  1. Expiries: a master whose weekly expiry moves from Thursday to Tuesday partway
     through the period, with two holiday shifts. Weekday arithmetic against the master.
  2. Following the ATM strike: a spot path that hovers near the midpoint between
     strikes, with and without hysteresis.
  3. Repainting: a Supertrend on 5-minute bars evaluated every minute on the bars
     INCLUDING the forming one, against the same indicator on completed bars only.

    python chapters/ch14/run.py
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from algo_builder.strategies.option_chart_momentum import AtmTracker, nearest_expiry
from algo_builder.strategies.supertrend_rsi import supertrend

STEP = 50                                                           # ILLUSTRATIVE strike step
HOLIDAYS = {date(2025, 8, 14), date(2025, 9, 30)}                  # synthetic: a Thursday, then a Tuesday


def synthetic_master() -> pd.DataFrame:
    days = pd.date_range("2025-06-02", "2025-12-31", freq="D").date
    holidays = HOLIDAYS
    rows = []
    for d in days:
        expiry_weekday = 3 if d < date(2025, 9, 1) else 1           # Thursday, then Tuesday
        if d.weekday() != expiry_weekday:
            continue
        e = d - timedelta(days=1) if d in holidays else d           # a holiday moves it a day earlier
        for k in range(24_000, 26_001, STEP):
            for kind in ("CE", "PE"):
                rows.append(dict(underlying="NIFTY", kind=kind, expiry=e, strike=float(k),
                                 symbol=f"NIFTY{e:%d%b%y}{k}{kind}".upper(), lot_size=75))
    return pd.DataFrame(rows)


def next_thursday(d: date) -> date:
    return d + timedelta(days=(3 - d.weekday()) % 7)


def demo_expiries() -> None:
    m = synthetic_master()
    trading = [d.date() for d in pd.bdate_range("2025-06-02", "2025-12-23") if d.date() not in HOLIDAYS]
    wrong = [d for d in trading if next_thursday(d) != nearest_expiry(m, "NIFTY", d)]
    after = [d for d in wrong if d >= date(2025, 9, 1)]
    print(f"1. {len(trading)} trading days. Weekday arithmetic ('next Thursday') picked the wrong "
          f"expiry on {len(wrong)}: {len(wrong) - len(after)} before the change (a holiday shift), "
          f"{len(after)} after the weekday changed.")


def demo_tracker(seed: int = 14) -> None:
    rng = np.random.default_rng(seed)
    strikes = np.arange(24_000, 26_001, STEP, dtype=float)
    switches = {0.0: 0, 10.0: 0}
    for _ in range(60):                                             # 60 sessions of 375 minutes
        spot = 25_025 + np.cumsum(rng.normal(0, 4, 375))           # starts on a midpoint
        for h in switches:
            t, last = AtmTracker(h), None
            for s in spot:
                k = t.update(s, strikes)
                switches[h] += last is not None and k != last
                last = k
    print(f"2. ATM switches over 60 sessions: no hysteresis {switches[0.0]}, "
          f"10 points of hysteresis {switches[10.0]}")


def demo_repaint(seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    n_days, flips_live, flips_done, ghosts = 40, 0, 0, 0
    for _ in range(n_days):
        px = 25_000 + np.cumsum(rng.normal(0, 3.0, 375))
        idx = pd.date_range("2026-01-05 09:15", periods=375, freq="1min")
        one = pd.DataFrame({"open": np.r_[px[0], px[:-1]], "close": px}, index=idx)
        one["high"], one["low"] = one[["open", "close"]].max(axis=1), one[["open", "close"]].min(axis=1)
        five = one.resample("5min").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        done = supertrend(five)
        seen = set()
        for i in range(len(one)):
            ts = one.index[i]
            b = ts.floor("5min")
            part = one.loc[b:ts]                                    # the forming 5-minute bar so far
            forming = pd.DataFrame({"open": [part.open.iloc[0]], "high": [part.high.max()],
                                    "low": [part.low.min()], "close": [part.close.iloc[-1]]}, index=[b])
            live = supertrend(pd.concat([five.loc[:b].iloc[:-1], forming]))
            if len(live) > 1 and live.iloc[-1] != live.iloc[-2]:
                seen.add(b)
        real = {t for t, (a, z) in zip(done.index[1:], zip(done.iloc[:-1], done.iloc[1:])) if a != z}
        flips_live += len(seen)
        flips_done += len(real)
        ghosts += len(seen - real)
    print(f"3. Over {n_days} sessions: flips on completed bars {flips_done}; bars on which a live loop "
          f"reading the forming bar saw a flip {flips_live}, of which {ghosts} had vanished by the close of the bar.")


if __name__ == "__main__":
    demo_expiries()
    demo_tracker()
    demo_repaint()
