"""
Chapter 12 — Market Profile day types as a state machine.

SYNTHETIC one-minute sessions, 09:15-15:29, four kinds of day:
  trend      — a quiet first hour, a break up, a pullback to VWAP, a run to the close;
  turns      — the same morning, then a collapse through the first hour's low after
               lunch: a 'trend day' at 11:30 that ends two-sided;
  range      — stays inside the first hour;
  fast       — breaks up, dips to VWAP for one bar, and runs away the next.

The classifier below uses TOY rules (range extension only, no widths, no thresholds).
They are not the author's rules; they exist to exercise the framework.

  1. Attribution: the same trades grouped by the day type known at entry and by the
     end-of-day label.
  2. The VWAP-touch buffer against a check of the current bar only.

    python chapters/ch12/run.py
"""
from __future__ import annotations

from datetime import time as dtime

import numpy as np
import pandas as pd

from algo_builder.robustness.regime import by_regime
from algo_builder.strategies.market_profile_framework import DayTypeMachine, InitialBalance, on_bar
from algo_builder.strategies.mcx_session_vwap import SessionVWAP, VWAPTouchBuffer

IB_END = dtime(10, 15)                    # the first hour (public convention)
LOOKBACK = 3                              # ILLUSTRATIVE
TOY_RULES = [("TWO_SIDED", lambda c: c["extension"] == "both"),
             ("TREND_UP", lambda c: c["extension"] == "up"),
             ("TREND_DOWN", lambda c: c["extension"] == "down"),
             ("INSIDE", lambda c: c["extension"] == "none")]
ALLOWED = {"UNDETERMINED": {"INSIDE", "TREND_UP", "TREND_DOWN", "TWO_SIDED"},
           "INSIDE": {"TREND_UP", "TREND_DOWN", "TWO_SIDED"},
           "TREND_UP": {"TWO_SIDED"}, "TREND_DOWN": {"TWO_SIDED"}, "TWO_SIDED": set()}


def session(day: pd.Timestamp, kind: str, rng) -> pd.DataFrame:
    """Build one day minute by minute, steering the pullback to the running VWAP."""
    n, p, vw = 375, 25_000.0, SessionVWAP()
    opens, closes = [], []
    for i in range(n):
        o = p
        if i < 60:                                                  # the first hour
            p = 25_000 + float(np.clip(p - 25_000 + rng.normal(0, 3), -18, 18))
        elif kind == "range":
            p = 25_000 + float(np.clip(p - 25_000 + rng.normal(0, 3), -18, 18))
        elif i < 105:                                               # the break and run
            p += 1.8 + rng.normal(0, 1.5)
        elif kind == "fast" and i == 125:
            p = vw.pv / vw.v - 2.0                                  # one bar dips just below VWAP...
        elif kind == "fast" and i == 126:
            o = p = vw.pv / vw.v + 25.0                             # ...and the next opens above and runs
        elif kind != "fast" and 105 <= i < 135:                    # a slow pullback to VWAP
            target = vw.pv / vw.v - 2.0
            p += (target - p) / max(135 - i, 1) + rng.normal(0, 1.0)
        elif kind == "turns" and i >= 180:                          # after lunch: through the low
            p += -1.6 + rng.normal(0, 1.5)
        else:
            p += 0.5 + rng.normal(0, 1.5)
        opens.append(o)
        closes.append(p)
        vw.update(p, 1.0)
    idx = pd.date_range(day + pd.Timedelta("9h15min"), periods=n, freq="1min")
    o, c = np.array(opens), np.array(closes)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + 0.5, "low": np.minimum(o, c) - 0.5,
                         "close": c, "volume": 1.0}, index=idx)


def run_day(bars: pd.DataFrame, current_bar_only: bool) -> tuple[list[dict], str]:
    ib, machine = InitialBalance(IB_END), DayTypeMachine(TOY_RULES, allowed=ALLOWED)
    vw, buf, ctx, log, taken = SessionVWAP(), VWAPTouchBuffer(1 if current_bar_only else LOOKBACK), {}, [], set()

    def trend_up_pullback(c: dict):
        if buf.touched_recently() and c["bar"].close > c["vwap"]:
            return dict(side="long", entry_ts=c["ts"], entry=float(c["bar"].close))
        return None

    for ts, b in bars.iterrows():
        ctx["vwap"] = vw.update((b.high + b.low + b.close) / 3, b.volume)
        buf.push(b.high, b.low, ctx["vwap"])
        on_bar(ts, b, ib, machine, {"TREND_UP": trend_up_pullback}, ctx, log, taken)
    for t in log:                                                   # toy exit: the session's last close
        t["pnl"] = float(bars.close.iloc[-1]) - t["entry"]
    return log, machine.label


def main() -> None:
    rng = np.random.default_rng(12)
    days = pd.bdate_range("2025-01-01", periods=400)
    kinds = rng.choice(["trend", "turns", "range", "fast"], len(days), p=[0.3, 0.2, 0.3, 0.2])
    trades, entries = [], {True: 0, False: 0}
    for d, k in zip(days, kinds):
        bars = session(d, k, rng)
        log, end_label = run_day(bars, current_bar_only=False)
        for t in log:
            trades.append(dict(t, kind=k, end_of_day_label=end_label))
        if k == "fast":
            for flag in (True, False):
                entries[flag] += len(run_day(bars, current_bar_only=flag)[0])
    t = pd.DataFrame(trades)
    print("1. The same trades, grouped two ways (points, synthetic)")
    print("   by the day type known at entry:")
    print(by_regime(t, "day_type_at_entry").round(1).to_string())
    print("   by the end-of-day label (look-ahead):")
    print(by_regime(t, "end_of_day_label").round(1).to_string())
    n_fast = int((kinds == "fast").sum())
    print(f"\n2. Fast trend days: {n_fast}. Entries with the current bar only: {entries[True]}; "
          f"with a {LOOKBACK}-bar buffer: {entries[False]}")


if __name__ == "__main__":
    main()
