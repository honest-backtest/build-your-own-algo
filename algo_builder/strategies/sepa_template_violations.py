"""
Book 2, Chapter 20 — Breakout pipelines (after Mark Minervini's published SEPA method).

Shows: the trend template on a date x symbol panel, an in-house relative-strength
       percentile across the universe, the breakout trigger, a corporate-action
       guard, and the reusable VIOLATIONS engine for the first sessions of a trade.

The template mirrors the book's published criteria; numeric parameters are the
book's well-known defaults or ILLUSTRATIVE and should be treated as starting points.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

P = dict(rs_min=70, slope_days=20, vol_expand=1.4, ca_gap=0.25,       # ILLUSTRATIVE
         violation_window=10, violations_to_exit=2, min_traded_value=5e7,
         weak_breakout_vol=1.2, heavy_reversal_vol=1.5)


def indicators(panel: dict[str, pd.DataFrame], p=P) -> dict:
    cl, vol = panel["close"], panel["volume"]
    ma50, ma150, ma200 = (cl.rolling(n).mean() for n in (50, 150, 200))
    hi52, lo52 = cl.rolling(250).max(), cl.rolling(250).min()
    score = 2 * cl.pct_change(126) + cl.pct_change(63) + cl.pct_change(250)
    rs = score.rank(axis=1, pct=True) * 100                     # percentile vs universe
    template = ((cl > ma150) & (cl > ma200) & (ma150 > ma200)
                & (ma200 > ma200.shift(p["slope_days"]))
                & (ma50 > ma150) & (cl > ma50)
                & (cl >= 1.30 * lo52) & (cl >= 0.75 * hi52) & (rs >= p["rs_min"]))
    breakout = ((cl > hi52.shift(1))
                & (vol >= p["vol_expand"] * vol.rolling(50).mean().shift(1))
                & ((cl * vol).rolling(20).mean() >= p["min_traded_value"]))
    overnight = panel["open"] / cl.shift(1)
    ca_ok = (overnight > 1 - p["ca_gap"]) & (overnight < 1.40)   # demergers / bad ticks
    return dict(template=template, breakout=breakout & ca_ok, rs=rs,
                ma20=cl.rolling(20).mean(), ma50=ma50)


class TradeTracker:
    """Per-position counters updated each session after entry. `breakout_vol_ratio`
    is the breakout day's volume over its average, known when the trade is entered."""

    def __init__(self, entry: float, breakout_vol_ratio: float):
        self.entry, self.breakout_vol_ratio, self.max_gain = entry, breakout_vol_ratio, 0.0
        self.lower_lows = self.good_closes = self.bad_closes = 0
        self.heavy_reversal = False
        self.prev_low = None

    def update(self, o, h, lo, c, vol: float, avg_vol: float, p=P) -> None:
        self.max_gain = max(self.max_gain, c / self.entry - 1)
        self.lower_lows = self.lower_lows + 1 if self.prev_low is not None and lo < self.prev_low else 0
        self.prev_low = lo
        if c < self.entry and avg_vol > 0 and vol >= p["heavy_reversal_vol"] * avg_vol:
            self.heavy_reversal = True                         # back below the pivot, on heavy volume
        rng = h - lo
        if rng > 0:
            if (c - lo) / rng >= 0.5:
                self.good_closes += 1
            else:
                self.bad_closes += 1


def violations(t: TradeTracker, close: float, ma20: float, held: int, p=P) -> bool:
    """True -> exit. Counts independent warning signs early in the trade."""
    if held > p["violation_window"]:
        return False
    v = 0
    v += t.breakout_vol_ratio < p["weak_breakout_vol"]          # the breakout itself was weak
    v += t.heavy_reversal
    v += bool(np.isfinite(ma20) and close < ma20)
    v += t.lower_lows >= 3
    v += t.bad_closes > t.good_closes and t.bad_closes + t.good_closes >= 6
    v += t.max_gain >= 0.05 and close <= t.entry                 # gave back a decent gain
    return v >= p["violations_to_exit"]
