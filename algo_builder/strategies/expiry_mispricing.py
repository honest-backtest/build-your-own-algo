"""
Book 2, Chapter 15 — Multi-leg selling (expiry-afternoon pricing sidebar).

Shows: a cumulative VWAP tracker for the underlying, a Black-Scholes fair value
       with time-to-expiry in FRACTIONS OF A DAY, a % deviation signal, and
       exits on convergence or a multiple-of-premium stop.

Caution from the chapter: with T measured in minutes, model prices are extremely
sensitive — validate the pricing function against real quotes first.
Values ILLUSTRATIVE.
"""
from __future__ import annotations

from datetime import datetime

from algo_builder.strategies.opening_range_options import bs_price

P = dict(threshold_pct=15, stop_mult=4.0, target_pct=5, max_open=2, min_fair=2.0)   # ILLUSTRATIVE
MINUTES_PER_YEAR = 252 * 375


class VWAP:
    def __init__(self):
        self.pv = self.v = 0.0
        self.value = None

    def update(self, price: float, volume: float) -> None:
        self.pv += price * volume
        self.v += volume
        if self.v:
            self.value = self.pv / self.v


def t_years(now: datetime, expiry_close: datetime) -> float:
    return max((expiry_close - now).total_seconds() / 60, 1) / MINUTES_PER_YEAR


def deviation_pct(mid: float, vwap_spot: float, K: float, T: float, iv: float, is_call: bool,
                  p=P) -> float:
    """Market mid against the model's fair value, in per cent. Below `min_fair` points
    the model's value is too small for a percentage to mean anything: no signal."""
    fair = bs_price(vwap_spot, K, T, iv, is_call)
    return 100 * (mid - fair) / fair if fair >= p["min_fair"] else 0.0


def manage(open_pos: dict, side: str, mid: float, dev: float, p=P) -> str | None:
    """open_pos: 'CE'/'PE' -> {'entry': premium}. Short the rich side."""
    if side not in open_pos:
        if dev > p["threshold_pct"] and len(open_pos) < p["max_open"]:
            open_pos[side] = {"entry": mid}
            return "SELL"
        return None
    e = open_pos[side]["entry"]
    if mid >= e * p["stop_mult"] or dev <= p["target_pct"]:
        del open_pos[side]
        return "BUY_TO_CLOSE"
    return None
