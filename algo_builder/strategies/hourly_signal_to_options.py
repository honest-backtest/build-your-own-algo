"""
Book 2, Chapter 19 — One signal, three vehicles, two monitoring clocks.

Shows: mapping a stock signal to an option (monthly expiry, strike step), a
       liquidity check for SELLING a put (two-sided book, bid size, spread,
       price band; execution price = the BID, not the LTP), a capped marketable
       limit, a premium-capture take-profit — and the chapter's main lesson:
       bar-close vs intra-bar exit monitoring give different results.

Values ILLUSTRATIVE.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

P = dict(max_spread_pct=5.0, min_bid_lots=2, take_profit_capture_pct=70,   # ILLUSTRATIVE
         otm_pct=5.0)


def strike_step_heuristic(price: float) -> float:
    """Fallback only — prefer the actual listed strikes from the instrument master."""
    return max(5.0, round(price * 0.005 / 5.0) * 5.0)


def short_put_strike(spot: float, listed: list[float], p=P) -> float:
    target = spot * (1 - p["otm_pct"] / 100)
    below = [k for k in listed if k <= target]
    return max(below) if below else min(listed)


def can_short(quote: Optional[dict], qty: int, p=P) -> tuple[bool, str, Optional[float]]:
    if quote is None:
        return False, "no quote", None
    bid, ask = quote.get("bid", 0), quote.get("ask", 0)
    if bid <= 0 or ask <= 0:
        return False, "one-sided book — a short could be trapped", None
    lo, hi = quote.get("lower_circuit", 0), quote.get("upper_circuit", 0)
    if (hi and bid > hi) or (lo and bid < lo):
        return False, "bid outside price band", None
    if quote.get("bid_qty", 0) < qty * p["min_bid_lots"]:
        return False, "thin bid", None
    spread = (ask - bid) / ((ask + bid) / 2) * 100
    if spread > p["max_spread_pct"]:
        return False, f"spread {spread:.1f}%", None
    return True, "ok", bid                                   # premium actually received


def premium_captured(entry_prem: float, now_prem: float, p=P) -> bool:
    return (entry_prem - now_prem) / entry_prem * 100 >= p["take_profit_capture_pct"]


def stop_hit(bars: pd.DataFrame, stop: float, mode: str) -> pd.Timestamp | None:
    """mode='close': live loop checks once per completed bar.
       mode='intrabar': live loop watches ticks. Pick the one your live loop does."""
    hit = bars.close <= stop if mode == "close" else bars.low <= stop
    return hit.idxmax() if hit.any() else None


def stop_fill(bars: pd.DataFrame, stop: float, mode: str) -> tuple[pd.Timestamp, float] | None:
    """When and where a long position's stop is actually filled. 'intrabar': a stop
    order at the exchange fills at the stop, or at the open if the bar gaps below it.
    'close': the loop sees a close below the stop and sells at the NEXT bar's open."""
    ts = stop_hit(bars, stop, mode)
    if ts is None:
        return None
    if mode == "intrabar":
        return ts, min(float(bars.loc[ts, "open"]), stop)
    later = bars.loc[ts:].iloc[1:]
    return (later.index[0], float(later.open.iloc[0])) if len(later) else (ts, float(bars.loc[ts, "close"]))


def shadow_row(trade_id: str, bars_after_entry: pd.DataFrame, stop: float, live_mode: str) -> dict:
    """What the OTHER monitoring rule would have done to the same trade, logged beside
    the live result so the two can be compared on real trades as they happen."""
    other = "close" if live_mode == "intrabar" else "intrabar"
    live, alt = stop_fill(bars_after_entry, stop, live_mode), stop_fill(bars_after_entry, stop, other)
    return dict(trade=trade_id, live_mode=live_mode,
                live_stop_ts=live[0] if live else None, live_stop_px=live[1] if live else None,
                shadow_mode=other, shadow_stop_ts=alt[0] if alt else None,
                shadow_stop_px=alt[1] if alt else None)
