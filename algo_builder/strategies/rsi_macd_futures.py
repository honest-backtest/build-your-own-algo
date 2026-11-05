"""
Book 2, Chapter 6 — Gross, net and the friction sweep (worked example: RSI-MACD on index futures).

Shows: a 3-phase scan loop (exits FIRST, then watchlist, then entries), a
       continuous futures series stitched across rolls, and a trailing stop
       that activates after a favourable move and only ratchets one way.

Point values are ILLUSTRATIVE and per-index. Report gross and net side by side:
positive gross / negative net is a cost problem, not a signal problem.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

CONFIG = {                                   # ILLUSTRATIVE points per index
    "INDEX_A": dict(sl_pts=60, tp_pts=120, trail_activate_pts=60, trail_pts=40),
}
RSI_LOW, RSI_HIGH, SMA_N = 30, 70, 50        # ILLUSTRATIVE


def stitch_continuous(closes: list[pd.Series], roll_ts: list[pd.Timestamp]) -> pd.Series:
    """Back-adjusted continuous series. closes[i] is contract i's close series;
    roll_ts[i] is the bar where we switch from contract i to i+1 (both must trade
    there). Earlier history is shifted by the roll gap so indicators see no jump."""
    parts, adj = [closes[-1][closes[-1].index >= roll_ts[-1]]], 0.0
    for i in range(len(closes) - 2, -1, -1):
        adj += closes[i + 1].loc[roll_ts[i]] - closes[i].loc[roll_ts[i]]   # cumulative gap
        start = roll_ts[i - 1] if i > 0 else closes[i].index[0]
        seg = closes[i][(closes[i].index >= start) & (closes[i].index < roll_ts[i])]
        parts.append(seg + adj)
    return pd.concat(parts[::-1])


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def macd_cross_up(close: pd.Series) -> pd.Series:
    m = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    s = m.ewm(span=9, adjust=False).mean()
    return (m > s) & (m.shift(1) <= s.shift(1))


@dataclass
class Pos:
    direction: int
    entry: float
    sl: float
    tp: float
    peak: float
    trail_on: bool = False


def check_exit(p: Pos, price: float, cfg: dict) -> str | None:
    d = p.direction
    p.peak = max(p.peak, price) if d > 0 else min(p.peak, price)
    if not p.trail_on and d * (p.peak - p.entry) >= cfg["trail_activate_pts"]:
        p.trail_on = True
    if p.trail_on:
        new_sl = p.peak - d * cfg["trail_pts"]
        p.sl = max(p.sl, new_sl) if d > 0 else min(p.sl, new_sl)   # ratchet only
    if d * (price - p.sl) <= 0:
        return "TRAIL" if p.trail_on else "SL"
    if d * (price - p.tp) >= 0:
        return "TP"
    return None


def scan(bars15: pd.DataFrame, watch: dict, positions: dict, name: str, cfg: dict) -> list[str]:
    """One cycle on COMPLETED 15-minute bars. Phase order matters: exits first."""
    events, last = [], bars15.close.iloc[-1]
    if name in positions and (why := check_exit(positions[name], last, cfg)):
        events.append(f"EXIT {name} {why}"); del positions[name]
    r = rsi(bars15.close).iloc[-1]
    if r <= RSI_LOW:
        watch[name] = +1
    elif r >= RSI_HIGH:
        watch[name] = -1
    d = watch.get(name)
    sma = bars15.close.rolling(SMA_N).mean().iloc[-1]
    crossed = macd_cross_up(bars15.close if d == 1 else -bars15.close).iloc[-1] if d else False
    if d and name not in positions and crossed and d * (last - sma) > 0:
        positions[name] = Pos(d, last, last - d * cfg["sl_pts"], last + d * cfg["tp_pts"], last)
        watch.pop(name)
        events.append(f"ENTRY {name} {'LONG' if d > 0 else 'SHORT'} @ {last:.1f}")
    return events
