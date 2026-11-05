"""
Book 2, Chapter 22 — Timing-critical execution: buy-on-gap (after Ernest Chan's buy-on-gap model).

Shows: the vectorised daily scan — gap below the prior LOW scaled by the stock's
       own volatility, a moving-average filter, a corporate-action/bad-tick guard,
       top-N slots — plus the live-order sizing guard and the signal/fill log.

Read with robustness/fill_delay_test.py: this backtest fills at the official
open print, and the chapter shows most of the edge disappears a minute later.
Parameters ILLUSTRATIVE.
"""
from __future__ import annotations

import pandas as pd

P = dict(lookback_std=90, lookback_ma=20, entry_z=1.0, top_n=10,   # ILLUSTRATIVE
         cost_bps_side=10, per_stock_budget=5_000, max_notional=10_000)


def scan(panel: dict[str, pd.DataFrame], p=P) -> pd.DataFrame:
    """panel: open/high/low/close, date x symbol. Returns daily strategy return."""
    op, lo, cl = panel["open"], panel["low"], panel["close"]
    std = cl.pct_change().rolling(p["lookback_std"]).std().shift(1)
    ma = cl.rolling(p["lookback_ma"]).mean().shift(1)
    prev_lo = lo.shift(1)
    overnight = op / cl.shift(1)
    ca_ok = (overnight > 0.75) & (overnight < 1.40)          # demergers / unadjusted splits
    cand = (op < prev_lo * (1 - p["entry_z"] * std)) & (op > ma) & ca_ok
    gap = ((op - prev_lo) / prev_lo).where(cand)
    ret = (cl - op) / op - 2 * p["cost_bps_side"] / 1e4       # open -> close, net
    rows = []
    for day, row in gap.iterrows():
        picks = row.dropna().sort_values().index[:p["top_n"]]  # most negative gaps first
        r = ret.loc[day, picks].dropna()
        rows.append((day, r.sum() / p["top_n"], len(r)))       # unused slots earn 0
    return pd.DataFrame(rows, columns=["date", "ret", "picks"]).set_index("date")


def live_qty(price: float, p=P) -> int:
    """Per-stock budget with a hard notional cap — a fat-finger guard."""
    q = int(p["per_stock_budget"] // price)
    return q if q * price <= p["max_notional"] else 0


def log_row(symbol: str, signal_px: float, fill_px: float | None, exit_px: float | None) -> dict:
    """Signal, fill and exit recorded separately: signal->fill is how much of the
    backtest you can actually collect."""
    slip = (fill_px / signal_px - 1) * 100 if fill_px else None
    return dict(symbol=symbol, signal_px=signal_px, fill_px=fill_px, exit_px=exit_px, slip_pct=slip)
