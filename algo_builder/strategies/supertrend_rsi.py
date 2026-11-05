"""
Book 2, Chapter 14 — Working with option contracts in code (repainting sidebar).

Shows: a Supertrend implementation with the standard band "ratchet" rules,
       a flip+RSI entry on COMPLETED bars only, and a candle counter so one bar
       can never produce two entries.

Indicators recomputed on a forming bar repaint; a backtest on completed bars
will not match a live loop that does that. Parameters are ILLUSTRATIVE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ATR_N, MULT, RSI_LONG, RSI_SHORT = 7, 3.0, 55, 45     # ILLUSTRATIVE


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    pc = df.close.shift(1)
    tr = np.maximum(df.high - df.low, np.maximum((df.high - pc).abs(), (df.low - pc).abs()))
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def supertrend(df: pd.DataFrame, n: int = ATR_N, m: float = MULT) -> pd.Series:
    """Returns +1 (up-trend) / -1 (down-trend) per bar."""
    hl2, a = (df.high + df.low) / 2, atr(df, n)
    ub, lb = (hl2 + m * a).to_numpy(), (hl2 - m * a).to_numpy()
    c = df.close.to_numpy()
    fub, flb, trend = ub.copy(), lb.copy(), np.ones(len(df), dtype=int)
    for i in range(1, len(df)):
        fub[i] = min(ub[i], fub[i - 1]) if c[i - 1] <= fub[i - 1] else ub[i]
        flb[i] = max(lb[i], flb[i - 1]) if c[i - 1] >= flb[i - 1] else lb[i]
        if c[i] > fub[i - 1]:
            trend[i] = 1
        elif c[i] < flb[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
    return pd.Series(trend, index=df.index)


class Signaller:
    def __init__(self):
        self.last_bar_used = None                     # candle counter

    def on_completed_bar(self, bars: pd.DataFrame, rsi: pd.Series) -> int:
        """Call only after a bar has CLOSED; `bars` must exclude the forming bar."""
        ts = bars.index[-1]
        if rsi.index[-1] != ts:
            raise ValueError(f"RSI ends at {rsi.index[-1]}, bars at {ts}: not the same bar")
        if ts == self.last_bar_used or len(bars) < 2:
            return 0
        st = supertrend(bars)
        flip_up = st.iloc[-1] == 1 and st.iloc[-2] == -1
        flip_dn = st.iloc[-1] == -1 and st.iloc[-2] == 1
        sig = 1 if flip_up and rsi.iloc[-1] >= RSI_LONG else -1 if flip_dn and rsi.iloc[-1] <= RSI_SHORT else 0
        if sig:
            self.last_bar_used = ts                   # buy ATM CE on +1, ATM PE on -1
        return sig
