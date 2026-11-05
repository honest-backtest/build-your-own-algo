"""
Book 2, Chapter 20 — Breakout pipelines: VCP and SEPA.

Shows: pattern confirmation for screener candidates (uptrend, distance from the
       52-week high, momentum band, volume surge, contracting pullbacks), the
       volatility-stop / R-target exit, and the chapter's audit finding as code:
       drawdown must be measured on DAILY MARKED-TO-MARKET equity, not on
       closed-trade equity (which hides open losses and understates DD).

Values ILLUSTRATIVE. The pattern step is where a discretionary eye can beat
code — a hybrid (auto-screen, manual confirm) is a sensible design.
"""
from __future__ import annotations

import pandas as pd

P = dict(max_from_high_pct=25, rsi_lo=50, rsi_hi=75, vol_surge=1.5,      # ILLUSTRATIVE
         n_contractions=3, atr_stop_mult=2.0, target_r=3.0, max_hold=30)


def contracting_pullbacks(close: pd.Series, n: int, window: int = 60) -> bool:
    """Each successive pullback (peak-to-trough %) shallower than the one before."""
    c = close.tail(window).to_numpy()
    depths, peak = [], c[0]
    trough = c[0]
    for x in c[1:]:
        if x > peak:
            if peak > trough:
                depths.append((peak - trough) / peak)
            peak = trough = x
        trough = min(trough, x)
    d = depths[-n:]
    return len(d) == n and all(a > b for a, b in zip(d, d[1:]))


def validate(df: pd.DataFrame, rsi, p=P) -> bool:
    c, v = df.close, df.volume
    hi52 = c.rolling(250).max().iloc[-1]
    uptrend = c.iloc[-1] > c.rolling(50).mean().iloc[-1] > c.rolling(200).mean().iloc[-1]
    near_high = (hi52 - c.iloc[-1]) / hi52 * 100 <= p["max_from_high_pct"]
    r = rsi(c).iloc[-1]
    surge = v.iloc[-1] >= p["vol_surge"] * v.rolling(50).mean().iloc[-2]
    return bool(uptrend and near_high and p["rsi_lo"] <= r <= p["rsi_hi"] and surge
                and contracting_pullbacks(c, p["n_contractions"]))


def levels(entry: float, atr: float, p=P) -> tuple[float, float]:
    stop = entry - p["atr_stop_mult"] * atr
    return stop, entry + p["target_r"] * (entry - stop)


def mtm_drawdown(positions: pd.DataFrame, closes: pd.DataFrame, cash: pd.Series) -> float:
    """positions: date x symbol share counts held at each close; closes: same shape.
    Equity = cash + sum(shares x close) EVERY day — open losses included."""
    equity = cash + (positions * closes).sum(axis=1)
    return float((equity / equity.cummax() - 1).min())


def mtm_equity(trades: pd.DataFrame, closes: pd.DataFrame, capital: float) -> pd.Series:
    """Daily marked-to-market equity from a trade list (symbol, entry_date, exit_date,
    shares, entry_px, exit_px): cash moves on the entry and exit days, and every open
    position is valued at each day's close in between."""
    days = closes.index
    cash = pd.Series(0.0, index=days)
    held = pd.DataFrame(0.0, index=days, columns=closes.columns)
    for t in trades.itertuples():
        cash.loc[t.entry_date] -= t.shares * t.entry_px
        cash.loc[t.exit_date] += t.shares * t.exit_px
        held.loc[(days >= t.entry_date) & (days < t.exit_date), t.symbol] += t.shares
    return capital + cash.cumsum() + (held * closes.ffill()).sum(axis=1)


def review_queue(candidates: list[str], decisions: dict[str, tuple[str, str]]) -> tuple[list[str], list[str]]:
    """The hybrid: the screen proposes, a person disposes. `decisions` maps symbol ->
    (verdict, reviewer note), written by the reviewer. Returns (approved, still waiting);
    nothing is bought on the screen's word alone."""
    approved = [s for s in candidates if decisions.get(s, ("", ""))[0] == "approve"]
    waiting = [s for s in candidates if s not in decisions]
    return approved, waiting
