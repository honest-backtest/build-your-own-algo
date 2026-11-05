"""
Book 2, Chapter 21 — Trade-level edge, portfolio-level reality: the pullback in an uptrend
(after Tushar Chande, "Beyond Technical Analysis").

Shows: the daily engine on COMPLETED bars with every level shifted one bar
       (no look-ahead), the signal, the single wide trailing-low exit, an
       optional trend gate, and a slot manager with momentum-ranked overflow.
       Fills are at the NEXT session's open in both backtest and paper.

Pair this with robustness/drift_benchmark.py (trade-level edge) AND a portfolio
simulation (slots, cash) — many systems pass the first and fail the second.
Window lengths are ILLUSTRATIVE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

P = dict(high_lookback=20, recent_high_within=7, pullback_low=5,   # ILLUSTRATIVE
         trail_low=40, gate_sma=65, slots=10, momentum_n=126)


def prep(d: pd.DataFrame, p=P) -> pd.DataFrame:
    d = d.copy()
    n = p["high_lookback"]
    # bars since the n-day high, measured on data up to YESTERDAY
    d["bars_since_high"] = d.high.rolling(n).apply(lambda x: n - 1 - int(np.argmax(x)), raw=True).shift(1)
    d["low_prior"] = d.low.rolling(p["pullback_low"]).min().shift(1)
    d["trail"] = d.low.rolling(p["trail_low"]).min().shift(1)
    d["gate"] = (d.close > d.close.rolling(p["gate_sma"]).mean()).shift(1, fill_value=False)
    return d


def signal(d: pd.DataFrame, gated: bool, p=P) -> pd.Series:
    """Recent high (uptrend) + today's low undercuts the short-term low (pullback)."""
    s = (d.bars_since_high < p["recent_high_within"]) & (d.low < d.low_prior)
    return s & d.gate.astype(bool) if gated else s


def exit_today(d: pd.DataFrame, i: int) -> bool:
    """Exit when the close breaks the wide trailing low (decided at close, filled next open)."""
    return bool(d.close.iloc[i] < d.trail.iloc[i])


def allocate(candidates: list[str], held: set[str], closes: pd.DataFrame, p=P) -> list[str]:
    """Fill free slots; if signals exceed slots, take the strongest momentum names.
    A name with too little history to measure momentum ranks last, never first."""
    free = p["slots"] - len(held)
    fresh = sorted(c for c in candidates if c not in held)
    if free <= 0 or not fresh:
        return []
    look = min(p["momentum_n"], len(closes))
    mom = closes[fresh].iloc[-1] / closes[fresh].iloc[-look] - 1
    return list(mom.fillna(-np.inf).sort_values(ascending=False, kind="stable").index[:free])


def simulate(panel: dict[str, pd.DataFrame], capital: float, gated: bool = False,
             p=P) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """A slot-limited portfolio. panel: open/high/low/close, date x symbol. Signals on
    day t's close are bought at day t+1's open, equal weight from free cash; exits
    decided at a close are sold at the next open. Returns (taken trades, signals that
    found no free slot, daily equity marked to market with the cash part of it)."""
    syms = list(panel["close"].columns)
    feats = {s: prep(pd.DataFrame({k: panel[k][s] for k in ("open", "high", "low", "close")}), p) for s in syms}
    sig = pd.DataFrame({s: signal(feats[s], gated, p) for s in syms})
    ext = pd.DataFrame({s: feats[s].close < feats[s].trail for s in syms})
    days, op, cl = panel["close"].index, panel["open"], panel["close"]
    cash, pos, taken, skipped, equity = float(capital), {}, [], [], []
    for i in range(1, len(days)):
        d, prev = days[i], days[i - 1]
        for s in [s for s in pos if ext.at[prev, s]]:                 # exits decided yesterday
            sh, e_day, e_px = pos.pop(s)
            cash += sh * op.at[d, s]
            taken.append(dict(symbol=s, entry_date=e_day, exit_date=d, entry_px=e_px,
                              exit_px=op.at[d, s], shares=sh))
        cands = [s for s in syms if sig.at[prev, s] and np.isfinite(op.at[d, s])]
        buy = allocate(cands, set(pos), cl.loc[:prev], p)
        skipped += [dict(date=d, symbol=s) for s in cands if s not in pos and s not in buy]
        for s in buy:
            budget = cash / (p["slots"] - len(pos))
            sh = budget / op.at[d, s]
            pos[s] = (sh, d, op.at[d, s])
            cash -= budget
        equity.append((cash + sum(sh * cl.at[d, s] for s, (sh, _, _) in pos.items()), cash))
    return (pd.DataFrame(taken), pd.DataFrame(skipped),
            pd.DataFrame(equity, index=days[1:], columns=["equity", "cash"]))


def futures_leg_pnl(fut: pd.DataFrame, entry_date, exit_date, roll_cost_pts: float) -> float:
    """Points earned holding the near-month future from entry to exit, CONTRACT BY
    CONTRACT. fut: date-indexed, columns contract, price, and old_price on roll days
    (the expiring contract's price that day, at which it is closed). Price changes are
    summed only within a contract; each roll pays `roll_cost_pts`. A back-adjusted
    series is right for signals and wrong for this."""
    f = fut.loc[entry_date:exit_date]
    rolled = f.contract.ne(f.contract.shift()).iloc[1:]
    prev = f.price.shift().iloc[1:]
    held_move = f.price.iloc[1:] - prev                              # same contract
    if "old_price" in f:
        roll_move = f.old_price.iloc[1:] - prev                      # last day of the old one
    else:
        roll_move = pd.Series(0.0, index=prev.index)
    moves = held_move.where(~rolled, roll_move).sum()
    return float(moves - int(rolled.sum()) * roll_cost_pts)
