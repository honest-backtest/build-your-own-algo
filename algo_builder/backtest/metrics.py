"""
Book 2, Chapter 1 (the report card as code) and Chapter 11 (the default report).

Trade-level and equity-level metrics plus a trade-sequence Monte Carlo.
Input: a trades DataFrame with at least columns  entry_ts, exit_ts, pnl  (net of costs)
and optionally  r  (R-multiple against the INITIAL stop).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def realised_equity(trades: pd.DataFrame, capital: float,
                    start: pd.Timestamp | None = None) -> pd.Series:
    """Account value after each day's closed trades, booked on the day each
    trade CLOSED, starting from the capital itself. Valid for drawdown only when
    trades open and close the same day; for overnight holdings use
    marked-to-market equity (Chapter 20)."""
    daily = (trades.groupby(pd.to_datetime(trades.exit_ts).dt.normalize())
             .pnl.sum().sort_index())
    if start is None:
        start = daily.index[0] - pd.Timedelta(days=1)
    opening = pd.Series([float(capital)], index=[start])
    return pd.concat([opening, capital + daily.cumsum()])


def max_drawdown(eq: pd.Series) -> float:
    return float((eq / eq.cummax() - 1).min())


def cagr(eq: pd.Series) -> float:
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    return float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1)


def sharpe(eq: pd.Series, rf_annual: float = 0.0) -> float:
    r = eq.pct_change().dropna() - rf_annual / TRADING_DAYS
    return float(np.sqrt(TRADING_DAYS) * r.mean() / r.std()) if r.std() > 0 else 0.0


def trade_stats(trades: pd.DataFrame) -> dict:
    p = trades.pnl
    wins, losses = p[p > 0], p[p < 0]
    out = dict(n=len(p), win_rate=float((p > 0).mean()),
               profit_factor=float(wins.sum() / -losses.sum()) if len(losses) else np.inf,
               avg_win_loss=float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else np.nan,
               mean_pnl=float(p.mean()),
               t_stat=float(p.mean() / (p.std(ddof=1) / np.sqrt(len(p)))) if len(p) > 1 else np.nan)
    if "r" in trades:
        out["expectancy_r"] = float(trades.r.mean())
    return out


def monte_carlo(trades: pd.DataFrame, capital: float, n_sim: int = 5000, block: int = 1,
                ruin_frac: float = 0.5, seed: int = 1) -> dict:
    """Bootstrap the trade list: draw trades at random, with replacement, in blocks of
    `block` consecutive trades (block > 1 keeps losing clusters together). How bad could
    the drawdown and the losing streak have been from trades like these?"""
    rng = np.random.default_rng(seed)
    pnl = trades.sort_values("exit_ts").pnl.to_numpy(float) if "exit_ts" in trades else trades.pnl.to_numpy(float)
    n = len(pnl)
    starts = np.arange(max(n - block + 1, 1))
    mdd, streak, ruined = np.empty(n_sim), np.empty(n_sim), 0
    for k in range(n_sim):
        picks = rng.choice(starts, -(-n // block))
        s = np.concatenate([pnl[p:p + block] for p in picks])[:n]
        eq = capital + np.concatenate([[0.0], np.cumsum(s)])        # the curve starts at the capital
        mdd[k] = (eq / np.maximum.accumulate(eq) - 1).min()
        ruined += eq.min() <= capital * (1 - ruin_frac)
        runs = np.diff(np.flatnonzero(np.concatenate([[True], s >= 0, [True]]))) - 1
        streak[k] = runs.max() if len(runs) else 0
    return dict(median_max_dd=float(np.median(mdd)), worst_5pct_max_dd=float(np.percentile(mdd, 5)),
                median_streak=float(np.median(streak)), worst_5pct_streak=float(np.percentile(streak, 95)),
                p_lose_half=ruined / n_sim)
