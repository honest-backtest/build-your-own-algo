"""
Book 2, Chapter 7 — Benchmarks that fight back.
Also used in Chapters 21 and 25.

In a rising market, almost any long entry held N days makes money. The question
is whether YOUR entries beat RANDOM entries into the same instruments, held for
the same periods, at the same costs. Bootstrap the random distribution and report
where the strategy's mean sits in it (an empirical p-value).

Inputs:
  closes : DataFrame of daily closes, one column per symbol.
  trades : DataFrame with columns  symbol, ret (net fractional return), and either
           a `hold` column (bars held, trade by trade) or a fixed `hold` argument.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def random_entry_means(closes: pd.DataFrame, trades: pd.DataFrame, hold: int | None = None,
                       cost_rt: float = 0.0, n_draws: int = 2000, warmup: int = 66,
                       seed: int = 7) -> np.ndarray:
    """Each draw re-enters every trade's symbol on a random day and holds it for that
    trade's own holding period; the draw's mean is one sample of 'skill-free' trading."""
    rng = np.random.default_rng(seed)
    holds = trades["hold"].to_numpy(int) if hold is None else np.full(len(trades), hold)
    series = {s: closes[s].dropna().to_numpy() for s in trades.symbol.unique()}
    syms = trades.symbol.to_numpy()
    out = np.full(n_draws, np.nan)
    for k in range(n_draws):
        rets = []
        for sym, h in zip(syms, holds):
            c = series[sym]
            hi = len(c) - h - 2
            if hi <= warmup:
                continue
            s = rng.integers(warmup, hi)
            rets.append(c[s + 1 + h] / c[s + 1] - 1 - cost_rt)
        if rets:
            out[k] = np.mean(rets)
    return out[np.isfinite(out)]


def drift_report(trades: pd.DataFrame, random_means: np.ndarray) -> dict:
    """Strategy mean against the random distribution. The p-value counts the strategy
    itself as one draw, so it is never reported as exactly zero."""
    m = trades.ret.mean()
    beats = int((random_means >= m).sum())
    return dict(strategy_mean_pct=100 * m,
                random_mean_pct=100 * random_means.mean(),
                excess_pct=100 * (m - random_means.mean()),
                p_value=(beats + 1) / (len(random_means) + 1))      # small = beats drift
