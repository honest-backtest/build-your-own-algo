"""
Book 2, Chapter 7 — two more benchmarks. Illustrative.

  * same_day_means: the strategy's DATES, with random other stocks. If a strategy
    beats random days in the same stock but not random stocks on the same days,
    its edge is timing (often a market rebound), not stock selection.
  * hold_the_list: the control every strategy on a curated list needs: buy every
    name on the list at the start, equal-weighted, and do nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def same_day_means(closes: pd.DataFrame, trades: pd.DataFrame, cost_rt: float = 0.0,
                   n_draws: int = 2000, seed: int = 7) -> np.ndarray:
    """trades: entry_idx (row of the signal bar in `closes`) and hold. Each draw keeps
    every trade's dates and replaces its stock with a random one that traded then."""
    rng = np.random.default_rng(seed)
    c = closes.to_numpy(float)
    out = np.empty(n_draws)
    for k in range(n_draws):
        rets = []
        for i, h in zip(trades.entry_idx.to_numpy(int), trades.hold.to_numpy(int)):
            a, b = c[i + 1], c[min(i + 1 + h, len(c) - 1)]
            ok = np.flatnonzero(np.isfinite(a) & np.isfinite(b))
            if len(ok):
                j = rng.choice(ok)
                rets.append(b[j] / a[j] - 1 - cost_rt)
        out[k] = np.mean(rets) if rets else np.nan
    return out[np.isfinite(out)]


def hold_the_list(closes: pd.DataFrame, symbols: list[str], start, end,
                  cost_rt: float = 0.0) -> dict:
    """Equal-weight buy-and-hold of `symbols` from `start` to `end`, one round trip."""
    px = closes.loc[start:end, symbols].dropna(axis=1, how="any")
    growth = px.iloc[-1] / px.iloc[0]
    total = float(growth.mean()) * (1 - cost_rt) - 1
    years = max((px.index[-1] - px.index[0]).days / 365.25, 1e-9)
    return dict(names=px.shape[1], total_return_pct=100 * total,
                cagr_pct=100 * ((1 + total) ** (1 / years) - 1))
