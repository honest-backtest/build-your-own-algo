"""
Chapter 21 — Trade-level edge, portfolio-level reality: the pullback in an uptrend.

A SYNTHETIC universe of 60 shares, 2014-2025: a common market factor, a persistent
drift that differs by share, and a little short-term reversal (so dips in rising
shares tend to recover). The same pullback rule is then tested two ways:
  1. trade by trade, every signal taken, against random entries held as long;
  2. as a portfolio of ten slots with cash, marked to market daily, against an
     equal-weight hold of the whole universe.

Every setting is ILLUSTRATIVE.

    python chapters/ch21/run.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from algo_builder.backtest.metrics import cagr, max_drawdown
from algo_builder.robustness.drift_benchmark import drift_report, random_entry_means
from algo_builder.strategies.pullback_in_uptrend import P, prep, signal, simulate


def universe(n: int = 60, seed: int = 21) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2014-01-01", "2025-12-31")
    T = len(days)
    mkt = rng.normal(0.0003, 0.009, T)
    mkt[(days >= "2020-02-20") & (days < "2020-03-24")] -= 0.015
    closes, opens, highs, lows = {}, {}, {}, {}
    for k in range(n):
        mu = rng.normal(0.0002, 0.0004)
        r = np.zeros(T)
        e = rng.normal(0, 0.014, T)
        for t in range(T):
            r[t] = mu + 1.0 * mkt[t] + e[t] - 0.06 * r[max(0, t - 5):t].sum()   # short-term reversal
        c = 100 * np.exp(np.cumsum(r))
        o = c / np.exp(r) * np.exp(rng.normal(0, 0.003, T))
        s = f"S{k:02d}"
        closes[s], opens[s] = c, o
        highs[s] = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.006, T)))
        lows[s] = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.006, T)))
    return {k: pd.DataFrame(v, index=days) for k, v in
            (("open", opens), ("high", highs), ("low", lows), ("close", closes))}


def every_signal(panel) -> pd.DataFrame:
    """Trade level: every signal taken, from the next open to the next open after an exit."""
    rows = []
    for s in panel["close"]:
        d = prep(pd.DataFrame({k: panel[k][s] for k in ("open", "high", "low", "close")}))
        sig, ext = signal(d, gated=False).to_numpy(), (d.close < d.trail).to_numpy()
        i, n = 0, len(d)
        while i < n - 2:
            if sig[i]:
                j = i + 1
                while j < n - 1 and not ext[j]:
                    j += 1
                rows.append(dict(symbol=s, entry_idx=i, hold=j - i,
                                 ret=d.close.iloc[min(j + 1, n - 1)] / d.close.iloc[i + 1] - 1))
                i = j + 1
            else:
                i += 1
    return pd.DataFrame(rows)


def main() -> None:
    panel = universe()
    cl = panel["close"]
    t = every_signal(panel)
    rnd = random_entry_means(cl, t, cost_rt=0.0, n_draws=300, warmup=P["trail_low"] + 1)
    rep = drift_report(t, rnd)
    pf = t.ret[t.ret > 0].sum() / -t.ret[t.ret < 0].sum()
    print(f"1. Trade level: {len(t):,} trades, mean {rep['strategy_mean_pct']:+.1f}%, "
          f"random entries held as long {rep['random_mean_pct']:+.1f}%, "
          f"excess {rep['excess_pct']:+.1f}% (p = {rep['p_value']:.3f}), profit factor {pf:.2f}")

    taken, skipped, book = simulate(panel, capital=10_00_000)
    eq = book.equity
    hold = (cl / cl.iloc[0]).mean(axis=1) * 10_00_000
    print(f"2. Portfolio of {P['slots']} slots: {len(taken):,} trades taken; CAGR {cagr(eq):.1%}, "
          f"max drawdown {max_drawdown(eq):.1%}; on average {(book.cash / eq).mean():.0%} of the book in cash")
    print(f"   equal-weight hold of the universe: CAGR {cagr(hold.loc[eq.index]):.1%}, "
          f"max drawdown {max_drawdown(hold.loc[eq.index]):.1%}")
    per_day = skipped.groupby("date").size()
    print(f"   signals turned away for want of a slot: {len(skipped):,} signal-days on {per_day.size:,} days, "
          f"up to {int(per_day.max())} on one day")


if __name__ == "__main__":
    main()
