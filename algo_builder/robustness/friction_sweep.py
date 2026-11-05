"""
Book 2, Chapter 6 — Gross, net and the friction sweep.

Re-price the same trade list at increasing per-side costs and find the
BREAK-EVEN cost. If break-even sits below what you realistically pay, the
signal does not survive, however good the gross Sharpe looks. Also scales with
turnover: a high-turnover signal needs a much lower break-even.

Input: trades with  gross_ret  (fractional return per trade, before costs).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def friction_sweep(trades: pd.DataFrame, bps_per_side=(0, 2, 5, 10, 15, 20, 30, 50)) -> pd.DataFrame:
    rows = []
    for bps in bps_per_side:
        net = trades.gross_ret - 2 * bps / 1e4               # entry + exit
        t = net.mean() / (net.std(ddof=1) / np.sqrt(len(net))) if len(net) > 1 else np.nan
        rows.append(dict(bps_per_side=bps, mean_net_pct=100 * net.mean(),
                         win_rate=(net > 0).mean(), t_stat=t, sum_net_pct=100 * net.sum()))
    return pd.DataFrame(rows)


def break_even_bps(trades: pd.DataFrame) -> float:
    """Per-side cost (bps) at which mean net return reaches zero."""
    return float(trades.gross_ret.mean() / 2 * 1e4)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    demo = pd.DataFrame({"gross_ret": rng.normal(0.0012, 0.02, 800)})   # synthetic demo data
    print(friction_sweep(demo).round(3).to_string(index=False))
    print(f"break-even: {break_even_bps(demo):.1f} bps per side")
