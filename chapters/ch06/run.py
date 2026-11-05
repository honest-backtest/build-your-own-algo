"""
Chapter 6 — Gross, net and the friction sweep.

A SYNTHETIC trade list shaped like the chapter's worked example (101 trades,
about 590 index points gross in total, so a little under six points a trade):
the friction sweep in points, the round-trip cost on two dates, the diagnosis,
and the author's percentage-return sweep on the same trades.

    python chapters/ch06/run.py
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from algo_builder.robustness.friction_sweep import break_even_bps, friction_sweep
from algo_builder.robustness.gross_net import cost_or_signal, friction_sweep_points, round_trip_points


def synthetic_trades(n: int = 101, total_pts: float = 590.0, seed: int = 11) -> pd.Series:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 45.0, n)
    return pd.Series(x - x.mean() + total_pts / n, name="gross_pts")      # mean fixed at 590/101


def main() -> None:
    gross = synthetic_trades()
    print(f"101 synthetic trades: {gross.sum():.0f} points gross, {gross.mean():.1f} a trade\n")
    print("Friction sweep, flat round-trip cost in index points:")
    print(friction_sweep_points(gross).round(2).to_string(index=False))

    then = round_trip_points(24_000, 75, date(2025, 6, 2))      # ILLUSTRATIVE level and lot
    now = round_trip_points(25_000, 50, date(2026, 6, 1))
    print(f"\nRound trip incl. 1 point slippage each side: {then:.1f} points (mid-2025 rates), "
          f"{now:.1f} points (2026 rates)")
    print("Diagnosis at 2025 cost:", cost_or_signal(gross, then))
    print("Diagnosis at 2026 cost:", cost_or_signal(gross, now))

    pct = pd.DataFrame({"gross_ret": gross / 24_000})           # the same trades as returns
    print(f"\nPercentage sweep (author's tool): break-even {break_even_bps(pct):.2f} bps per side")
    print(friction_sweep(pct, bps_per_side=(0, 1, 2, 3, 5)).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
