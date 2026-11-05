"""
Chapter 24 — Design exercises from the indicator-soup years.

  1. The central pivot range on SYNTHETIC 15-minute bars: how often "touched"
     means something different under two exact definitions.
  2. Two timeframes: a toy rule judged with the hourly bar aligned by its START
     (it sees the future) and by its COMPLETION.
  3. Gate ablation: four gates, one redundant, one useless, one useful.
  4. Per-recipe accounts for an "any one fires" option buyer.
  5. A system with no edge, judged on five-month windows.

Every number here is SYNTHETIC and every setting ILLUSTRATIVE.

    python chapters/ch24/run.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from algo_builder.data.synthetic import minute_bars, resample, trading_days
from algo_builder.lab.design_exercises import ablate, align_completed, by_recipe, cpr, touched, window_pf


def part1(m15: pd.DataFrame) -> None:
    days = sorted(set(m15.index.normalize()))
    trade = close = differ = 0
    for prev, day in zip(days, days[1:]):
        p = m15[m15.index.normalize() == prev]
        d = m15[m15.index.normalize() == day]
        lo, hi = cpr(p.high.max(), p.low.min(), p.close.iloc[-1])
        t = touched(d, lo, "trade").any() or touched(d, hi, "trade").any()
        c = touched(d, lo, "close").any() or touched(d, hi, "close").any()
        trade += t; close += c; differ += t != c
    n = len(days) - 1
    print(f"1. CPR: {n} days. Touched by a trade on {trade}; closed across on {close}; "
          f"the two definitions disagree on {differ} days ({differ / n:.0%}).")


def part2(m15: pd.DataFrame, h1: pd.DataFrame) -> None:
    lo = m15[["open", "close"]].rename_axis("ts")
    lo["fwd"] = lo.close.shift(-4) / lo.close - 1                 # next hour's return
    hi = h1[["open", "close"]].rename_axis("ts")
    good = align_completed(lo, "15min", hi, "60min")
    naive = pd.merge_asof(lo.reset_index(), hi.add_suffix("_hi").reset_index(), on="ts").set_index("ts")
    for name, df in (("by start (sees the future)", naive), ("by completion", good)):
        sig = (df.close_hi > df.open_hi) & (df.close > df.open)
        r = df.fwd[sig].dropna() * 1e4
        print(f"2. Hourly aligned {name}: {len(r)} signals, mean next-hour move {r.mean():+.1f} bp")


def part3(rng) -> None:
    n = 20_000
    trend, strength = rng.normal(size=n), rng.normal(size=n)
    outcome = pd.Series(0.10 * trend + 0.10 * strength + rng.normal(size=n) - 0.05)
    gates = pd.DataFrame({"rsi_up": trend + 0.5 * rng.normal(size=n) > 0,
                          "macd_up": trend + 0.3 * rng.normal(size=n) > 0,
                          "stoch_up": rng.normal(size=n) > 0,
                          "adx_strong": strength > 0})
    print("3. Gate ablation (outcome in units of one trade's noise):")
    print(ablate(gates, outcome).round(3).to_string())


def part4(rng) -> None:
    n = 3_000
    common = rng.normal(size=n)                                     # shared information
    fired = pd.DataFrame({k: common + rng.normal(size=n) * s > 1.4 for k, s in
                          (("oversold_macd", 0.8), ("band_volume", 0.8), ("vwap_adx", 0.6),
                           ("stochastic", 0.8), ("vol_burst", 1.5))})
    edge = {"vol_burst": 0.25}
    pnl = pd.Series(rng.normal(size=n) - 0.08, index=fired.index)  # time decay and costs
    for k, e in edge.items():
        pnl[fired[k]] += e
    any_ = fired.any(axis=1)
    p = pnl[any_]
    print(f"4. Any-one-fires: {any_.sum()} trades, mean {p.mean():+.3f}, "
          f"profit factor {p[p > 0].sum() / -p[p < 0].sum():.2f}")
    print(by_recipe(fired[any_], p).round(3).to_string())


def part5(rng) -> None:
    days = pd.bdate_range("2020-01-01", "2025-12-31")
    take = rng.random(len(days)) < 0.8                                # about 17 trades a month
    pnl = pd.Series(rng.normal(-0.02, 1, take.sum()), index=days[take])
    w = window_pf(pnl, 5)
    full = pnl[pnl > 0].sum() / -pnl[pnl < 0].sum()
    print(f"5. No edge, {len(pnl)} trades, overall PF {full:.2f}. Five-month windows: "
          f"PF from {w.pf.min():.2f} to {w.pf.max():.2f}; above 1.2 in {(w.pf > 1.2).sum()} "
          f"of {len(w)}, below 0.8 in {(w.pf < 0.8).sum()}.")


if __name__ == "__main__":
    bars = minute_bars(trading_days("2024-01-01", "2024-12-31"), seed=24)
    m15, h1 = resample(bars, "15min"), resample(bars, "60min")
    part1(m15)
    part2(m15, h1)
    rng = np.random.default_rng(24)
    part3(rng)
    part4(rng)
    part5(rng)
