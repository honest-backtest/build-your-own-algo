"""
Chapter 19 — One signal, three vehicles, two monitoring clocks.

  1. Two monitoring clocks. 3,000 SYNTHETIC long trades on hourly bars with a small
     edge in their closes and long wicks below them. The same trades, exited by a
     loop that checks closes once an hour, and by stop and target orders resting at
     the exchange.
  2. The every-filter test, on a toy backtest whose settings include an ADX filter
     that the entry rule never reads.
  3. One shadow-log row.

Every setting is ILLUSTRATIVE.

    python chapters/ch19/run.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from algo_builder.strategies.hourly_signal_to_options import shadow_row, stop_fill
from algo_builder.strategies.rsi_macd_dipbuy import dead_filters

STOP, TARGET, HOLD = 0.97, 1.05, 30                             # ILLUSTRATIVE


def trade_bars(rng, n: int = HOLD) -> pd.DataFrame:
    c = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.009, n)))   # a small edge in the closes
    o = np.r_[100.0, c[:-1]]
    hours = [d + pd.Timedelta(hours=h, minutes=15) for d in pd.bdate_range("2025-03-03", periods=6) for h in range(9, 15)]
    idx = pd.DatetimeIndex(hours[:n])
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.003, n))),
                         "low": np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.008, n))),   # long lower wicks
                         "close": c}, index=idx)


def exit_price(b: pd.DataFrame, mode: str) -> float:
    s = stop_fill(b, 100 * STOP, mode)
    if mode == "intrabar":
        t = b.high >= 100 * TARGET
        t_ts = t.idxmax() if t.any() else None
    else:
        t = b.close >= 100 * TARGET
        t_ts = t.idxmax() if t.any() else None
    if s and (t_ts is None or s[0] <= t_ts):                    # the stop first when both
        return s[1]
    if t_ts is not None:
        if mode == "intrabar":
            return 100 * TARGET
        later = b.loc[t_ts:].iloc[1:]
        return float(later.open.iloc[0]) if len(later) else float(b.loc[t_ts, "close"])
    return float(b.close.iloc[-1])


def part1(seed: int = 19) -> None:
    rng = np.random.default_rng(seed)
    trades = [trade_bars(rng) for _ in range(3_000)]
    print("1. 3,000 synthetic hourly trades, the same signals, two monitoring clocks")
    for mode in ("close", "intrabar"):
        r = np.array([exit_price(b, mode) / 100 - 1 for b in trades]) - 0.001
        pf = r[r > 0].sum() / -r[r < 0].sum()
        print(f"   {mode:>8}: profit factor {pf:.2f}, mean {100 * r.mean():+.2f}% a trade, "
              f"stopped out {np.mean([stop_fill(b, 100 * STOP, mode) is not None for b in trades]):.0%}")


def part2() -> None:
    days = pd.bdate_range("2025-01-01", periods=300)
    rng = np.random.default_rng(2)
    fixture = pd.DataFrame({"rsi": rng.uniform(10, 90, 300), "above_sma": rng.random(300) < 0.5,
                            "adx": rng.uniform(5, 50, 300)}, index=days)

    def run_backtest(f: pd.DataFrame, p: dict) -> list:
        log = {"adx_min": p["adx_min"]}  # noqa: F841  configured, and logged...
        entries = f[(f.rsi <= p["rsi_oversold"]) & (f.above_sma | (not p["require_sma"]))]
        return [str(d.date()) for d in entries.index]           # ...but never used to decide

    params = dict(rsi_oversold=30, require_sma=True, adx_min=20)
    alts = dict(rsi_oversold=25, require_sma=False, adx_min=40)
    print("\n2. Dead filters:", dead_filters(run_backtest, fixture, params, alts))


def part3() -> None:
    rng = np.random.default_rng(5)
    b = trade_bars(rng)
    print("\n3. Shadow log:", shadow_row("T-0001", b, 100 * STOP, live_mode="close"))


if __name__ == "__main__":
    part1()
    part2()
    part3()
