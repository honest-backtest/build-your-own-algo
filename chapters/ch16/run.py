"""
Chapter 16 — Expiry day: daily data, then the recorder.

  1. 300 SYNTHETIC expiry days. One-minute index paths with overnight gaps; option
     prices from the model at 1.15x the true volatility (the seller's edge, which
     synthetic data has no other way to show). The exchange-style daily file holds
     each option's open, high and close, with the closes OVERWRITTEN by the
     settlement, as on a real expiry day. Then:
       - the daily file read naively, and read with leg_exit_price;
       - the sum of the two legs' daily highs against the straddle's real peak.
  2. The recorder, run against a simulated broker for a session (no real sleeping),
     then read back: both legs on one clock, and the measured spreads.

Every setting is ILLUSTRATIVE.

    python chapters/ch16/run.py
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from algo_builder.strategies.expiry_straddle_framework import (atm_from_futures_open, daily_straddle,
                                                               legs_on_one_clock, record_quotes,
                                                               spread_stats, straddle_pnl_pct)
from algo_builder.strategies.opening_range_options import bs_price

VOL, IV_MULT, YEAR_MIN, STEP = 0.13, 1.15, 252 * 375, 50


def expiry_day(prev_close: float, rng) -> tuple[pd.DataFrame, float, float, float]:
    """A daily file for one expiry day, plus the futures open, the settlement, and
    the straddle's true intraday peak at the open-based strike."""
    open_ = prev_close * (1 + rng.normal(0, 0.006))
    path = open_ * np.exp(np.cumsum(np.r_[0.0, rng.normal(0, VOL / np.sqrt(YEAR_MIN), 374)]))
    settle, iv = float(path[-1]), VOL * IV_MULT
    centre = round(open_ / STEP) * STEP
    rows, peak = [], None
    for k in np.arange(centre - 10 * STEP, centre + 10 * STEP + 1, STEP):
        for kind in ("CE", "PE"):
            px = [bs_price(s, k, (375 - i) / YEAR_MIN, iv, kind == "CE") for i, s in enumerate(path)]
            rows.append(dict(strike=float(k), opt_type=kind, open=px[0], high=max(px), close=settle))
        if k == centre:
            c = [bs_price(s, k, (375 - i) / YEAR_MIN, iv, True) for i, s in enumerate(path)]
            p = [bs_price(s, k, (375 - i) / YEAR_MIN, iv, False) for i, s in enumerate(path)]
            peak = max(a + b for a, b in zip(c, p))
    return pd.DataFrame(rows), open_, settle, peak


def summary(x: list[float]) -> str:
    a = np.array([v for v in x if v is not None])
    return (f"average {a.mean():+.0f}%, median {np.median(a):+.0f}%, won {np.mean(a > 0):.0%}, "
            f"worst {a.min():+.0f}%")


def part1(n: int = 300, seed: int = 16) -> None:
    rng = np.random.default_rng(seed)
    close = 25_000.0
    from_open, naive, over = [], [], []
    for _ in range(n):
        chain, fut_open, settle, peak = expiry_day(close, rng)
        strikes = sorted(chain.strike.unique())
        k_open = atm_from_futures_open(fut_open, strikes)
        from_open.append(daily_straddle(chain, k_open, settle))
        legs = chain[chain.strike == k_open].set_index("opt_type")
        naive.append(straddle_pnl_pct(legs.loc["CE", "open"], legs.loc["PE", "open"],
                                      legs.loc["CE", "close"], legs.loc["PE", "close"]))
        over.append(100 * (legs.high.sum() / peak - 1))
        close = settle
    print(f"1. {n} synthetic expiry days, short ATM straddle, % of premium")
    print(f"   read with leg_exit_price: {summary(from_open)}")
    print(f"   overwritten closes read as prices: average {np.mean(naive):+,.0f}%")
    print(f"   sum of the legs' daily highs overstates the straddle's real peak by "
          f"{np.mean(over):.0f}% on average (median {np.median(over):.0f}%)")


class SimBroker:
    """Quotes the ATM call and put from a synthetic path with a varying spread;
    one read in fifty fails, as real reads sometimes do."""

    def __init__(self, rng, k: float, path: np.ndarray):
        self.rng, self.k, self.path, self.i = rng, k, path, 0

    def get_quote(self, symbol: str, exchange: str):
        if self.rng.random() < 0.02:
            return None
        s, t = self.path[self.i], (375 - self.i / 12) / YEAR_MIN
        mid = bs_price(s, self.k, t, VOL * IV_MULT, symbol.endswith("CE"))
        half = self.rng.uniform(0.1, 0.6)
        return dict(bid=round(max(mid - half, 0.05), 2), ask=round(mid + half, 2), bid_qty=1500, ask_qty=1500,
                    ltp=round(mid, 2))


def part2(seed: int = 6) -> None:
    rng = np.random.default_rng(seed)
    path = 25_000 * np.exp(np.cumsum(rng.normal(0, VOL / np.sqrt(YEAR_MIN * 12), 4_500)))
    broker = SimBroker(rng, 25_000.0, path)
    t0, state = datetime(2026, 1, 13, 9, 15), {"n": 0}

    def clock():
        return t0 + timedelta(seconds=5 * broker.i)

    def until():                                   # one call per recording loop
        if state["n"] >= 4_500:
            return True
        broker.i, state["n"] = state["n"], state["n"] + 1
        return False

    with tempfile.TemporaryDirectory() as d:
        out = str(Path(d) / "2026-01-13.csv")
        record_quotes(broker, [("NIFTY25000CE", "NFO"), ("NIFTY25000PE", "NFO")], out, every_s=5,
                      until=until, clock=clock, sleep=lambda s: None)
        rec = pd.read_csv(out)
    both = legs_on_one_clock(rec, "NIFTY25000CE", "NIFTY25000PE", max_stale="5s")
    print(f"\n2. Recorder: {len(rec):,} rows, {rec.bid.isna().sum()} failed reads recorded as gaps; "
          f"straddle value unknown at {both.straddle.isna().sum()} of {len(both):,} timestamps")
    print(spread_stats(rec).round(2).to_string())
    print("   a backtest charging 2 points of spread per leg would overstate the cost "
          f"{2 / spread_stats(rec)['median'].mean():.1f}-fold")


if __name__ == "__main__":
    part1()
    part2()
