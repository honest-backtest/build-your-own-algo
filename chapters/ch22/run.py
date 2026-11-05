"""
Chapter 22 — Timing-critical execution: buy-on-gap.

  1. The daily scan on a SYNTHETIC universe of 100 shares, 2017-2025, with a market
     factor and a handful of crash mornings each year: how many picks fall on the
     busiest days.
  2. The exit job on a bad afternoon: positions unreadable on the first run, one
     share already closed by the broker's square-off, an order that works for a
     while before it fills.
  3. The signal/fill log: what slippage does to a small edge.

Every setting is ILLUSTRATIVE.

    python chapters/ch22/run.py
"""
from __future__ import annotations

import tempfile
from datetime import date

import numpy as np
import pandas as pd

from algo_builder.common.broker_adapter import OrderStatus, PaperBroker, Position
from algo_builder.infra import state_store as ss
from algo_builder.strategies.buy_on_gap import log_row, scan


def panel(n: int = 100, seed: int = 22) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2017-01-01", "2025-12-31")
    T = len(days)
    gap = rng.normal(0, 0.004, T)
    crash = rng.random(T) < 0.012                                   # a few crash mornings a year
    gap[crash] -= rng.uniform(0.02, 0.05, crash.sum())
    cols = [f"S{i:03d}" for i in range(n)]
    beta = rng.uniform(0.7, 1.3, n)
    og = gap[:, None] * beta + rng.normal(0, 0.006, (T, n))         # overnight
    intr = -0.25 * og + rng.normal(0.0002, 0.012, (T, n))          # part of a gap reverses
    close = 100 * np.exp(np.cumsum(og + intr, axis=0))
    open_ = close / np.exp(intr)
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, (T, n))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, (T, n))))
    mk = lambda a: pd.DataFrame(a, index=days, columns=cols)
    return {"open": mk(open_), "high": mk(high), "low": mk(low), "close": mk(close)}


def part1() -> None:
    out = scan(panel())
    traded = out[out.picks > 0]
    top = traded.picks.groupby(traded.index.year).apply(lambda s: s.nlargest(5).sum() / s.sum())
    print(f"1. {int(traded.picks.sum()):,} picks on {len(traded):,} days; in a typical year, "
          f"{top.median():.0%} of the year's picks fell on its five busiest days")


class Afternoon(PaperBroker):
    """Positions unreadable for the first run's three looks; each order then works
    through one look and fills before the next."""

    def __init__(self):
        super().__init__()
        self.looks, self.reqs, self.working = 0, {}, {}
        self._pos[("A", "NSE")] = Position("A", "NSE", 1)
        self._pos[("C", "NSE")] = Position("C", "NSE", 1)           # B already squared off by the broker

    def positions(self):
        self.looks += 1
        for oid, left in list(self.working.items()):
            if left == 0:                                           # fills arrive between looks
                req = self.reqs[oid]
                self._orders[oid] = OrderStatus(oid, "COMPLETE", req.qty, 99.0)
                self._pos[(req.symbol, req.exchange)].net_qty -= req.qty
                del self.working[oid]
            else:
                self.working[oid] = left - 1
        return None if self.looks <= 3 else super().positions()

    def place_order(self, req):
        self._seq += 1
        oid = f"SIM-{self._seq}"
        self._orders[oid] = OrderStatus(oid, "OPEN")
        self.reqs[oid], self.working[oid] = req, 1
        return oid


def part2() -> None:
    with tempfile.TemporaryDirectory() as d:
        ss.STATE_DIR = d
        today = date(2026, 11, 3)
        ss.save_state("gap", {today.isoformat(): dict(status="open", positions=[
            dict(symbol=s, exchange="NSE", qty=1) for s in ("A", "B", "C")])})
        b = Afternoon()
        print("\n2. Exit job, run every few minutes:")
        for k in range(5):
            print(f"   run {k + 1}: {ss.exit_tracked_longs(b, 'gap', today, live=True)}")


def part3(seed: int = 3) -> None:
    rng = np.random.default_rng(seed)
    rows = [log_row(f"S{i}", 100.0, 100.0 * (1 + rng.normal(0.0034, 0.004)), None) for i in range(30)]
    slip = pd.DataFrame(rows).slip_pct
    print(f"\n3. Signal/fill log, 30 fills: mean slippage {slip.mean():.2f}%, median {slip.median():.2f}%, "
          f"worst {slip.max():.2f}%")


if __name__ == "__main__":
    part1()
    part2()
    part3()
