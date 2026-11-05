"""
Chapter 20 — Breakout pipelines: VCP and SEPA.

  1. Drawdown two ways. A SYNTHETIC long-only book of ten shares, mostly invested,
     through one sharp fall and recovery: closed-trade equity against daily
     marked-to-market equity.
  2. The violations engine. 2,000 synthetic breakouts, most of which fail; the same
     trades exited by the maximum stop alone, and by violations first.
  3. Paper go/no-go gates: registered, judged, and a quiet edit detected.

Every setting is ILLUSTRATIVE.

    python chapters/ch20/run.py
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from algo_builder.backtest.metrics import max_drawdown, realised_equity
from algo_builder.live.paper_gates import Gates, decide, register
from algo_builder.strategies.sepa_template_violations import TradeTracker, violations
from algo_builder.strategies.vcp_breakout import mtm_equity

STOP = 0.07                                                       # ILLUSTRATIVE maximum stop


def part1(seed: int = 20) -> None:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2019-06-03", "2021-06-30")
    mkt = rng.normal(0.0006, 0.009, len(days))
    crash = (days >= "2020-02-20") & (days < "2020-03-24")
    mkt[crash] -= 0.012
    mkt[(days >= "2020-03-24") & (days < "2020-06-30")] += 0.006
    closes = pd.DataFrame({f"S{i}": 100 * np.exp(np.cumsum(mkt + rng.normal(0, 0.012, len(days))))
                           for i in range(10)}, index=days)
    rows = []
    for s in closes:                                              # hold each share in long blocks
        i = 0
        while i < len(days) - 5:
            j = min(i + int(rng.integers(150, 300)), len(days) - 1)
            px = closes[s]
            rows.append(dict(symbol=s, entry_date=days[i], exit_date=days[j], shares=1_000 / px.iloc[i],
                             entry_px=px.iloc[i], exit_px=px.iloc[j]))
            i = j + int(rng.integers(1, 10))
    t = pd.DataFrame(rows)
    t["pnl"] = t.shares * (t.exit_px - t.entry_px)
    closed = realised_equity(t.assign(exit_ts=t.exit_date), capital=10_000)
    mtm = mtm_equity(t, closes, capital=10_000)
    print(f"1. Ten-share book, {len(t)} trades, mostly invested through a crash (synthetic)")
    print(f"   max drawdown on closed-trade equity:        {max_drawdown(closed):6.1%}")
    print(f"   max drawdown on marked-to-market equity:    {max_drawdown(mtm):6.1%}")


def breakout_path(rng, fails: bool) -> pd.DataFrame:
    n = 30
    drift = -0.009 if fails else 0.006
    c = 100 * np.exp(np.cumsum(rng.normal(drift, 0.015, n)))
    o = np.r_[100.0, c[:-1]]
    h, lo = np.maximum(o, c) * 1.006, np.minimum(o, c) * 0.994
    vol = rng.uniform(0.8, 1.2, n) * (np.where(c < 100, 1.8, 1.0) if fails else 1.0)
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": vol})


def part2(seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    res = {"stop only": [], "violations first": []}
    for _ in range(2_000):
        fails = rng.random() < 0.65
        b = breakout_path(rng, fails)
        ratio = rng.uniform(0.9, 1.6) if fails else rng.uniform(1.3, 2.5)
        ma20 = b.close.rolling(5, min_periods=1).mean()            # a short average, for a 30-bar toy
        for mode in res:
            tr, out = TradeTracker(100.0, ratio), None
            for k, r in enumerate(b.itertuples()):
                if r.low <= 100 * (1 - STOP):
                    out = min(r.open, 100 * (1 - STOP)); break
                tr.update(r.open, r.high, r.low, r.close, r.volume, 1.0)
                if mode == "violations first" and violations(tr, r.close, ma20.iloc[k], held=k + 1):
                    out = r.close; break
            res[mode].append((fails, (out if out is not None else b.close.iloc[-1]) / 100 - 1))
    print("\n2. 2,000 synthetic breakouts, 65% failing")
    for mode, v in res.items():
        a = pd.DataFrame(v, columns=["fails", "ret"])
        print(f"   {mode:>16}: average failed trade {100 * a[a.fails].ret.mean():+.1f}%, "
              f"average good trade {100 * a[~a.fails].ret.mean():+.1f}%, all trades {100 * a.ret.mean():+.2f}%")


def part3() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "gates.json")
        register(path, Gates(min_trades=30, min_mean_return=0.005, max_drawdown=0.15),
                 now=datetime(2026, 7, 1, 18, 0))
        led = pd.DataFrame({"ret": [0.02, -0.01, 0.03] * 4})
        eq = pd.Series([100, 102, 101, 104, 103.0])
        print("\n3. Gates:", decide(path, led, eq), "after 12 trades;",
              decide(path, pd.DataFrame({"ret": [0.02, -0.01, 0.03] * 12}), eq), "after 36;", end=" ")
        rec = json.loads(Path(path).read_text())
        rec["gates"]["max_drawdown"] = 0.40                       # a quiet edit, months later
        Path(path).write_text(json.dumps(rec))
        print("after a quiet edit:", decide(path, led, eq))


if __name__ == "__main__":
    part1()
    part2()
    part3()
