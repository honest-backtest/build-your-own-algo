"""
Chapter 29 — From backtest to paper to live.

  1. Three months of SYNTHETIC forward trades against the backtest over the same
     dates: three days the job did not run, a filter the live code did not apply,
     and ordinary slippage. Reconciled trade by trade, and the gap attributed.
  2. Realised slippage against the backtest's assumption.
  3. Gates registered for each stage, and one stage at a time.

    python chapters/ch29/run.py
"""
from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from algo_builder.live.paper_gates import Gates, decide, register
from algo_builder.live.reconcile import gap_attribution, next_stage, reconcile_trades, slippage_check
from algo_builder.strategies.buy_on_gap import log_row


def trades(rng) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    days = pd.bdate_range("2026-07-01", "2026-09-30")
    rows = []
    for d in days:
        for s in rng.choice([f"S{i:02d}" for i in range(25)], rng.poisson(1.6), replace=False):
            px = float(rng.uniform(200, 2000))
            rows.append(dict(symbol=s, entry_date=d.date().isoformat(), entry_px=px,
                             exit_px=px * (1 + rng.normal(0.004, 0.025)), qty=int(20_000 // px), side=1,
                             filtered=bool(rng.random() < 0.85)))
    sig = pd.DataFrame(rows)
    bt = sig[sig.filtered].drop(columns="filtered").copy()           # the backtest applies the filter
    off = set(rng.choice(sorted(set(sig.entry_date)), 3, replace=False))
    fw = sig[~sig.entry_date.isin(off)].drop(columns="filtered").copy()   # live forgot it; 3 days missed
    fw["signal_px"] = fw.entry_px
    fw["entry_px"] = fw.entry_px * (1 + np.abs(rng.normal(0.0025, 0.002, len(fw))))
    fw["exit_px"] = fw.exit_px * (1 - np.abs(rng.normal(0.0008, 0.001, len(fw))))
    bad = ~fw.index.isin(bt.index)                                   # unfiltered signals were worse
    fw.loc[bad, "exit_px"] = fw.loc[bad, "entry_px"] * (1 + rng.normal(-0.012, 0.025, bad.sum()))
    for df in (bt, fw):
        df["pnl"] = df.side * (df.exit_px - df.entry_px) * df.qty
    log = pd.DataFrame([log_row(r.symbol, r.signal_px, r.entry_px, r.exit_px) for r in fw.itertuples()])
    return bt, fw.drop(columns="signal_px"), log


def part1(bt, fw) -> None:
    g = reconcile_trades(bt, fw)
    print(f"1. Backtest {len(bt)} trades, forward {len(fw)}: both {len(g['both'])}, "
          f"backtest only {len(g['backtest_only'])}, forward only {len(g['forward_only'])}")
    print(f"   backtest P&L {bt.pnl.sum():+,.0f}   forward P&L {fw.pnl.sum():+,.0f}")
    for k, v in gap_attribution(g).items():
        print(f"   {k:<31}{v:>+10,.0f}")


def part2(log) -> None:
    c = slippage_check(log, assumed_pct=0.10)
    print(f"2. Slippage: median {c['median_pct']:.2f}%, mean {c['mean_pct']:.2f}% against "
          f"{c['assumed_pct']:.2f}% assumed; within assumption: {c['within_assumption']}")


def part3(fw) -> None:
    with tempfile.TemporaryDirectory() as d:
        g = Gates(min_trades=60, min_mean_return=0.002, max_drawdown=0.08)
        register(str(Path(d) / "paper.json"), g, datetime(2026, 6, 30, 18, 0))
        ret = fw.pnl / (fw.entry_px * fw.qty)
        for n in (30, len(fw)):
            led = pd.DataFrame({"ret": ret.iloc[:n].to_numpy()})
            eq = 1_000_000 + fw.pnl.iloc[:n].cumsum()
            dec = decide(str(Path(d) / "paper.json"), led, eq)
            print(f"3. After {n:>3} paper trades: mean {led.ret.mean():+.2%}, gates say {dec!r} "
                  f"-> stage {next_stage('paper', dec)!r}")


if __name__ == "__main__":
    bt, fw, log = trades(np.random.default_rng(29))
    part1(bt, fw)
    part2(log)
    part3(fw)
