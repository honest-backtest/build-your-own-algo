"""
Book 2, Chapter 29 — From backtest to paper to live.

Reconciling a forward record (paper or live) with the backtest run over the same
dates, trade by trade. Every difference falls into one of three groups:

  * trades the backtest took and the forward record did not  (discipline, or a
    job that did not run);
  * trades the forward record took and the backtest did not  (the code is not
    the rules, or the rules were overridden by hand);
  * trades both took, at different prices                    (execution).

`gap_attribution` splits the difference in total P&L between those groups, so
the gap between a backtest and a live account has an explanation, not a shrug.
ILLUSTRATIVE.
"""
from __future__ import annotations

import pandas as pd

KEY = ["symbol", "entry_date"]


def reconcile_trades(backtest: pd.DataFrame, forward: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Both frames: symbol, entry_date, entry_px, exit_px, qty, side (+1 long, -1
    short), pnl (net). Matched on symbol and entry date. Returns the three groups;
    `both` carries both sets of prices and the differences."""
    for df in (backtest, forward):
        if df.duplicated(KEY).any():
            raise ValueError("two trades share a symbol and entry date; add a sequence column to KEY")
    m = backtest.merge(forward, on=KEY, how="outer", suffixes=("_bt", "_fw"), indicator=True)
    both = m[m._merge == "both"].drop(columns="_merge").copy()
    side, qty = both.side_fw, both.qty_fw
    both["entry_cost"] = side * (both.entry_px_fw - both.entry_px_bt) * qty   # paid by worse entries
    both["exit_cost"] = side * (both.exit_px_bt - both.exit_px_fw) * qty      # paid by worse exits
    both["size_diff"] = both.qty_fw - both.qty_bt
    cols = lambda s: {c: c[:-3] for c in m.columns if c.endswith(s)}
    return dict(backtest_only=m[m._merge == "left_only"].rename(columns=cols("_bt"))[backtest.columns],
                forward_only=m[m._merge == "right_only"].rename(columns=cols("_fw"))[forward.columns],
                both=both)


def gap_attribution(groups: dict[str, pd.DataFrame]) -> pd.Series:
    """Backtest total minus forward total, split by cause. The parts add up to the
    gap exactly; `other` is what the three named causes do not explain (size
    differences, costs charged differently)."""
    b = groups["both"]
    bt_total = groups["backtest_only"].pnl.sum() + b.pnl_bt.sum()
    fw_total = groups["forward_only"].pnl.sum() + b.pnl_fw.sum()
    parts = pd.Series({"missed trades (backtest only)": groups["backtest_only"].pnl.sum(),
                       "extra trades (forward only)": -groups["forward_only"].pnl.sum(),
                       "worse entries": b.entry_cost.sum(),
                       "worse exits": b.exit_cost.sum()})
    parts["other"] = (bt_total - fw_total) - parts.sum()
    parts["gap (backtest - forward)"] = bt_total - fw_total
    return parts


def slippage_check(log: pd.DataFrame, assumed_pct: float) -> dict:
    """`log` holds Chapter 22's log_row records. Compares the realised signal-to-fill
    slippage with what the backtest assumed. Unfilled orders are counted, not
    dropped: an order that never filled is a trade the backtest kept."""
    filled = log.dropna(subset=["fill_px"])
    s = filled.slip_pct
    return dict(orders=len(log), unfilled=len(log) - len(filled),
                median_pct=float(s.median()), mean_pct=float(s.mean()),
                assumed_pct=assumed_pct, within_assumption=bool(s.median() <= assumed_pct))


STAGES = ["paper", "live_minimum", "live_quarter", "live_full"]


def next_stage(stage: str, decision: str) -> str:
    """Move one stage at a time, on the decision of gates registered for that stage
    (Chapter 20). 'go' promotes; 'keep testing' stays; 'no-go' and 'tampered' stop."""
    if decision == "go":
        return STAGES[min(STAGES.index(stage) + 1, len(STAGES) - 1)]
    if decision == "keep testing":
        return stage
    return "stopped"
