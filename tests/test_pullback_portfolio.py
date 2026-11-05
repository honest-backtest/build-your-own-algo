import numpy as np
import pandas as pd

from algo_builder.strategies.pullback_in_uptrend import P, allocate, futures_leg_pnl, simulate


def test_allocate_survives_short_history_and_ranks_unknown_last():
    cl = pd.DataFrame({"A": [100, 110.0], "B": [100, 105.0], "C": [np.nan, 100.0]})
    assert allocate(["C", "B", "A"], set(), cl, dict(P, slots=2)) == ["A", "B"]


def test_futures_pnl_is_summed_contract_by_contract():
    idx = pd.bdate_range("2026-01-26", periods=4)
    fut = pd.DataFrame({"contract": ["JAN", "JAN", "FEB", "FEB"], "price": [100.0, 102.0, 105.0, 107.0],
                        "old_price": [np.nan, np.nan, 103.0, np.nan]}, index=idx)
    # JAN: 100 -> 102 -> 103 (closed on the roll day); FEB: 105 -> 107; one roll at 0.5
    assert futures_leg_pnl(fut, idx[0], idx[3], roll_cost_pts=0.5) == (2 + 1 + 2) - 0.5


def test_portfolio_never_exceeds_its_slots():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2024-01-01", periods=200)
    c = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, (200, 15)), axis=0)), index=days,
                     columns=[f"S{i}" for i in range(15)])
    panel = {"open": c.shift().bfill(), "high": c * 1.01, "low": c * 0.99, "close": c}
    taken, skipped, book = simulate(panel, capital=1_00_000, p=dict(P, slots=3))
    open_count = [sum((t.entry_date <= d) & (t.exit_date > d) for t in taken.itertuples()) for d in days]
    assert max(open_count) <= 3 and (book.cash >= -1e-6).all()
    assert abs(book.equity.iloc[0] - 1_00_000) / 1_00_000 < 0.05
