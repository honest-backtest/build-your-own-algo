"""Chapter 29: reconciling forward trades with the backtest."""
import pandas as pd
import pytest

from algo_builder.live.reconcile import gap_attribution, next_stage, reconcile_trades, slippage_check


def frame(rows):
    df = pd.DataFrame(rows, columns=["symbol", "entry_date", "entry_px", "exit_px", "qty", "side"])
    df["pnl"] = df.side * (df.exit_px - df.entry_px) * df.qty
    return df


def test_three_groups_and_an_exact_attribution():
    bt = frame([("A", "d1", 100, 110, 1, 1), ("B", "d1", 50, 55, 2, 1), ("C", "d2", 20, 19, 5, 1),
                ("S", "d2", 80, 70, 1, -1)])
    fw = frame([("A", "d1", 101, 109, 1, 1), ("C", "d2", 20.2, 19, 5, 1), ("D", "d3", 30, 28, 1, 1),
                ("S", "d2", 79, 71, 1, -1)])
    g = reconcile_trades(bt, fw)
    assert list(g["backtest_only"].symbol) == ["B"] and list(g["forward_only"].symbol) == ["D"]
    a = gap_attribution(g)
    assert a["missed trades (backtest only)"] == 10 and a["extra trades (forward only)"] == 2
    assert a["worse entries"] == pytest.approx(1 + 1 + 1)          # short sold 1 lower
    assert a["worse exits"] == pytest.approx(1 + 0 + 1)            # short bought back 1 higher
    assert a["other"] == pytest.approx(0) and a["gap (backtest - forward)"] == pytest.approx(bt.pnl.sum() - fw.pnl.sum())
    with pytest.raises(ValueError):
        reconcile_trades(pd.concat([bt, bt.iloc[:1]]), fw)


def test_slippage_counts_unfilled_and_stages_move_one_at_a_time():
    log = pd.DataFrame(dict(fill_px=[100.3, None, 100.1], slip_pct=[0.3, None, 0.1]))
    c = slippage_check(log, assumed_pct=0.1)
    assert c["unfilled"] == 1 and c["median_pct"] == pytest.approx(0.2) and not c["within_assumption"]
    assert next_stage("paper", "go") == "live_minimum" and next_stage("live_full", "go") == "live_full"
    assert next_stage("live_quarter", "keep testing") == "live_quarter" and next_stage("paper", "tampered") == "stopped"
