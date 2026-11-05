import json
from datetime import datetime

import pandas as pd
import pytest

from algo_builder.live.paper_gates import Gates, decide, register
from algo_builder.strategies.sepa_template_violations import TradeTracker, violations
from algo_builder.strategies.vcp_breakout import mtm_equity, review_queue


def test_mtm_equity_sees_the_open_loss():
    days = pd.bdate_range("2026-01-05", periods=5)
    closes = pd.DataFrame({"A": [100, 60, 60, 90, 110.0]}, index=days)
    t = pd.DataFrame([dict(symbol="A", entry_date=days[0], exit_date=days[4], shares=10,
                           entry_px=100.0, exit_px=110.0)])
    eq = mtm_equity(t, closes, capital=1_000)
    assert list(eq) == [1_000, 600, 600, 900, 1_100]


def test_volume_violations_count():
    t = TradeTracker(100.0, breakout_vol_ratio=1.0)                 # a weak-volume breakout
    t.update(101, 102, 97, 98, vol=3.0, avg_vol=1.0)               # heavy reversal below the pivot
    assert t.heavy_reversal and violations(t, close=98, ma20=float("nan"), held=1)
    strong = TradeTracker(100.0, breakout_vol_ratio=2.0)
    strong.update(101, 103, 100.5, 102.5, vol=1.0, avg_vol=1.0)
    assert not violations(strong, close=102.5, ma20=101.0, held=1)


def test_nothing_is_bought_on_the_screen_alone():
    approved, waiting = review_queue(["A", "B", "C"], {"A": ("approve", "clean base"), "B": ("reject", "ragged")})
    assert approved == ["A"] and waiting == ["C"]


def test_gates_are_registered_once_and_edits_are_caught(tmp_path):
    path = str(tmp_path / "g.json")
    register(path, Gates(30, 0.005, 0.15), now=datetime(2026, 7, 1, 18))
    with pytest.raises(FileExistsError):
        register(path, Gates(10, 0.0, 0.5), now=datetime(2026, 8, 1, 18))
    deep = pd.Series([100, 80.0])
    assert decide(path, pd.DataFrame({"ret": [0.01]}), deep) == "no-go"     # drawdown ends it early
    rec = json.loads(open(path).read())
    rec["gates"]["min_trades"] = 5
    open(path, "w").write(json.dumps(rec))
    assert decide(path, pd.DataFrame({"ret": [0.01]}), pd.Series([100.0])) == "tampered"
