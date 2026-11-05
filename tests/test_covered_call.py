from datetime import date
from pathlib import Path

import pandas as pd

from algo_builder.infra import state_store as ss
from algo_builder.strategies import covered_call_framework as cc


def test_square_off_waits_for_the_broker_not_the_order(monkeypatch, tmp_path):
    monkeypatch.setattr(ss, "STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(cc, "HALT_ALL", tmp_path / "HALT_ALL")
    from chapters.ch17.run import AsyncBroker
    b = AsyncBroker(outcomes=["reject", "fill"], blind=set())
    ss.save_state("leg", dict(open=True, symbol="NIFTY-CALL", exchange="NFO", qty=75, last_entry_week=[2026, 45]))
    tick = lambda: cc.tick(b, "leg", date(2026, 11, 3), lambda: False, lambda: True, lambda: None)
    assert tick() == "square-off pending: buy-back sent"
    assert tick() == "square-off pending: buy-back working"     # still working: no second order
    b.step()                                                     # rejected
    assert tick() == "square-off pending: buy-back sent"        # sent again
    b.step()                                                     # filled
    assert tick() == "squared off" and ss.load_state("leg")["open"] is False


def test_strike_guard_and_mode_helpers():
    assert cc.call_strike(25_000, 325, 1.25, 50) == 25_450
    closes = pd.Series([100, 101, 102, 101, 103.0])
    assert cc.melt_up(closes, vix=11, vix_low=12, lookback=3)
    assert not cc.melt_up(closes, vix=15, vix_low=12, lookback=3)
    assert cc.entry_allowed({"entry": "live", "roll": "live", "exit": "live"})
    assert not cc.entry_allowed({"entry": "live", "roll": "live", "exit": "dry-run"})
