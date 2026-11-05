"""Chapter 27: rails and kill switches."""
from datetime import datetime

import pytest

from algo_builder.infra import kill_switches as ks


@pytest.fixture
def ctl(monkeypatch, tmp_path):
    monkeypatch.setattr(ks, "CONTROL_DIR", tmp_path)
    for name in ("HALT", "HALT_ALL", "KILL"):
        monkeypatch.setattr(ks, name, tmp_path / name)
    return tmp_path


class Ledger:
    def __init__(self, n=0):
        self.n = n

    def count_orders_today(self, sleeve):
        return self.n


def test_switch_matrix_and_per_system_files(ctl):
    assert ks.job_gate(False) is None and ks.job_gate(True) is None
    (ctl / "expiry").mkdir()
    (ctl / "expiry" / "HALT").write_text("x")
    assert ks.job_gate(False, "expiry") and ks.job_gate(False, "covered_call") is None
    assert ks.job_gate(True, "expiry") is None                       # exits still run
    ks.HALT_ALL.write_text("x")
    assert "frozen" in ks.job_gate(True, "covered_call")


def test_caps_table_is_validated_and_only_tightened():
    with pytest.raises(ValueError):
        ks.validate_caps({"s": dict(max_qty_per_ordr=1, max_orders_per_day=1, max_daily_loss=1)})
    with pytest.raises(ValueError):
        ks.validate_caps({"s": dict(max_orders_per_day=1, max_daily_loss=1)})   # no size cap
    eff = ks.effective_caps(ks.ABS_CAPS, {"sleeve_a": dict(max_qty_per_order=50, max_orders_per_day=3)})
    assert eff["sleeve_a"]["max_qty_per_order"] == 1 and eff["sleeve_a"]["max_orders_per_day"] == 3


def test_entry_rails_refuse_what_they_cannot_check(ctl):
    ok = lambda *a, **k: ks.check_order(Ledger(), *a, **k)[0]
    assert ok("sleeve_a", 1, price=100.0, ltp=100.0)
    assert not ok("sleeve_a", 1, price=100.0)                         # last price unknown
    assert not ok("sleeve_a", 1, price=1000.0, ltp=100.0)            # decimal slip
    assert not ok("sleeve_b", 75, price=10.0, ltp=10.0)              # lot size unknown
    assert ok("sleeve_b", 75, price=10.0, ltp=10.0, lot_size=75)
    assert not ok("sleeve_b", 150, price=10.0, ltp=10.0, lot_size=75)
    assert not ks.check_order(Ledger(10), "sleeve_a", 1, price=100.0, ltp=100.0)[0]
    assert ks.check_exit(1, 1)[0] and not ks.check_exit(2, 1)[0] and not ks.check_exit(1, None)[0]


def test_daily_loss_trips_the_right_file_once(ctl):
    now = datetime(2026, 11, 4, 11, 0)
    assert ks.check_daily_loss({"sleeve_a": -500, "sleeve_b": -100}, now) == []
    assert ks.check_daily_loss({"sleeve_a": -1_500, "sleeve_b": -100}, now) == ["sleeve_a: -1,500"]
    assert (ctl / "KILL_sleeve_a").exists() and not ks.KILL.exists()
    assert not ks.check_order(Ledger(), "sleeve_a", 1, price=100.0, ltp=100.0)[0]
    assert ks.check_order(Ledger(), "sleeve_b", 75, price=10.0, ltp=10.0, lot_size=75)[0]
    ks.check_daily_loss({"sleeve_a": -1_600, "sleeve_b": -4_000}, now)
    assert ks.KILL.exists() and (ctl / "KILL_sleeve_a").read_text().count("sleeve_a") == 1
