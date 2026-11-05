from datetime import date

from algo_builder.common.broker_adapter import OrderRequest, PaperBroker, Position
from algo_builder.common.notify import send_alert
from algo_builder.infra import kill_switches as ks
from algo_builder.infra import state_store as ss
from algo_builder.infra.execution import cancel_pending, is_resting


def _redirect(monkeypatch, tmp_path):
    monkeypatch.setattr(ss, "STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(ks, "CONTROL_DIR", tmp_path)
    for name in ("HALT", "HALT_ALL", "KILL"):
        monkeypatch.setattr(ks, name, tmp_path / name)


class Blind(PaperBroker):
    def positions(self):
        return None

    def order_status(self, order_id):
        return None


def test_covered_call_never_treats_unknown_as_flat(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    from algo_builder.strategies import covered_call_framework as cc
    monkeypatch.setattr(cc, "HALT_ALL", tmp_path / "HALT_ALL")
    ss.save_state("leg", dict(open=True, symbol="C1", exchange="NFO", qty=50, last_entry_week=[2026, 43]))
    res = cc.tick(Blind(), "leg", date(2026, 10, 23), lambda: False, lambda: True, lambda: None)
    assert "pending" in res and ss.load_state("leg")["open"] is True


def test_covered_call_buys_back_what_the_broker_shows(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    from algo_builder.strategies import covered_call_framework as cc
    monkeypatch.setattr(cc, "HALT_ALL", tmp_path / "HALT_ALL")
    b = PaperBroker()
    b.set_quote("C1", "NFO", 10.0)
    b.place_order(OrderRequest("C1", "NFO", "SELL", 50))
    ss.save_state("leg", dict(open=True, symbol="C1", exchange="NFO", qty=50, last_entry_week=[2026, 43]))
    assert cc.tick(b, "leg", date(2026, 10, 23), lambda: False, lambda: True, lambda: None) == "squared off"
    assert b.positions() == [] and ss.load_state("leg")["open"] is False


def test_exit_never_sells_more_than_the_broker_holds(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    b = PaperBroker()
    b.set_quote("S1", "NSE", 100.0)
    b.place_order(OrderRequest("S1", "NSE", "BUY", 3))            # broker holds 3, we track 5
    today = date.today().isoformat()
    ss.save_state("gap", {today: dict(status="open", positions=[dict(symbol="S1", exchange="NSE", qty=5)])})
    ss.exit_tracked_longs(b, "gap", date.today(), live=True)
    assert b.positions() == []                                     # flat, not short 2
    assert ss.exit_tracked_longs(b, "gap", date.today(), live=True) == "closed"   # confirmed on the next run


def test_kill_switch_blocks_entries_but_never_exits(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    ks.HALT.write_text("stop")

    class Ledger:
        def count_orders_today(self, sleeve):
            return 0

    assert ks.check_order(Ledger(), "sleeve_a", 1, price=100.0, ltp=100.0)[0] is False
    assert ks.check_exit(1, held=1)[0] is True                     # exits pass every switch
    assert ks.job_gate(is_exit_job=True) is None and ks.job_gate(is_exit_job=False)


def test_unreadable_order_book_is_unknown_and_never_cancelled_blind():
    assert is_resting(Blind(), "X") is None
    assert cancel_pending(Blind(), dict(order_id="X", symbol="S", id=1), store=None) is False


def test_alerts_fall_back_to_logging_and_survive_transport_errors():
    assert send_alert("hello") is False

    def broken(text, quiet):
        raise RuntimeError("network down")

    assert send_alert("hello", transport=broken) is False
    assert send_alert("hello", transport=lambda t, q: True) is True


def test_unknown_position_sleeves_are_refused(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)

    class Ledger:
        def count_orders_today(self, sleeve):
            return 0

    ok, why = ks.check_order(Ledger(), "no_such_sleeve", 1)
    assert ok is False and "unknown sleeve" in why
    assert Position("A", "NSE", 1).net_qty == 1
