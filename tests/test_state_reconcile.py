"""Chapter 26: state, send-once, reconciliation and the execution adapter."""
import os
from datetime import date

import pytest

from algo_builder.common.broker_adapter import OrderRequest, OrderStatus, PaperBroker
from algo_builder.infra import state_store as ss
from algo_builder.infra.execution import cancel_pending, poll_fills, reconcile_pending


class Crash(Exception):
    pass


def test_state_files_missing_corrupt_and_atomic(monkeypatch, tmp_path):
    monkeypatch.setattr(ss, "STATE_DIR", str(tmp_path))
    assert ss.load_state("none") == {}
    (tmp_path / "bad.json").write_text("{half a fi")
    with pytest.raises(ss.StateCorrupt):
        ss.load_state("bad")
    ss.save_state("ok", {"a": 1})
    assert ss.load_state("ok") == {"a": 1} and not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]


def test_send_once_survives_a_crash_between_send_and_record():
    b = PaperBroker()
    b.set_quote("A", "NSE", 100.0)
    b.place_order(OrderRequest("A", "NSE", "BUY", 1))
    rec, disk, calls = {}, {}, []

    def crash_on_second():
        calls.append(1)
        if len(calls) == 2:
            raise Crash                                   # after the order went out
        disk.update(rec)
    req = OrderRequest("A", "NSE", "SELL", 1, tag=ss.make_tag("gap", "2026-11-04", "A", 0))
    with pytest.raises(Crash):
        ss.send_once(b, rec, req, crash_on_second)
    rec = dict(disk)                                      # the re-run starts from what was saved
    assert "exit_tag" in rec and "exit_oid" not in rec
    assert ss.send_once(b, rec, req, lambda: None)        # found by its tag, not resent
    assert b.positions() == []                            # flat, not short

    class NoLookup(PaperBroker):
        orders_with_tag = None
    nb = NoLookup()
    assert ss.send_once(nb, {"exit_tag": "abc"}, req, lambda: None) is None and nb._seq == 0


def test_exit_retries_rejections_but_not_fills_the_book_has_not_shown(monkeypatch, tmp_path):
    monkeypatch.setattr(ss, "STATE_DIR", str(tmp_path))
    today = date(2026, 11, 4)

    class Lagging(PaperBroker):
        """First exit is rejected; the second fills but positions lag one look."""
        def __init__(self):
            super().__init__()
            self.sent = []
            self.set_quote("A", "NSE", 100.0)
            super().place_order(OrderRequest("A", "NSE", "BUY", 1))

        def place_order(self, req):
            self.sent.append(req.tag)
            self._seq += 1
            oid = f"X{self._seq}"
            self._tags[oid] = req.tag
            self._orders[oid] = OrderStatus(oid, "REJECTED" if len(self.sent) == 1 else "COMPLETE", 0 if len(self.sent) == 1 else 1, 100.0)
            return oid

    b = Lagging()
    ss.save_state("g", {today.isoformat(): dict(status="open", positions=[dict(symbol="A", exchange="NSE", qty=1)])})
    assert ss.exit_tracked_longs(b, "g", today, live=True) == "closing"     # sent, rejected
    assert ss.exit_tracked_longs(b, "g", today, live=True) == "closing"     # new attempt, filled
    assert ss.exit_tracked_longs(b, "g", today, live=True) == "closing"     # book still shows 1: wait
    assert len(b.sent) == 2 and b.sent[0] != b.sent[1]
    b._pos[("A", "NSE")].net_qty = 0                                         # the book catches up
    assert ss.exit_tracked_longs(b, "g", today, live=True) == "closed"


def test_reconcile_book_and_stale_days():
    b = PaperBroker()
    b.set_quote("A", "NSE", 10.0); b.set_quote("Z", "NSE", 10.0)
    b.place_order(OrderRequest("A", "NSE", "BUY", 2)); b.place_order(OrderRequest("Z", "NSE", "BUY", 1))
    assert ss.reconcile_book({"A": 2, "B": 1}, b) == ["B: state 1, broker 0", "Z: state 0, broker 1"]
    st = {"2026-11-02": {"status": "closed"}, "2026-11-03": {"status": "open"}, "2026-11-04": {"status": "open"},
          "last_entry_week": [2026, 45]}
    assert ss.stale_days(st, date(2026, 11, 4)) == ["2026-11-03"]


class Store:
    def __init__(self, trades):
        self.trades, self.open, self.failed = trades, {}, {}

    def pending_trades(self):
        return [t for t in self.trades if t["id"] not in self.open and t["id"] not in self.failed]

    def promote_to_open(self, id_, price, qty, price_known):
        self.open[id_] = (price, qty, price_known)

    def mark_failed(self, id_, reason):
        self.failed[id_] = reason


def test_cancel_races_part_fills_and_injected_sleep():
    class Racy(PaperBroker):
        def cancel_order(self, oid):                      # fills before the cancel lands
            self._orders[oid] = OrderStatus(oid, "COMPLETE", 10, 101.0)
            return True
    b = Racy()
    b._orders["E1"] = OrderStatus("E1", "OPEN")
    s = Store([dict(id=1, order_id="E1", symbol="S", qty=10, limit_price=100.0)])
    assert cancel_pending(b, s.trades[0], s) and s.open[1] == (101.0, 10, True) and not s.failed

    b2 = PaperBroker()
    b2._orders.update(P=OrderStatus("P", "CANCELLED", 4, 0.0), N=OrderStatus("N", "REJECTED"))
    s2 = Store([dict(id=1, order_id="P", symbol="S", qty=10, limit_price=50.0),
                dict(id=2, order_id="N", symbol="T", qty=5, limit_price=20.0)])
    reconcile_pending(b2, s2)
    assert s2.open[1] == (50.0, 4, False) and "REJECTED" in s2.failed[2]

    naps = []
    out = poll_fills(PaperBroker(), ["Q"], tries=3, pause=1.0, sleep=naps.append)
    assert out["Q"].status == "UNKNOWN" and naps == [1.0, 1.0]
