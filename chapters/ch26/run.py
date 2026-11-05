"""
Chapter 26 — State, idempotency and reconciliation.

  1. The crash window: an exit job that dies after sending each order but before
     recording it, then runs again. Recording after sending, against send_once.
  2. Cancelling resting entries when some fill, or part-fill, on the way: marking
     them failed on cancel, against leaving the outcome to the reconcile.
  3. The next morning's reconciler: a day left open and a book that disagrees.

Everything runs against SIMULATED brokers.

    python chapters/ch26/run.py
"""
from __future__ import annotations

import logging
import tempfile
from datetime import date

import numpy as np

from algo_builder.common.broker_adapter import OrderRequest, OrderStatus, PaperBroker
from algo_builder.infra import state_store as ss
from algo_builder.infra.execution import cancel_pending

SYMS = [f"S{i:02d}" for i in range(10)]


class Crash(Exception):
    pass


def holding_broker() -> PaperBroker:
    b = PaperBroker()
    for s in SYMS:
        b.set_quote(s, "NSE", 100.0)
        b.place_order(OrderRequest(s, "NSE", "BUY", 1))
    return b


def record_after_send(broker, rec, req, persist):
    """The obvious version: send, then write down the order id."""
    if rec.get("exit_oid"):
        return rec["exit_oid"]
    oid = broker.place_order(req)
    rec["exit_oid"] = oid
    persist()
    return oid


def part1() -> None:
    for name, sender in (("record after send", record_after_send), ("send_once", ss.send_once)):
        b = holding_broker()
        disk = {s: {} for s in SYMS}
        for s in SYMS:                                     # first run: dies after each send
            rec, n = dict(disk[s]), [0]

            def persist(rec=rec, s=s, n=n):
                n[0] += 1
                if n[0] == (2 if sender is ss.send_once else 1):
                    raise Crash
                disk[s] = dict(rec)
            req = OrderRequest(s, "NSE", "SELL", 1, tag=ss.make_tag("gap", "2026-11-04", s, 0))
            try:
                sender(b, rec, req, persist)
            except Crash:
                pass
        for s in SYMS:                                     # the re-run, from what was saved
            rec = dict(disk[s])
            sender(b, rec, OrderRequest(s, "NSE", "SELL", 1, tag=ss.make_tag("gap", "2026-11-04", s, 0)),
                   lambda: None)
        net = {p.symbol: p.net_qty for p in b.positions()}
        print(f"1. {name:>17}: orders sent {b._seq - len(SYMS)}, "
              f"shares short afterwards {sum(1 for q in net.values() if q < 0)}")


class Store:
    def __init__(self, trades):
        self.trades, self.open, self.failed = trades, {}, {}

    def pending_trades(self):
        return [t for t in self.trades if t["id"] not in self.open and t["id"] not in self.failed]

    def promote_to_open(self, id_, price, qty, price_known):
        self.open[id_] = qty

    def mark_failed(self, id_, reason):
        self.failed[id_] = reason


class Busy(PaperBroker):
    """Resting entries; some fill completely and some partly before a cancel lands."""
    def __init__(self, fate):
        super().__init__()
        self.fate = fate

    def cancel_order(self, oid):
        f, o = self.fate[oid], self._orders[oid]
        o.status = "COMPLETE" if f == 10 else "CANCELLED"
        o.filled_qty, o.avg_price = f, (100.0 if f else 0.0)
        return True


def part2(rng) -> None:
    fate = {f"E{i}": int(rng.choice([0, 0, 0, 10, 4])) for i in range(20)}
    held = sum(fate.values())
    for name in ("mark failed on cancel", "reconcile decides"):
        b = Busy(fate)
        trades = [dict(id=i, order_id=f"E{i}", symbol=f"T{i}", qty=10, limit_price=100.0) for i in range(20)]
        for t in trades:
            b._orders[t["order_id"]] = OrderStatus(t["order_id"], "OPEN")
        s = Store(trades)
        for t in trades:
            if name.startswith("mark"):
                b.cancel_order(t["order_id"]); s.mark_failed(t["id"], "cancelled")
            else:
                cancel_pending(b, t, s)
        print(f"2. {name:>21}: shares held at the broker {held}, recorded {sum(s.open.values())}")


def part3() -> None:
    with tempfile.TemporaryDirectory() as d:
        ss.STATE_DIR = d
        b = holding_broker()
        b.place_order(OrderRequest("S03", "NSE", "SELL", 1))      # closed by hand, not recorded
        b.set_quote("XYZ", "NSE", 50.0)
        b.place_order(OrderRequest("XYZ", "NSE", "BUY", 5))       # bought by hand, not tracked
        ss.save_state("gap", {"2026-11-03": dict(status="open", positions=[dict(symbol="S09", exchange="NSE", qty=1)]),
                              "2026-11-04": dict(status="open", positions=[dict(symbol=s, exchange="NSE", qty=1)
                                                                            for s in SYMS[:9]])})
        st = ss.load_state("gap")
        tracked = {}
        for k in ss.stale_days(st, date(2026, 11, 5)):
            for p in st[k]["positions"]:
                tracked[p["symbol"]] = tracked.get(p["symbol"], 0) + p["qty"]
        print("3. Morning check, 5 Nov:")
        print(f"   days not closed: {ss.stale_days(st, date(2026, 11, 5))}")
        for line in ss.reconcile_book(tracked, b):
            print(f"   mismatch  {line}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    part1()
    part2(np.random.default_rng(26))
    part3()
