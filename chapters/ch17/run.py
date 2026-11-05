"""
Chapter 17 — The covered-call overlay: three jobs that must agree.

A SIMULATED broker that behaves like a real one at the worst moment: orders are
accepted first and filled or rejected later, and a positions read occasionally
fails. One short call is open; the exit slot arrives; the host loop calls tick()
every few minutes. Then the same morning with an exit that believes an accepted
order is a closed position.

    python chapters/ch17/run.py
"""
from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

from algo_builder.common.broker_adapter import OrderStatus, PaperBroker, Position
from algo_builder.infra import kill_switches as ks
from algo_builder.infra import state_store as ss
from algo_builder.strategies import covered_call_framework as cc


class AsyncBroker(PaperBroker):
    """Accepts every order as OPEN; on each later step, fills or rejects it per a script.
    Positions are unreadable on the steps listed in `blind`."""

    def __init__(self, outcomes: list[str], blind: set[int]):
        super().__init__()
        self.outcomes, self.blind, self.step_no, self.working = list(outcomes), blind, 0, {}
        self._pos[("NIFTY-CALL", "NFO")] = Position("NIFTY-CALL", "NFO", -75)   # the open short call

    def place_order(self, req):
        self._seq += 1
        oid = f"SIM-{self._seq}"
        self._orders[oid] = OrderStatus(oid, "OPEN")
        self.working[oid] = req
        return oid

    def step(self) -> None:
        self.step_no += 1
        for oid, req in list(self.working.items()):
            result = self.outcomes.pop(0) if self.outcomes else "fill"
            if result == "fill":
                self._orders[oid] = OrderStatus(oid, "COMPLETE", req.qty, 5.0)
                self._pos[(req.symbol, req.exchange)].net_qty += req.qty if req.side == "BUY" else -req.qty
            else:
                self._orders[oid] = OrderStatus(oid, "REJECTED", message="price band")
            del self.working[oid]

    def positions(self):
        return None if self.step_no in self.blind else super().positions()


def morning(exit_fn, label: str) -> None:
    b = AsyncBroker(outcomes=["reject", "fill"], blind={2})
    ss.save_state("leg", dict(open=True, symbol="NIFTY-CALL", exchange="NFO", qty=75,
                              last_entry_week=[2026, 45]))
    print(f"\n{label}")
    for k in range(6):
        res = exit_fn(b)
        held = {p.symbol: p.net_qty for p in (PaperBroker.positions(b) or [])}.get("NIFTY-CALL", 0)
        print(f"   call {k + 1}: {res:<45} state open={ss.load_state('leg').get('open')!s:<5} broker holds {held}")
        b.step()


def naive_exit(b) -> str:
    """Believes an accepted order is a closed position (shown for contrast only)."""
    st = ss.load_state("leg")
    if not st.get("open"):
        return "idle"
    b.place_order(cc.OrderRequest(st["symbol"], st["exchange"], "BUY", st["qty"]))
    ss.save_state("leg", dict(open=False))
    return "squared off (order accepted)"


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        ss.STATE_DIR = str(Path(d) / "state")
        cc.HALT_ALL = ks.HALT_ALL = Path(d) / "HALT_ALL"
        cc.HALT = ks.HALT = Path(d) / "HALT"
        exit_now = lambda b: cc.tick(b, "leg", date(2026, 11, 3), lambda: False, lambda: True, lambda: None)
        morning(exit_now, "1. tick(): the buy-back is rejected once, and positions are unreadable once")
        morning(naive_exit, "2. An exit that trusts the order, on the same morning")
    print("\n3. Job modes:", {"entry": "live", "roll": "live", "exit": "dry-run"},
          "-> entry allowed:", cc.entry_allowed({"entry": "live", "roll": "live", "exit": "dry-run"}))


if __name__ == "__main__":
    main()
