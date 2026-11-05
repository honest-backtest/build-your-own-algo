"""
Book 2, Chapter 26 — State, idempotency and reconciliation (the execution adapter).

  1. A tri-state answer: "resting", "terminal", or UNKNOWN (book unreadable).
     Unknown is never treated as either of the other two.
  2. Never cancel blind. If the book can't be read, keep the signal alive and retry.
     A cancel request is a request: the order may fill before it arrives, so the
     outcome is decided by the next reconcile, not by the cancel.
  3. Reconcile every poll: an order placed earlier may have filled since. Promote
     it with its REAL fill price and quantity, including part-fills of orders that
     were later cancelled.

Entry limits are "marketable with a cap": cross the spread, but never pay more
than a configured buffer beyond the reference price (no chasing).
"""
from __future__ import annotations

import logging
import math
import time
from typing import Callable, Optional

from algo_builder.common.broker_adapter import BrokerAdapter, OrderRequest, OrderStatus

log = logging.getLogger("execution")

TERMINAL = {"COMPLETE", "REJECTED", "CANCELLED"}


def marketable_limit(reference: float, side: str, max_slip_pct: float, tick: float = 0.05) -> float:
    """A limit that crosses the spread but never beyond the cap: a BUY is rounded
    DOWN to the tick and a SELL UP, so rounding can never breach the buffer."""
    if side == "BUY":
        px = math.floor(reference * (1 + max_slip_pct / 100) / tick + 1e-9) * tick
    else:
        px = math.ceil(reference * (1 - max_slip_pct / 100) / tick - 1e-9) * tick
    return max(round(px, 2), tick)


def place(broker: BrokerAdapter, req: OrderRequest) -> Optional[str]:
    oid = broker.place_order(req)
    if oid is None:
        log.error("order not accepted: %s %s x%d", req.side, req.symbol, req.qty)
    return oid


def poll_fills(broker: BrokerAdapter, order_ids: list[str], tries: int = 6, pause: float = 5.0,
               sleep: Callable[[float], None] = time.sleep) -> dict[str, OrderStatus]:
    """order_id -> latest status; UNKNOWN until the broker has answered. Stops early
    once all are terminal. `sleep` is injected so tests do not wait."""
    out = {oid: OrderStatus(oid, "UNKNOWN") for oid in order_ids}
    for i in range(tries):
        for oid in order_ids:
            st = broker.order_status(oid)
            if st is not None:
                out[oid] = st
        if all(s.status in TERMINAL for s in out.values()) or i == tries - 1:
            break
        sleep(pause)
    return out


def is_resting(broker: BrokerAdapter, order_id: str) -> Optional[bool]:
    """True = still working, False = terminal, None = UNKNOWN (do not assume)."""
    st = broker.order_status(order_id)
    if st is None or st.status == "UNKNOWN":
        return None
    return st.status == "OPEN"


def cancel_pending(broker: BrokerAdapter, trade: dict, store) -> bool:
    """Ask to cancel a still-resting entry. Returns False when nothing could be done
    (book unreadable, or the cancel refused); the caller keeps the signal active and
    tries again. Whether the order ended filled, part-filled or cancelled is left
    to `reconcile_pending`, which reads it from the broker."""
    resting = is_resting(broker, trade["order_id"])
    if resting is None:
        log.error("%s: order book unreadable: NOT cancelling blind", trade["symbol"])
        return False
    if resting and not broker.cancel_order(trade["order_id"]):
        log.error("%s: cancel refused; will retry", trade["symbol"])
        return False
    reconcile_pending(broker, store)
    return True


def reconcile_pending(broker: BrokerAdapter, store) -> None:
    """Promote late fills and part-fills; close out entries that ended unfilled.
    `store` is any object with pending_trades(), promote_to_open(id, price, qty,
    price_known) and mark_failed(id, reason)."""
    for tr in store.pending_trades():
        st = broker.order_status(tr["order_id"])
        if st is None or st.status not in TERMINAL:
            continue                              # unreadable or still working: next poll
        if st.filled_qty > 0:
            known = st.avg_price > 0
            store.promote_to_open(tr["id"], st.avg_price if known else tr["limit_price"],
                                  st.filled_qty, known)
            if not known:
                log.error("%s: filled, price not reported; recorded at the limit", tr["symbol"])
            elif st.filled_qty < tr["qty"]:
                log.warning("%s: part-filled %d of %d", tr["symbol"], st.filled_qty, tr["qty"])
        else:
            store.mark_failed(tr["id"], f"entry {st.status} without filling")
