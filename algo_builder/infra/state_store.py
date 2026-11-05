"""
Book 2, Chapter 26 — State, idempotency and reconciliation.
Also used in Chapters 17 (covered call) and 22 (buy-on-gap exits).

A scheduled job may run twice, crash half-way, or find that a human (or the
broker's auto-square-off) already closed a leg. The patterns:

  * State written atomically; a missing file is empty, a corrupt one is an error.
  * Idempotency keys by trading day and ISO week: re-running is a no-op.
  * Intent before action: an order's tag is saved BEFORE the order is sent, so a
    run that crashed in between can ask the broker instead of sending it again.
  * Before selling, read BROKER truth and sell min(tracked, held).
  * Unreadable means unknown: change nothing, try again later.
  * A day is closed only when the broker shows every tracked position flat.
  * A reconciler compares the state with the broker, and reports days left open.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import date
from typing import Callable, Optional, Protocol

from algo_builder.common.broker_adapter import BrokerAdapter, OrderRequest, OrderStatus

STATE_DIR = os.environ.get("ALGO_STATE_DIR", "./state")


class StateCorrupt(RuntimeError):
    """A state file exists but cannot be read. Never treated as empty."""


def load_state(name: str) -> dict:
    """The saved state, or {} if there is none yet. A file that exists but cannot
    be parsed raises: an empty state would look like 'no positions'."""
    path = os.path.join(STATE_DIR, f"{name}.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise StateCorrupt(f"{path}: {e}") from e


def save_state(name: str, state: dict) -> None:
    """Write to a temporary file in the same folder, flush it to disk, then rename
    it over the old file in one step. A crash leaves the old file or the new one,
    never half of either."""
    os.makedirs(STATE_DIR, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=STATE_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=1, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, os.path.join(STATE_DIR, f"{name}.json"))
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def already_done_this_week(state: dict, today: date) -> bool:
    """Idempotency key for a once-a-week entry job."""
    return state.get("last_entry_week") == list(today.isocalendar()[:2])


def make_tag(*parts) -> str:
    """A short tag that is the same every time for the same job, day, symbol and
    attempt. Brokers limit tag length, so the parts are hashed."""
    return "ab" + hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:10]


class TagLookup(Protocol):
    def orders_with_tag(self, tag: str) -> Optional[list[OrderStatus]]:
        """Orders carrying `tag`: [] if none, None if the order book is unreadable."""


def send_once(broker: BrokerAdapter, record: dict, req: OrderRequest,
              persist: Callable[[], None], key: str = "exit") -> Optional[str]:
    """Send `req` at most once for `record`, across crashes and re-runs. The tag is
    saved before the order goes out. A run that finds a tag but no order id asks
    the broker for that tag; if it cannot ask, it sends nothing."""
    oid_k, tag_k = f"{key}_oid", f"{key}_tag"
    if record.get(oid_k):
        return record[oid_k]
    if record.get(tag_k):                                  # sent before, outcome not recorded
        lookup = getattr(broker, "orders_with_tag", None)
        found = lookup(record[tag_k]) if lookup else None
        if found is None:
            return None                                    # unknown: send nothing
        if found:
            record[oid_k] = found[0].order_id
            persist()
            return record[oid_k]
    else:
        record[tag_k] = req.tag
        persist()                                          # intent on disk first
    oid = broker.place_order(req)
    if oid:
        record[oid_k] = oid
        persist()
    return oid


def broker_net_qty(broker: BrokerAdapter, retries: int = 3) -> Optional[dict[str, int]]:
    """symbol -> signed net qty, or None if positions could not be read."""
    for _ in range(retries):
        pos = broker.positions()
        if pos is not None:
            return {p.symbol: p.net_qty for p in pos}
    return None


def exit_tracked_longs(broker: BrokerAdapter, state_name: str, today: date, live: bool) -> str:
    """Sell today's tracked long positions. Safe to run repeatedly: sells only what
    the broker holds, sends each exit at most once, never stacks a second order on
    a working or filled one, and marks the day closed only when the broker shows
    every position flat."""
    st = load_state(state_name)
    day = st.get(today.isoformat())
    if not day or day.get("status") == "closed":
        return "nothing to do"                             # idempotent
    book = broker_net_qty(broker)
    if book is None:
        return "positions unreadable: nothing sent, will retry"
    remaining = 0
    for pos in day["positions"]:
        held = max(0, book.get(pos["symbol"], 0))
        pos["sell_qty"] = min(pos["qty"], held)            # never more than the broker holds
        if pos["sell_qty"] <= 0:
            continue                                       # already flat at the broker
        remaining += 1
        if not live:
            continue
        if pos.get("exit_oid"):
            o = broker.order_status(pos["exit_oid"])
            if o is None or o.status in ("OPEN", "UNKNOWN", "COMPLETE"):
                continue                                   # working, unknown, or filled: wait for the book
            pos["attempt"] = pos.get("attempt", 0) + 1     # rejected or cancelled: a new attempt
            pos.pop("exit_oid"), pos.pop("exit_tag", None)
        req = OrderRequest(pos["symbol"], pos["exchange"], "SELL", pos["sell_qty"],
                           tag=make_tag(state_name, today, pos["symbol"], pos.get("attempt", 0)))
        send_once(broker, pos, req, lambda: save_state(state_name, st))
    if live:                                               # dry runs never change status
        day["status"] = "closed" if remaining == 0 else "closing"
    save_state(state_name, st)
    return day.get("status", "dry run")


def reconcile_book(tracked: dict[str, int], broker: BrokerAdapter) -> Optional[list[str]]:
    """Every difference between what the state says is held and what the broker
    holds, as lines for an alert. [] means they agree; None means unknown."""
    book = broker_net_qty(broker)
    if book is None:
        return None
    diffs = []
    for sym in sorted(set(tracked) | {s for s, q in book.items() if q}):
        mine, theirs = tracked.get(sym, 0), book.get(sym, 0)
        if mine != theirs:
            diffs.append(f"{sym}: state {mine}, broker {theirs}")
    return diffs


def stale_days(state: dict, today: date) -> list[str]:
    """Trading days before today whose status is not 'closed'. An exit job that
    never ran leaves exactly this behind, and nothing else will say so."""
    return sorted(k for k, v in state.items()
                  if isinstance(v, dict) and "status" in v and k < today.isoformat()
                  and v["status"] != "closed")
