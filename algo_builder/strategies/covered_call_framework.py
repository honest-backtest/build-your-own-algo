"""
Book 2, Chapter 17 — The covered-call overlay: three jobs that must agree.

GENERIC FRAMEWORK ONLY. No strike multiples, no volatility thresholds, no guard
parameters, no entry/roll/exit days or times. You supply all of those.

Reusable pieces shown:
  * expected move from the live ATM straddle (not from the VIX);
  * strike rounding to the exchange step;
  * beta-matched lot sizing;
  * an event-window check over the holding period;
  * an idempotent tick() with HALT / HALT_ALL handling, and a square-off that
    acts only on what the broker actually holds, and changes nothing when it
    cannot read the broker (unknown is never flat).
"""
from __future__ import annotations

import logging
import math
from datetime import date
from typing import Callable, Optional

import pandas as pd

from algo_builder.common.broker_adapter import BrokerAdapter, OrderRequest
from algo_builder.infra.state_store import load_state, save_state, already_done_this_week
from algo_builder.infra.kill_switches import HALT, HALT_ALL

log = logging.getLogger("covered_call")

STRADDLE_TO_SIGMA = math.sqrt(math.pi / 2)    # ATM straddle ~ 0.8 x 1-sigma move


def expected_move(atm_call: float, atm_put: float) -> float:
    """One-standard-deviation move to expiry implied by the live ATM straddle."""
    return (atm_call + atm_put) * STRADDLE_TO_SIGMA


def round_strike(price: float, step: float, up: bool = True) -> float:
    return (math.ceil if up else math.floor)(price / step) * step


def beta_matched_lots(portfolio_value: float, beta: float, index_level: float, lot: int) -> int:
    """Lots of index calls the portfolio covers, rounded DOWN: a fraction of a lot
    more than the portfolio covers is a naked short call."""
    return max(0, math.floor(portfolio_value * beta / (index_level * lot)))


def call_strike(spot: float, move: float, multiple: float, step: float) -> float:
    """`multiple` expected moves above spot, rounded UP to a listed strike step."""
    return round_strike(spot + multiple * move, step, up=True)


def melt_up(index_close: pd.Series, vix: float, vix_low: float, lookback: int) -> bool:
    """Low volatility AND the index at or above its high of the previous `lookback`
    closes. `index_close` must end at the last close known before the decision."""
    prior_high = index_close.iloc[-lookback - 1:-1].max()
    return vix <= vix_low and index_close.iloc[-1] >= prior_high


def entry_allowed(job_modes: dict[str, str]) -> bool:
    """Entry, roll and exit must run in the SAME mode. A live entry with a dry-run
    exit leaves a real short call open into settlement."""
    return len(set(job_modes.values())) == 1


def event_in_window(start: date, end: date, events: list[date]) -> bool:
    """Any calendar event (supplied by you) between entry and expiry?"""
    return any(start <= e <= end for e in events)


def tick(broker: BrokerAdapter, leg_key: str, today: date,
         is_entry_slot: Callable[[], bool], is_exit_slot: Callable[[], bool],
         choose_contract: Callable[[], Optional[tuple[str, str, int]]]) -> str:
    """Safe to call repeatedly from a host loop. The slot predicates and the
    contract chooser encapsulate YOUR schedule and strike rules. A leg is closed
    only when the broker shows no position, never because an order was accepted."""
    if HALT_ALL.exists():
        return "frozen"
    st = load_state(leg_key)

    if is_exit_slot() and st.get("open"):
        held = _held(broker, st["symbol"])
        if held is None:                            # unknown is NOT flat: change nothing
            log.error("%s: positions unreadable at square-off; state kept, will retry", leg_key)
            return "square-off pending: positions unreadable"
        if held > 0:
            log.error("%s: broker shows a LONG %d, expected short; not trading", leg_key, held)
            return "square-off pending: unexpected position"
        if held < 0:                                # still short at the broker
            pending = st.get("close_order")
            status = broker.order_status(pending) if pending else None
            if pending and (status is None or status.status == "OPEN"):
                return "square-off pending: buy-back working"
            oid = broker.place_order(OrderRequest(st["symbol"], st["exchange"], "BUY",
                                                  -held, product="CARRYFORWARD"))
            if oid is None:
                log.error("%s: buy-back not accepted; state kept, will retry", leg_key)
                return "square-off pending: order rejected"
            save_state(leg_key, dict(st, close_order=oid))
            if _held(broker, st["symbol"]) != 0:
                return "square-off pending: buy-back sent"
        save_state(leg_key, dict(open=False, last_entry_week=st.get("last_entry_week")))
        return "squared off"

    if is_entry_slot() and not st.get("open") and not already_done_this_week(st, today):
        if HALT.exists():
            return "entries halted"
        pick = choose_contract()             # returns None when a guard says skip
        if pick is None:
            save_state(leg_key, dict(open=False, last_entry_week=list(today.isocalendar()[:2])))
            return "skipped this week"
        symbol, exchange, qty = pick
        oid = broker.place_order(OrderRequest(symbol, exchange, "SELL", qty, product="CARRYFORWARD"))
        if oid:
            save_state(leg_key, dict(open=True, symbol=symbol, exchange=exchange, qty=qty,
                                     order_id=oid, last_entry_week=list(today.isocalendar()[:2])))
            return "entered"
    return "idle"


def _held(broker: BrokerAdapter, symbol: str) -> Optional[int]:
    """Net quantity the broker shows for `symbol`; None if positions cannot be read."""
    positions = broker.positions()
    if positions is None:
        return None
    return {p.symbol: p.net_qty for p in positions}.get(symbol, 0)
