"""
Book 2, Chapter 27 — Rails and kill switches that never block exits.

Two layers:

  1. File-based switches a person can create from a phone or a remote shell,
     for all systems or for one:
       HALT      -> block NEW entries; exits still run.
       HALT_ALL  -> freeze everything (only when the code itself is suspect).
       KILL      -> written by the rails on a loss breach; blocks entries.
     A switch whose state cannot be read counts as ON.
  2. Rails every live order must pass: per-sleeve caps on order size, orders a
     day, notional and price distance, checked against a validated table; and a
     daily-loss check on marked-to-market P&L that trips KILL. Configuration may
     only tighten the absolute caps; only a person deletes a tripped kill file.

Exits are never stopped by a switch or a cap. They are checked only so that an
exit can never be larger than the position it closes.
Cap values below are PLACEHOLDERS; size them to your own account.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger("rails")

CONTROL_DIR = Path(os.environ.get("ALGO_CONTROL_DIR", "./control"))
HALT = CONTROL_DIR / "HALT"
HALT_ALL = CONTROL_DIR / "HALT_ALL"
KILL = CONTROL_DIR / "KILL"            # tripped automatically by the rails


def switches(system: Optional[str] = None) -> dict[str, list[Path]]:
    """The files that control one system: the global switch and its own."""
    own = (lambda n: [CONTROL_DIR / system / n]) if system else (lambda n: [])
    return {"HALT": [HALT] + own("HALT"), "HALT_ALL": [HALT_ALL] + own("HALT_ALL"),
            "KILL": [KILL] + own("KILL")}


def _on(path: Path) -> bool:
    """A switch that cannot be read is treated as ON."""
    try:
        return path.exists()
    except OSError as e:
        log.error("cannot read switch %s (%s): treating it as ON", path, e)
        return True


def job_gate(is_exit_job: bool, system: Optional[str] = None) -> Optional[str]:
    """A reason to skip this job run, or None to proceed."""
    sw = switches(system)
    if any(map(_on, sw["HALT_ALL"])):
        return "HALT_ALL present: everything frozen; open positions need a person"
    if not is_exit_job and any(map(_on, sw["HALT"] + sw["KILL"])):
        return "HALT/KILL present: new entries blocked (exits still run)"
    return None


ABS_CAPS = {                            # PLACEHOLDER values — set for your account
    "sleeve_a": dict(max_qty_per_order=1, max_orders_per_day=10, max_notional_per_order=10_000,
                     max_price_dev_pct=5.0, max_daily_loss=1_000),
    "sleeve_b": dict(max_lots_per_order=1, max_orders_per_day=4, max_price_dev_pct=20.0,
                     max_daily_loss=5_000),
    "_total": dict(max_daily_loss=5_000),
}
CAP_KEYS = {"max_qty_per_order", "max_lots_per_order", "max_orders_per_day",
            "max_notional_per_order", "max_price_dev_pct", "max_daily_loss"}


def validate_caps(caps: dict) -> None:
    """Refuse to start with a table that would quietly cap nothing: every sleeve
    needs a size cap, an order-count cap and a daily-loss cap, and a misspelt key
    is an error, not an ignored setting."""
    for sleeve, c in caps.items():
        unknown = set(c) - CAP_KEYS
        if unknown:
            raise ValueError(f"{sleeve}: unknown cap {sorted(unknown)}")
        if "max_daily_loss" not in c:
            raise ValueError(f"{sleeve}: no max_daily_loss")
        if sleeve.startswith("_"):
            continue
        if not {"max_qty_per_order", "max_lots_per_order"} & set(c) or "max_orders_per_day" not in c:
            raise ValueError(f"{sleeve}: needs a size cap and max_orders_per_day")


def effective_caps(absolute: dict, configured: dict) -> dict:
    """Configuration may tighten a cap, never loosen it, and never remove it."""
    out = {}
    for sleeve, caps in absolute.items():
        mine = configured.get(sleeve, {})
        out[sleeve] = {k: min(v, mine.get(k, v)) for k, v in caps.items()}
        for k in set(mine) - set(caps):
            log.warning("%s: configured %s has no absolute cap; ignored", sleeve, k)
    return out


def check_exit(qty: int, held: Optional[int]) -> tuple[bool, str]:
    """Exits pass every switch and cap. The only check: never larger than the
    position, so an exit cannot open a position the other way."""
    if held is None:
        return False, "position unknown: send nothing, retry"
    if qty > abs(held):
        return False, f"exit {qty} larger than position {abs(held)}"
    return True, "exit"


def check_order(ledger, sleeve: str, qty: int, price: Optional[float] = None,
                ltp: Optional[float] = None, lot_size: Optional[int] = None,
                caps: dict = ABS_CAPS, system: Optional[str] = None) -> tuple[bool, str]:
    """Gate one ENTRY order. A cap that cannot be checked refuses the order.
    `ledger` needs count_orders_today(sleeve)."""
    sw = switches(system)
    if any(map(_on, sw["HALT_ALL"] + sw["HALT"] + sw["KILL"] + [CONTROL_DIR / f"KILL_{sleeve}"])):
        return False, "kill switch on"
    c = caps.get(sleeve)
    if c is None:
        return False, f"unknown sleeve {sleeve!r}: no caps defined, refusing"
    if qty <= 0 or qty > c.get("max_qty_per_order", qty):
        return False, f"qty {qty} outside cap"
    if "max_lots_per_order" in c:
        if not lot_size:
            return False, "lot size unknown"
        if qty % lot_size or qty // lot_size > c["max_lots_per_order"]:
            return False, f"{qty} is not whole lots of {lot_size}, at most {c['max_lots_per_order']}"
    if ledger.count_orders_today(sleeve) >= c["max_orders_per_day"]:
        return False, "order-count cap reached"
    if "max_notional_per_order" in c or "max_price_dev_pct" in c:
        px = price or ltp
        if px is None:
            return False, "price unknown"
        if qty * px > c.get("max_notional_per_order", float("inf")):
            return False, f"notional {qty * px:,.0f} over cap"
        if "max_price_dev_pct" in c:
            if ltp is None:
                return False, "last price unknown"
            if abs(px / ltp - 1) * 100 > c["max_price_dev_pct"]:
                return False, f"price {px} too far from last {ltp}"
    return True, "ok"


def trip_kill(reason: str, now: datetime, path: Optional[Path] = None) -> None:
    """Write (or add to) a kill file, the global one by default. Only a person deletes it."""
    path = path or KILL
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(f"{now:%Y-%m-%dT%H:%M:%S}  {reason}\nDelete this file to re-enable entries.\n")


def check_daily_loss(pnl_by_sleeve: dict[str, float], now: datetime,
                     caps: dict = ABS_CAPS) -> list[str]:
    """Today's P&L per sleeve, MARKED TO MARKET (realised plus open), checked on
    every run, not once at the close. A sleeve's breach trips its own kill file;
    a total breach trips the global one. Returns the breaches."""
    breaches = []
    for sleeve, c in caps.items():
        if sleeve.startswith("_"):
            continue
        pnl = pnl_by_sleeve.get(sleeve, 0.0)
        if pnl < -c["max_daily_loss"]:
            breaches.append(f"{sleeve}: {pnl:+,.0f}")
            if not _on(CONTROL_DIR / f"KILL_{sleeve}"):
                trip_kill(breaches[-1], now, CONTROL_DIR / f"KILL_{sleeve}")
    total = sum(pnl_by_sleeve.get(s, 0.0) for s in caps if not s.startswith("_"))
    if total < -caps["_total"]["max_daily_loss"]:
        breaches.append(f"TOTAL: {total:+,.0f}")
        if not _on(KILL):
            trip_kill(breaches[-1], now)
    return breaches
