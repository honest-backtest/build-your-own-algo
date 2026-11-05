"""
Book 2, Chapter 16 — Expiry day: daily data, then the recorder.

GENERIC FRAMEWORK ONLY. No rotation table, no weekday-to-index mapping, no
entry/exit times, no liquidity thresholds.

Pieces shown:
  * ATM strike from the FUTURES OPENING price (or put-call parity), not the
    prior close — after a gap the prior close leaves one leg deep in the money;
  * a liquidity gate whose thresholds you supply;
  * the daily-data straddle: enter at each leg's open, exit at its close, with the
    expiry-day overwrite handled;
  * a quote recorder that writes the bid/ask path of nearby strikes during the
    session, so stop rules and spreads can later be tested on REAL quotes;
  * tools to read the recording: both legs on one clock, and measured spreads.
"""
from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from typing import Callable, Optional

import pandas as pd

from algo_builder.common.broker_adapter import BrokerAdapter
from algo_builder.infra.option_bhavcopy_traps import leg_exit_price

FIELDS = ["ts", "symbol", "bid", "ask", "bid_qty", "ask_qty", "ltp"]


def atm_from_futures_open(fut_open: float, strikes: list[float]) -> float:
    return min(strikes, key=lambda k: abs(k - fut_open))


def atm_from_parity(quotes: dict[float, tuple[float, float]]) -> Optional[float]:
    """quotes: strike -> (call_mid, put_mid). The strike where C - P is closest to 0."""
    if not quotes:
        return None
    return min(quotes, key=lambda k: abs(quotes[k][0] - quotes[k][1]))


def liquidity_ok(quote: Optional[dict], min_bid_qty: int, max_spread_pct: float) -> bool:
    if not quote or quote.get("bid", 0) <= 0 or quote.get("ask", 0) <= 0:
        return False
    mid = (quote["bid"] + quote["ask"]) / 2
    return quote.get("bid_qty", 0) >= min_bid_qty and (quote["ask"] - quote["bid"]) / mid * 100 <= max_spread_pct


def straddle_pnl_pct(entry_ce: float, entry_pe: float, exit_ce: float, exit_pe: float) -> float:
    """Short straddle P&L as a % of premium collected — the unit used in the chapter."""
    prem = entry_ce + entry_pe
    return 100 * (prem - exit_ce - exit_pe) / prem if prem else 0.0


def daily_straddle(chain: pd.DataFrame, strike: float, underlying_settle: float) -> Optional[float]:
    """One expiry day from the exchange's daily file: sell both legs at their OPEN,
    buy back at their CLOSE, with the overwritten closes settled at intrinsic value.
    chain columns: strike, opt_type, open, close. None if either leg is missing."""
    legs = chain[chain.strike == strike].set_index("opt_type")
    if not {"CE", "PE"} <= set(legs.index):
        return None
    ce, pe = legs.loc["CE"], legs.loc["PE"]
    out_ce = leg_exit_price(float(ce.close), underlying_settle, strike, is_call=True)
    out_pe = leg_exit_price(float(pe.close), underlying_settle, strike, is_call=False)
    return straddle_pnl_pct(float(ce.open), float(pe.open), out_ce, out_pe)


def record_quotes(broker: BrokerAdapter, contracts: list[tuple[str, str]], out_csv: str,
                  every_s: float, until: Callable[[], bool],
                  clock: Callable[[], datetime] = datetime.now,
                  sleep: Callable[[float], None] = time.sleep) -> None:
    """Append one row per contract every `every_s` seconds until `until()` is True.
    A failed read is written as a row with empty prices, so a gap in the data is
    visible as a gap, not as a quiet market. One file per session; analyse later."""
    new = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(FIELDS)
        while not until():
            now = clock().isoformat(timespec="seconds")
            for sym, exch in contracts:
                q = broker.get_quote(sym, exch) or {}
                w.writerow([now, sym] + [q.get(k) for k in FIELDS[2:]])
            f.flush()
            sleep(every_s)


def legs_on_one_clock(rec: pd.DataFrame, ce: str, pe: str, max_stale: str) -> pd.DataFrame:
    """Both legs' mid prices on a common timeline. A leg's last quote is carried
    forward for at most `max_stale`; beyond that the straddle's value is unknown."""
    r = rec.assign(ts=pd.to_datetime(rec.ts), mid=(rec.bid + rec.ask) / 2).dropna(subset=["mid"])
    wide = r.pivot_table(index="ts", columns="symbol", values="mid", aggfunc="last")[[ce, pe]]
    seen = wide.notna().apply(lambda c: c.index.to_series().where(c)).ffill()
    fresh = (wide.index.to_series().values[:, None] - seen.values) <= pd.Timedelta(max_stale)
    out = wide.ffill().where(fresh)
    out["straddle"] = out[ce] + out[pe]
    return out


def spread_stats(rec: pd.DataFrame) -> pd.DataFrame:
    """Measured bid-ask spread by contract: what the backtest should charge."""
    r = rec.dropna(subset=["bid", "ask"])
    r = r[(r.bid > 0) & (r.ask > 0)].assign(spread=lambda d: d.ask - d.bid)
    g = r.groupby("symbol").spread
    return pd.DataFrame({"quotes": g.size(), "median": g.median(), "p90": g.quantile(0.9)})
