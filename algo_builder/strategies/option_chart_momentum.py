"""
Book 2, Chapter 14 — Working with option contracts in code.

Shows: resolving the nearest expiry and the ATM contract from the instrument
       master (never from weekday or strike arithmetic — expiry weekdays have
       changed), following the ATM strike without flapping, indicators computed
       on the OPTION PREMIUM series (RSI, VWAP, Bollinger) with an explicit
       warm-up, an ADX "trending but not exhausted" band on the underlying, a VIX
       floor, and premium, momentum-loss and time exits.

The master is the broker's file NORMALISED by your adapter to the columns:
  underlying, kind ("CE"/"PE"/"FUT"), expiry (date), strike, symbol, lot_size.

Thresholds are ILLUSTRATIVE. Costs are not modelled here; deduct them.
"""
from __future__ import annotations

from datetime import date, time as dtime

import numpy as np
import pandas as pd

from algo_builder.common.contract import Contract

P = dict(rsi_entry=60, rsi_exit=50, adx_lo=20, adx_hi=40, vix_min=12,   # ILLUSTRATIVE
         bb_n=20, bb_k=2.0, premium_sl_pct=25, min_bars=20)


def nearest_expiry(master: pd.DataFrame, underlying: str, today: date) -> date:
    """The first listed option expiry on or after `today`. Read, don't compute."""
    e = master[(master.underlying == underlying) & master.kind.isin(["CE", "PE"])].expiry
    later = sorted(d for d in e.unique() if d >= today)
    if not later:
        raise LookupError(f"no {underlying} option expiry on or after {today} in the master")
    return later[0]


def atm_contract(master: pd.DataFrame, underlying: str, expiry: date, spot: float,
                 kind: str, near: float | None = None) -> Contract:
    """The listed strike closest to `spot` (or to `near`, if given) for this expiry,
    with its symbol and lot size as the master states them."""
    rows = master[(master.underlying == underlying) & (master.expiry == expiry) & (master.kind == kind)]
    if rows.empty:
        raise LookupError(f"no {underlying} {kind} listed for {expiry}")
    r = rows.iloc[int(np.argmin(np.abs(rows.strike.to_numpy(float) - (spot if near is None else near))))]
    return Contract(r.symbol, underlying, kind, expiry, int(r.lot_size), float(r.strike))


class AtmTracker:
    """Follows the ATM strike as spot moves, but only switches once spot is closer
    to another listed strike by more than `hysteresis` points, so a spot sitting
    near the midpoint between two strikes does not flip the contract every bar."""

    def __init__(self, hysteresis: float):
        self.hysteresis, self.strike = hysteresis, None

    def update(self, spot: float, strikes: np.ndarray) -> float:
        best = float(strikes[np.argmin(np.abs(strikes - spot))])
        if self.strike is None or abs(spot - self.strike) - abs(spot - best) > self.hysteresis:
            self.strike = best
        return self.strike


def wilder_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def premium_features(c: pd.DataFrame, rsi_fn=wilder_rsi) -> pd.DataFrame:
    """c: intraday candles of ONE option contract (open/high/low/close/volume).
    Warm-up: a new ATM strike starts with no history — fetch prior sessions of THAT
    contract, or wait. `ready` is False until there are enough bars to trust."""
    f = pd.DataFrame(index=c.index)
    f["rsi"] = rsi_fn(c.close)
    tp = (c.high + c.low + c.close) / 3
    day = c.index.normalize()
    f["vwap"] = (tp * c.volume).groupby(day).cumsum() / c.volume.groupby(day).cumsum()
    mid = c.close.rolling(P["bb_n"]).mean()
    f["bb_up"] = mid + P["bb_k"] * c.close.rolling(P["bb_n"]).std()
    f["close"] = c.close
    f["ready"] = np.arange(len(c)) >= P["min_bars"] - 1
    return f


def entry_ok(f: pd.Series, underlying_adx: float, vix: float) -> bool:
    return (bool(f.ready) and vix >= P["vix_min"]
            and P["adx_lo"] <= underlying_adx <= P["adx_hi"]
            and f.rsi >= P["rsi_entry"] and f.close > f.vwap)


def exit_reason(f: pd.Series, entry_premium: float, now: pd.Timestamp, time_exit: dtime) -> str | None:
    if f.close <= entry_premium * (1 - P["premium_sl_pct"] / 100):
        return "premium stop"
    if now.time() >= time_exit:
        return "time exit"
    if f.rsi < P["rsi_exit"]:
        return "momentum faded"
    if f.close < f.vwap:
        return "below premium VWAP"
    return None
