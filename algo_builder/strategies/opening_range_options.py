"""
Book 2, Chapter 13 — One signal, two vehicles: the opening-range breakout.

Shows: range-width volatility band (against a typical daily range known before
today), one shot per side, mapping the signal to an ATM option, and pricing an
option you did not record with Black-Scholes: a realised-vol proxy, multipliers
to stress it, half the spread per side and the statutory charges on premium.

Exits are managed on the UNDERLYING's stop/target; the option P&L follows.
All parameters are ILLUSTRATIVE — the chapter's point is the plateau across
combinations, not one setting. Modelled premiums are a proxy; verify on real prices.
"""
from __future__ import annotations

from datetime import datetime, time as dtime, timedelta
from math import erf, exp, log, sqrt

import numpy as np
import pandas as pd

from algo_builder.infra.cost_model import OPTIONS, round_trip_cost
from algo_builder.robustness.gross_net import round_trip_points

PARAMS = dict(                     # ILLUSTRATIVE
    band_min_adr=0.25, band_max_adr=0.80,
    vol_mult=1.2, spread_pts=1.0, strike_step=50,
)
MINUTES_PER_YEAR = 365 * 24 * 60


def wilder_atr(daily: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder's ATR. The value on day d includes day d's own range: for a decision
    made DURING day d, use wilder_atr(daily).shift(1)."""
    pc = daily.close.shift(1)
    tr = np.maximum(daily.high - daily.low,
                    np.maximum((daily.high - pc).abs(), (daily.low - pc).abs()))
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def range_ok(or_high: float, or_low: float, adr: float, p=PARAMS) -> bool:
    """Reject both quiet days and days whose range already used up the move.
    `adr`: a typical daily range known before today (for example yesterday's ATR)."""
    w = (or_high - or_low) / adr
    return p["band_min_adr"] <= w <= p["band_max_adr"]


def _ncdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def bs_price(S: float, K: float, T_years: float, sigma: float, is_call: bool, r: float = 0.0) -> float:
    if T_years <= 0 or sigma <= 0:
        return max(0.0, S - K) if is_call else max(0.0, K - S)
    d1 = (log(S / K) + (r + sigma ** 2 / 2) * T_years) / (sigma * sqrt(T_years))
    d2 = d1 - sigma * sqrt(T_years)
    if is_call:
        return S * _ncdf(d1) - K * exp(-r * T_years) * _ncdf(d2)
    return K * exp(-r * T_years) * _ncdf(-d2) - S * _ncdf(-d1)


def realised_vol(closes: pd.Series, n: int = 20) -> float:
    """Annualised volatility of the last `n` daily log returns in `closes`. Pass
    closes up to YESTERDAY for a decision made today."""
    return float(np.log(closes).diff().tail(n).std() * sqrt(252))


def years_to_expiry(now: datetime, expiry: datetime, basis: str = "trading",
                    session=(dtime(9, 15), dtime(15, 30))) -> float:
    """Time to expiry in years. 'calendar' counts every minute; 'trading' counts only
    session minutes on weekdays (holidays ignored), a year being 252 sessions. The
    choice decides how much an option decays during a few hours' hold: stress it."""
    if basis == "calendar":
        return max((expiry - now).total_seconds() / 60, 0.0) / MINUTES_PER_YEAR
    s0, s1 = session
    per_day = (datetime.combine(now.date(), s1) - datetime.combine(now.date(), s0)).seconds / 60
    total, day = 0.0, now.date()
    while day <= expiry.date():
        if day.weekday() < 5:
            start = max(datetime.combine(day, s0), now)
            end = min(datetime.combine(day, s1), expiry)
            total += max((end - start).total_seconds() / 60, 0.0)
        day += timedelta(days=1)
    return total / (252 * per_day)


def option_trade(direction: int, S_entry: float, S_exit: float, t_entry_yrs: float,
                 t_exit_yrs: float, rv: float, p=PARAMS, qty: int = 1,
                 exit_vol_mult: float | None = None) -> dict:
    """Long ATM call for a long signal, ATM put for a short signal, priced by the
    model. Theta comes from the shorter T at exit; half the spread is paid on each
    side; charges are levied on the premium, at current rates.
    `exit_vol_mult` stresses a change of volatility while the trade is open."""
    K = round(S_entry / p["strike_step"]) * p["strike_step"]
    is_call = direction > 0
    sig_in = rv * p["vol_mult"]
    sig_out = rv * (exit_vol_mult if exit_vol_mult is not None else p["vol_mult"])
    buy = bs_price(S_entry, K, t_entry_yrs, sig_in, is_call) + p["spread_pts"] / 2
    sell = max(bs_price(S_exit, K, t_exit_yrs, sig_out, is_call) - p["spread_pts"] / 2, 0.0)
    charges = round_trip_cost(buy * qty, sell * qty, OPTIONS) / qty
    return dict(strike=K, opt="CE" if is_call else "PE", buy=buy, sell=sell,
                pts=sell - buy - charges, charges_pts=charges)


class OneShotPerSide:
    def __init__(self):
        self.used = {+1: False, -1: False}

    def allow(self, direction: int) -> bool:
        if self.used[direction]:
            return False
        self.used[direction] = True
        return True


def two_vehicles(trades: list, lot_size: int, expiry_for, rv_for, p=PARAMS,
                 option_lots: int = 2, exit_vol_mult: float | None = None,
                 basis: str = "trading") -> pd.DataFrame:
    """The same index-level trades (entry, exit, direction, times), expressed twice:
    one lot of futures, and `option_lots` lots of the ATM option, roughly the same
    starting exposure. `expiry_for(ts)` gives the option's expiry datetime;
    `rv_for(day)` the realised volatility known before that day. Both in rupees."""
    rows = []
    for t in trades:
        day = t.entry_ts.date()
        fut_pts = t.direction * (t.exit - t.entry) - round_trip_points(t.entry, lot_size)
        exp_dt = expiry_for(t.entry_ts)
        opt = option_trade(t.direction, t.entry, t.exit,
                           years_to_expiry(t.entry_ts.to_pydatetime(), exp_dt, basis),
                           years_to_expiry(t.exit_ts.to_pydatetime(), exp_dt, basis),
                           rv_for(day), p, qty=lot_size * option_lots, exit_vol_mult=exit_vol_mult)
        rows.append(dict(entry_ts=t.entry_ts, exit_ts=t.exit_ts, direction=t.direction,
                         index_pts=t.direction * (t.exit - t.entry),
                         futures=fut_pts * lot_size, options=opt["pts"] * lot_size * option_lots,
                         days_to_expiry=(exp_dt.date() - day).days))
    return pd.DataFrame(rows)
