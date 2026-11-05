"""
Book 2, Chapter 18 — A quarterly cycle on real prices: the DIY principal-protected note.

Shows: the quarterly cycle — accrue interest on the safe corpus, spend ONLY the
       interest on a slightly-out-of-the-money index call from REAL listed
       contracts with volume, in WHOLE LOTS, settle at intrinsic, repeat — plus
       the kill rule, checked against each quarter's own yield.

`cycles` yields Cycle records: the entry day, the expiry, a real bhavcopy slice
(strike, opt_type, close, contracts) for the entry day, the index at entry S0 and
at settlement ST (see infra/option_bhavcopy_traps.py for recovering both via
put-call parity), the safe yield known at entry, and the lot size on that day.
Values ILLUSTRATIVE.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional

import pandas as pd

P = dict(otm=0.03, slip=0.02, min_contracts=1,                  # ILLUSTRATIVE
         cost_per_lot=60.0, kill_below_yield=0.04)


@dataclass(frozen=True)
class Cycle:
    day: date
    expiry: date
    chain: pd.DataFrame
    S0: float
    ST: float
    safe_yield: float              # annual, known on `day`
    lot_size: int                  # in force on `day`


def nearest_call(chain: pd.DataFrame, target: float, min_contracts: int) -> Optional[tuple[float, float]]:
    c = chain[(chain.opt_type == "CE") & (chain.contracts >= min_contracts)]
    if c.empty:
        return None                       # listed but untraded = does not exist in practice
    r = c.iloc[(c.strike - target).abs().argsort().iloc[0]]
    return float(r.strike), float(r.close)


def run(cycles: Iterable[Cycle], capital: float, p=P) -> pd.DataFrame:
    """One row per quarter. The fund earns the safe yield; that quarter's interest,
    and nothing else, is spent on whole lots of the call; anything not spent stays
    in the fund. The capital itself is never spent."""
    fund, rows = float(capital), []
    for c in cycles:
        start, T = fund, (c.expiry - c.day).days / 365.0
        interest = start * c.safe_yield * T
        lots, K, prem, spent, payoff = 0, None, None, 0.0, 0.0
        pick = nearest_call(c.chain, c.S0 * (1 + p["otm"]), p["min_contracts"])
        if c.safe_yield >= p["kill_below_yield"] and pick is not None:
            K, prem = pick
            per_lot = prem * (1 + p["slip"]) * c.lot_size + p["cost_per_lot"]
            lots = math.floor(interest / per_lot)
            spent = lots * per_lot
            payoff = lots * c.lot_size * max(0.0, c.ST - K)
        fund += interest - spent + payoff
        rows.append(dict(day=c.day, expiry=c.expiry, lots=lots, strike=K, premium=prem,
                         interest=interest, spent=spent, payoff=payoff, fund=fund,
                         participation=lots * c.lot_size * c.S0 / start,
                         index_ret=c.ST / c.S0 - 1,
                         note="killed: yield too low" if c.safe_yield < p["kill_below_yield"]
                         else "no traded call" if pick is None else ""))
    return pd.DataFrame(rows)
