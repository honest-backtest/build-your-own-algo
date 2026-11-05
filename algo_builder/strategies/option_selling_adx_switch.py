"""
Book 2, Chapter 15 — Multi-leg selling and the stop-fill problem.

Shows: a leg manager with per-leg and combined stops, bought wings, the ADX regime
       switch (trend -> close the threatened leg; range -> hold both), a
       premium-decay take-profit, and a STOP-SLIPPAGE stress term — the chapter's
       key lesson is that stop-fill quality IS the risk model.

Values are ILLUSTRATIVE. Run the backtest at several stop_slip_pts values; a
strategy that dies at a couple of points of slippage per stop is not robust.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

P = dict(leg_sl_pct=30, combined_sl_pct=20, adx_trend=25,      # ILLUSTRATIVE
         decay_tp_pct=50, exit_slip_pts=0.5, stop_slip_pts=2.0)


@dataclass
class Leg:
    symbol: str
    opt: str                # "CE" / "PE"
    entry: float
    qty: int                # > 0 sold (short), < 0 bought (a wing)
    open: bool = True
    exit: float = 0.0

    @property
    def short(self) -> bool:
        return self.qty > 0


@dataclass
class ShortPosition:
    legs: list[Leg] = field(default_factory=list)

    def credit(self) -> float:
        """Net premium collected per unit: sold legs minus bought wings."""
        return sum(leg.entry * (1 if leg.short else -1) for leg in self.legs)

    def cost_to_close(self, px: dict[str, float]) -> float:
        """Net premium per unit to close everything now (closed legs at their exits)."""
        return sum((px[leg.symbol] if leg.open else leg.exit) * (1 if leg.short else -1) for leg in self.legs)

    def mtm(self, px: dict[str, float]) -> float:
        return sum((leg.entry - (px[leg.symbol] if leg.open else leg.exit)) * leg.qty for leg in self.legs)


def close_leg(leg: Leg, price: float, stop: bool, p=P) -> None:
    """Close at the price seen, always WORSE by the ordinary slippage, and worse again
    by stop_slip_pts if a stop fired: stops fire when the market is moving fast."""
    slip = p["exit_slip_pts"] + (p["stop_slip_pts"] if stop else 0.0)
    leg.open, leg.exit = False, price + slip if leg.short else max(price - slip, 0.0)


def manage(pos: ShortPosition, px: dict[str, float], adx15: float, spot_dir: int, p=P) -> list[str]:
    """spot_dir: +1 if the underlying is rising (threatens the CE), -1 if falling, 0 if flat."""
    ev = []
    if not any(leg.open for leg in pos.legs):
        return ev                                                 # nothing left to manage
    for leg in pos.legs:                                          # per-leg premium stops (sold legs)
        if leg.open and leg.short and px[leg.symbol] >= leg.entry * (1 + p["leg_sl_pct"] / 100):
            close_leg(leg, px[leg.symbol], stop=True, p=p); ev.append(f"leg stop {leg.opt}")
    credit, cost = pos.credit(), pos.cost_to_close(px)
    if cost >= credit * (1 + p["combined_sl_pct"] / 100):         # combined stop
        for leg in pos.legs:
            if leg.open:
                close_leg(leg, px[leg.symbol], stop=True, p=p)
        ev.append("combined stop")
    elif cost <= credit * (1 - p["decay_tp_pct"] / 100):          # premium decayed
        for leg in pos.legs:
            if leg.open:
                close_leg(leg, px[leg.symbol], stop=False, p=p)
        ev.append("decay take-profit")
    elif adx15 >= p["adx_trend"] and spot_dir != 0:               # regime switch
        threatened = "CE" if spot_dir > 0 else "PE"
        for leg in pos.legs:
            if leg.open and leg.short and leg.opt == threatened:
                close_leg(leg, px[leg.symbol], stop=False, p=p); ev.append(f"trend: closed {threatened}")
    return ev


def stop_slip_sweep(run: Callable[[dict], pd.DataFrame], slips=(0, 1, 2, 3, 5), p=P) -> pd.DataFrame:
    """Re-run the same backtest with more slippage on every stop fill. `run(params)`
    returns one row per day with columns pnl_pts and stops. Report where it dies."""
    rows = []
    for s in slips:
        d = run(dict(p, stop_slip_pts=float(s)))
        rows.append(dict(stop_slip_pts=s, days=len(d), stops_per_day=d.stops.mean(),
                         mean_pnl_pts=d.pnl_pts.mean(), total_pnl_pts=d.pnl_pts.sum()))
    return pd.DataFrame(rows)
