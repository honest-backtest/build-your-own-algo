"""
Book 2, Chapter 5 — sizing that cannot run away. Illustrative.

A position-sizing bug that compounds without limit can turn an ordinary strategy
into a fortune on paper. Two defences: size from the stop with a hard cap on
notional exposure, and check the whole equity curve for impossible states.
"""
from __future__ import annotations

import pandas as pd


class SizingError(ValueError):
    """Raised when a backtest reaches a state no real account could be in."""


def lots_for_risk(equity: float, risk_frac: float, entry: float, stop: float,
                  lot_size: int, max_notional_mult: float, min_lots: int = 0) -> int:
    """Lots that risk `risk_frac` of equity if the stop is hit, rounded down, never
    more than `max_notional_mult` times equity in notional value. A `min_lots` floor
    lets small accounts take a signal, but the leverage cap still wins."""
    risk_per_lot = abs(entry - stop) * lot_size
    if equity <= 0 or risk_per_lot <= 0:
        return 0
    lots = max(int(equity * risk_frac // risk_per_lot), min_lots)
    cap = int(equity * max_notional_mult // (entry * lot_size))
    return max(0, min(lots, cap))


def check_sane(equity: pd.Series, notional: pd.Series, max_notional_mult: float) -> None:
    """Fail loudly if equity ever reaches zero or exposure ever breaches the cap."""
    if (equity <= 0).any():
        raise SizingError(f"equity reached {equity.min():,.0f} on {equity.idxmin():%Y-%m-%d}")
    lev = notional / equity.reindex(notional.index).ffill()
    if (lev > max_notional_mult + 1e-9).any():
        raise SizingError(f"exposure {lev.max():.1f}x equity on {lev.idxmax():%Y-%m-%d}, "
                          f"above the {max_notional_mult}x cap")
