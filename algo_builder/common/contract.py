"""
Book 2, Chapter 1 — a contract and its arithmetic (Listing 1.2). Illustrative.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Contract:
    symbol: str
    underlying: str
    kind: str                 # "FUT", "CE" or "PE"
    expiry: date
    lot_size: int             # read from the instrument master for THIS date
    strike: float | None = None
    cash_settled: bool = True

    def rupees(self, points: float, lots: int = 1) -> float:
        return points * self.lot_size * lots

    def intrinsic(self, underlying_price: float) -> float:
        if self.kind == "CE":
            return max(0.0, underlying_price - self.strike)
        if self.kind == "PE":
            return max(0.0, self.strike - underlying_price)
        raise ValueError("intrinsic value applies to options only")
