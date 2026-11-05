"""
Book 2, Chapter 5 — A backtester that doesn't lie (the cost model).
Also used in Chapters 6 (friction sweep) and 25 (post-mortems).

Indian transaction costs are not one number. The structural differences matter
more than the exact rates:

  * equity intraday: STT on the SELL side, stamp duty on the BUY side;
  * equity delivery: STT on BOTH sides — fatal for daily-rebalanced stock pairs;
  * futures: STT on sell-side NOTIONAL;
  * options: STT on sell-side PREMIUM (tiny notional) — but the bid/ask spread
    as a % of premium dominates everything else for cheap options.

RATES BELOW ARE ILLUSTRATIVE ONLY. Statutory levies and brokerage change; look
up current values from your exchange/broker before relying on any result.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date


@dataclass(frozen=True)
class CostRates:
    brokerage_pct: float          # of turnover, per order
    brokerage_cap: float          # flat cap per order (currency)
    stt_buy_pct: float
    stt_sell_pct: float
    exchange_pct: float           # both sides
    regulator_pct: float          # both sides
    stamp_buy_pct: float
    gst_pct: float                # on brokerage + exchange + regulator


# Profiles at 2026 statutory rates (rounded); brokerage is ILLUSTRATIVE (a percentage
# with a flat cap per order). OPTIONS brokerage_pct is set so high that the flat cap
# always applies. Check current rates before relying on any result.
EQ_INTRADAY = CostRates(0.0003, 20, 0.0, 0.00025, 0.0000307, 0.000001, 0.00003, 0.18)
EQ_DELIVERY = CostRates(0.0, 0, 0.001, 0.001, 0.0000307, 0.000001, 0.00015, 0.18)
FUTURES = CostRates(0.0003, 20, 0.0, 0.0005, 0.0000183, 0.000001, 0.00002, 0.18)
OPTIONS = CostRates(1.0, 20, 0.0, 0.0015, 0.000355, 0.000001, 0.00003, 0.18)   # % of PREMIUM

# Sell-side STT on F&O changed twice; a backtest must charge the rate of its own day.
STT_SELL_HISTORY = {
    "FUTURES": [(date(2024, 10, 1), 0.000125), (date(2026, 4, 1), 0.0002), (date.max, 0.0005)],
    "OPTIONS": [(date(2024, 10, 1), 0.000625), (date(2026, 4, 1), 0.001), (date.max, 0.0015)],
}
PROFILES = {"EQ_INTRADAY": EQ_INTRADAY, "EQ_DELIVERY": EQ_DELIVERY,
            "FUTURES": FUTURES, "OPTIONS": OPTIONS}


def rates_on(profile: str, day: date) -> CostRates:
    """The profile with the sell-side STT in force on `day` (before each change date)."""
    base = PROFILES[profile]
    for until, rate in STT_SELL_HISTORY.get(profile, []):
        if day < until:
            return replace(base, stt_sell_pct=rate)
    return base


def leg_cost(turnover: float, side: str, r: CostRates) -> float:
    """Cost of one executed leg. For OPTIONS pass premium x qty as turnover."""
    side = side.upper()
    brokerage = min(r.brokerage_cap, r.brokerage_pct * turnover)
    stt = (r.stt_buy_pct if side == "BUY" else r.stt_sell_pct) * turnover
    stamp = r.stamp_buy_pct * turnover if side == "BUY" else 0.0
    exch, reg = r.exchange_pct * turnover, r.regulator_pct * turnover
    return brokerage + stt + stamp + exch + reg + r.gst_pct * (brokerage + exch + reg)


def round_trip_cost(buy_value: float, sell_value: float, r: CostRates) -> float:
    return leg_cost(buy_value, "BUY", r) + leg_cost(sell_value, "SELL", r)


def spread_cost(qty: int, bid: float, ask: float) -> float:
    """Half-spread paid on each side when crossing: for options this is usually
    the largest single cost and must be modelled from real quotes."""
    return qty * (ask - bid) / 2


def break_even_move_pct(buy_value: float, sell_value: float, r: CostRates) -> float:
    """Minimum move (fraction of entry value) needed just to cover costs.
    Compare this with your strategy's average gross move BEFORE optimising anything."""
    return round_trip_cost(buy_value, sell_value, r) / buy_value if buy_value else 0.0
