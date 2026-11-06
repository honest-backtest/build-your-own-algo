"""
Book 2, Chapter 6 — Gross, net and the friction sweep, in index points. Illustrative.

For futures, points are the natural unit: a round trip's cost in points does not
depend on the lot size, and a strategy's average trade in points can be compared
with it directly. Three tools: the cost of a round trip in points on a given date,
a friction sweep in points, and a one-line diagnosis.
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from algo_builder.infra.cost_model import PROFILES, round_trip_cost


def round_trip_points(level: float, lot_size: int, profile: str = "FUTURES",
                      slippage_pts: float = 1.0) -> float:
    """Charges for one lot bought and sold at `level`, at current rates, in index
    points, plus `slippage_pts` on each side."""
    value = level * lot_size
    charges = round_trip_cost(value, value, PROFILES[profile])
    return charges / lot_size + 2 * slippage_pts


def friction_sweep_points(gross_pts: pd.Series, cost_pts=(0, 2, 4, 6, 8, 10, 15, 20)) -> pd.DataFrame:
    """The same trades, net of a flat round-trip cost in points, from zero upwards."""
    rows = []
    for c in cost_pts:
        net = gross_pts - c
        wins, losses = net[net > 0].sum(), -net[net < 0].sum()
        rows.append(dict(cost_pts=c, mean_net_pts=net.mean(), total_net_pts=net.sum(),
                         profit_factor=wins / losses if losses else np.inf,
                         win_rate=(net > 0).mean()))
    return pd.DataFrame(rows)


def cost_or_signal(gross_pts: pd.Series, cost_pts: float) -> str:
    """Which problem does this strategy have, if any?"""
    gross = gross_pts.mean()
    t = gross / (gross_pts.std(ddof=1) / np.sqrt(len(gross_pts))) if len(gross_pts) > 1 else np.nan
    if gross <= 0:
        return f"signal problem: the average trade loses {gross:.1f} points before any cost"
    if gross <= cost_pts:
        return (f"cost problem: {gross:.1f} points gross against {cost_pts:.1f} points of cost "
                f"(break-even at {gross:.1f}; t = {t:.2f})")
    return f"survives costs: {gross - cost_pts:.1f} points a trade net (t of gross = {t:.2f})"
