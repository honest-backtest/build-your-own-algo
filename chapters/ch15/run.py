"""
Chapter 15 — Multi-leg selling and the stop-fill problem.

  1. A SYNTHETIC index, 300 sessions. At 09:30, sell the ATM straddle and buy wings
     500 points away; manage with per-leg and combined stops, the 15-minute ADX
     switch and a decay take-profit; close at 15:15. Option prices come from the
     model at 1.2x the market's true volatility, so the seller has an edge to
     lose (synthetic data has no option prices). Then: the same days with more
     slippage on every stop fill, and the same days with no stops at all.
  2. Expiry afternoon: how sensitive a model price is to a minute of time, a
     point of volatility and a point of quote.

Every setting is ILLUSTRATIVE.

    python chapters/ch15/run.py
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from algo_builder.strategies.expiry_mispricing import t_years
from algo_builder.strategies.opening_range_options import bs_price
from algo_builder.strategies.option_selling_adx_switch import P, Leg, ShortPosition, manage, stop_slip_sweep

TRUE_VOL, MODEL_MULT, YEAR_MIN = 0.13, 1.2, 252 * 375


def adx(h: pd.Series, lo: pd.Series, c: pd.Series, n: int = 14) -> float:
    up, dn = h.diff(), -lo.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum(h - lo, np.maximum((h - c.shift()).abs(), (lo - c.shift()).abs()))
    a = pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    pdi = 100 * pd.Series(plus).ewm(alpha=1 / n, adjust=False).mean().to_numpy() / a
    mdi = 100 * pd.Series(minus).ewm(alpha=1 / n, adjust=False).mean().to_numpy() / a
    dx = 100 * np.abs(pdi - mdi) / np.maximum(pdi + mdi, 1e-9)
    return float(pd.Series(dx).ewm(alpha=1 / n, adjust=False).mean().iloc[-1])


def sessions(n: int = 300, seed: int = 15) -> list[tuple[np.ndarray, int]]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        sd = TRUE_VOL / np.sqrt(YEAR_MIN)
        steps = rng.normal(0, sd, 375)
        if rng.random() < 0.25:
            steps += rng.choice([1, -1]) * sd * np.sqrt(375) * rng.uniform(0.5, 1.5) / 375   # a trending day
        out.append((25_000 * np.exp(np.cumsum(steps)), int(rng.integers(1, 5))))   # path, days to expiry
    return out


def run_days(days, p) -> pd.DataFrame:
    rows = []
    for path, dte in days:
        idx = pd.date_range("2026-01-05 09:15", periods=375, freq="1min")
        s = pd.Series(path, index=idx)
        k, sig = round(path[15] / 50) * 50, TRUE_VOL * MODEL_MULT
        T = lambda i: (dte * 375 + 375 - i) / YEAR_MIN
        price = {"CE": lambda S, i: bs_price(S, k, T(i), sig, True), "PE": lambda S, i: bs_price(S, k, T(i), sig, False),
                 "WCE": lambda S, i: bs_price(S, k + 500, T(i), sig, True),
                 "WPE": lambda S, i: bs_price(S, k - 500, T(i), sig, False)}
        pos = ShortPosition([Leg("CE", "CE", price["CE"](path[15], 15), 1), Leg("PE", "PE", price["PE"](path[15], 15), 1),
                             Leg("WCE", "CE", price["WCE"](path[15], 15), -1),
                             Leg("WPE", "PE", price["WPE"](path[15], 15), -1)])
        stops = 0
        for i in range(16, 360):
            px = {name: f(path[i], i) for name, f in price.items()}
            if i % 15 == 0 and any(leg.open for leg in pos.legs):
                b = s.iloc[:i + 1].resample("15min").agg(["max", "min", "last"])
                a = adx(b["max"], b["min"], b["last"]) if len(b) > 3 else 0.0
                ev = manage(pos, px, a, int(np.sign(path[i] - path[15])), p)
            else:
                ev = manage(pos, px, 0.0, 0, p)
            stops += sum("stop" in e for e in ev)
        px = {name: f(path[360], 360) for name, f in price.items()}
        rows.append(dict(pnl_pts=pos.mtm(px), stops=stops))
    return pd.DataFrame(rows)


def expiry_afternoon() -> None:
    expiry = datetime(2026, 1, 13, 15, 30)
    S, K, iv = 25_010.0, 25_000.0, 0.12
    print("\n2. Expiry afternoon: the 25,000 call with the index at 25,010, IV 12% (model)")
    for hhmm in ("13:00", "14:30", "15:00", "15:20", "15:28"):
        now = datetime.combine(expiry.date(), datetime.strptime(hhmm, "%H:%M").time())
        f = bs_price(S, K, t_years(now, expiry), iv, True)
        f_min = bs_price(S, K, t_years(now.replace(minute=now.minute + 1), expiry), iv, True)
        f_vol = bs_price(S, K, t_years(now, expiry), iv + 0.01, True)
        print(f"   {hhmm}: fair {f:6.2f}; one minute later {f_min - f:+.2f}; "
              f"one vol point higher {f_vol - f:+.2f}; a 1-point quote error is {100 / f:.1f}% of fair")


def main() -> None:
    pd.set_option("display.width", 140)
    days = sessions()
    print("1. Short straddle with wings, 300 synthetic sessions (points per unit)")
    print(stop_slip_sweep(lambda q: run_days(days, q), slips=(0, 1, 2, 3, 5)).round(2).to_string(index=False))
    with_stops = run_days(days, P)
    no_stops = run_days(days, dict(P, leg_sl_pct=1e9, combined_sl_pct=1e9, adx_trend=1e9))
    for name, d in (("with stops", with_stops), ("no stops", no_stops)):
        print(f"   {name}: mean {d.pnl_pts.mean():+.2f}, worst day {d.pnl_pts.min():+.1f}, "
              f"best day {d.pnl_pts.max():+.1f}")
    expiry_afternoon()


if __name__ == "__main__":
    main()
