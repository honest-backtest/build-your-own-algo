"""
Chapter 25 — Post-mortems from the research lab.

  1. Turnover: what replacing a share of the book every day costs, against a
     gross edge of 0.06% a day (Book 1's illustrative figure).
  2. A pair on delivery: four orders, each paying the transaction tax.
  3. An event study on SYNTHETIC shares that move one for one with the market,
     whose "insiders" buy after falls and have
     no knowledge at all: raw, against the same share's random entries, and
     against the market over the same dates.
  4. Both sides of a SYNTHETIC expiry straddle.
  5. A SYNTHETIC weekly covered call, and a volatility forecast used to skip weeks.

Every number here is SYNTHETIC or ILLUSTRATIVE.

    python chapters/ch25/run.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from algo_builder.infra.cost_model import EQ_DELIVERY, EQ_INTRADAY
from algo_builder.lab.post_mortems import (both_sides, event_study, pair_round_trip_pct, skip_rule_effect,
                                           turnover_drag)


def part1() -> None:
    gross = 0.0006
    print("1. Gross edge 0.060% a day. Replacing ... of the book daily:")
    for t in (0.1, 0.25, 0.5, 0.75, 1.0):
        c = turnover_drag(t, EQ_INTRADAY, half_spread=0.00025)
        print(f"   {t:>4.0%}: cost {100 * c:.3f}% a day, net {100 * (gross - c):+.3f}%")


def part2() -> None:
    p = pair_round_trip_pct(500_000, EQ_DELIVERY)
    print(f"2. Pair on delivery: {100 * p:.2f}% of one leg per round trip; "
          f"held about five days, {100 * p * 50:.0f}% of one leg a year.")


def market_path(n: int, rng) -> np.ndarray:
    r = rng.normal(0.0004, 0.009, n)
    for start in rng.choice(np.arange(100, n - 120), 6, replace=False):   # six sharp falls, then rebounds
        r[start:start + 25] -= 0.008
        r[start + 25:start + 85] += 0.003
    return 100 * np.exp(np.cumsum(r))


def part3(rng) -> None:
    days = pd.bdate_range("2014-01-01", periods=3000)
    mkt = pd.Series(market_path(len(days), rng), index=days)
    mret = np.log(mkt).diff().fillna(0).to_numpy()
    k = 150
    idio = rng.normal(0, 0.016, (len(days), k))
    closes = pd.DataFrame(100 * np.exp(np.cumsum(mret[:, None] + idio, axis=0)),
                          index=days, columns=[f"S{i:03d}" for i in range(k)])
    fall = closes / closes.shift(40) - 1
    mfall = (mkt / mkt.shift(40) - 1).to_numpy()[:, None]
    p = np.where((fall.to_numpy() < -0.15) & (mfall < -0.08), 0.03, 0.0004)   # buy after falls, no knowledge
    hits = np.argwhere(rng.random(p.shape) < p)
    events = pd.DataFrame({"date": days[hits[:, 0]], "symbol": closes.columns[hits[:, 1]]})
    ev = event_study(closes, events, horizon=60, market=mkt)
    print(f"3. {len(ev)} insider purchases, 60-day returns (no knowledge built in):")
    print(f"   raw {100 * ev.ret.mean():+.1f}%   vs same-share random entries "
          f"{100 * ev.vs_random.mean():+.1f}%   vs market over the same dates {100 * ev.vs_market.mean():+.1f}%")
    yr = ev.groupby(ev.date.dt.year)[["vs_random", "vs_market"]].mean()
    print(f"   years with events: {len(yr)}; negative against random entries in {(yr.vs_random < 0).sum()}, "
          f"against the market in {(yr.vs_market < 0).sum()}")


def part4(rng) -> None:
    n = 400
    premium = 1.0                                               # straddle cost, in units
    move = np.abs(rng.standard_t(3, n)) * 0.55                  # what it was worth at expiry
    gross = pd.Series(move - premium)
    b = both_sides(gross, cost_rt=0.04)
    print(f"4. Straddle on {n} expiries: bought {b['long_mean']:+.3f}, sold {b['short_mean']:+.3f} "
          f"(units of premium, after costs on each side); the sold side's worst day {(-gross).min() - 0.04:.2f}")


def part5(rng) -> None:
    n = 300
    vol = np.empty(n); vol[0] = 1.0
    for t in range(1, n):                                       # volatility clusters
        vol[t] = 0.2 + 0.8 * vol[t - 1] + abs(rng.normal(0, 0.25))
    move = rng.normal(0, vol)
    meltup = (vol < np.median(vol)) & (rng.random(n) < 0.05)    # sudden rallies from calm
    move[meltup] += rng.uniform(4, 7, meltup.sum())
    pnl = pd.Series(1.0 - np.maximum(move - 1.5, 0))            # premium, minus any breach of the call
    prev = np.r_[vol[0], vol[:-1]]
    forecast = pd.Series(0.2 + 0.8 * prev + 0.2)                # the clustering model's own forecast
    right = np.mean(np.sign(forecast - prev)[1:] == np.sign(vol - prev)[1:])
    e = skip_rule_effect(pnl, forecast > forecast.quantile(0.8))
    print(f"5. Forecast direction right {right:.0%} of weeks. Skipping its top fifth: "
          f"{e['skipped']} weeks, income given up {e['income_given_up']:.0f}, losses avoided "
          f"{e['losses_avoided']:.0f}, worst 10 weeks skipped {e['worst_skipped']}; total "
          f"{e['total_before']:.0f} -> {e['total_after']:.0f}")


if __name__ == "__main__":
    rng = np.random.default_rng(25)
    part1(); part2(); part3(rng); part4(rng); part5(rng)
