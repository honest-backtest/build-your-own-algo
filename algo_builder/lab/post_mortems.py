"""
Book 2, Chapter 25 — Post-mortems from the research lab.

The checks that would have killed each idea sooner: what turnover costs a day,
what a two-legged delivery trade costs, an event study measured against the
same share's random entries and the market over the same dates, results stated
at several horizons, both sides of a trade, the effect of a skip rule on the strategy it serves, and
an implied-volatility percentile
computed from the past only. Uses the cost model of Chapter 5. ILLUSTRATIVE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from algo_builder.infra.cost_model import CostRates, round_trip_cost


def turnover_drag(daily_turnover: float, r: CostRates, half_spread: float,
                  capital: float = 1e7) -> float:
    """Cost per day, as a fraction of capital, of replacing `daily_turnover` of the
    book every day: one sale and one purchase of that value, charged at `r`,
    plus the half-spread paid on each. Compare it with the gross edge per day."""
    traded = daily_turnover * capital
    if traded <= 0:
        return 0.0
    return (round_trip_cost(traded, traded, r) + 2 * half_spread * traded) / capital


def pair_round_trip_pct(leg_value: float, r: CostRates) -> float:
    """Cost of opening and closing both legs of a pair, as a fraction of ONE leg's
    value. Four executed orders; on delivery, securities transaction tax is paid
    on every one of them."""
    return 2 * round_trip_cost(leg_value, leg_value, r) / leg_value


def event_study(closes: pd.DataFrame, events: pd.DataFrame, horizon: int,
                market: pd.Series, n_draws: int = 200, seed: int = 25) -> pd.DataFrame:
    """For each event (columns `date`, `symbol`), the return from the next close
    over `horizon` days, beside two fair comparisons: the mean return of the SAME
    share bought on random days and held as long (its own drift), and the market's
    return over the SAME dates (the regime the event happened in)."""
    rng = np.random.default_rng(seed)
    pos = {d: i for i, d in enumerate(closes.index)}
    m = market.reindex(closes.index).to_numpy()
    rows = []
    for d, s in zip(events.date, events.symbol):
        c = closes[s].to_numpy()
        i = pos.get(pd.Timestamp(d))
        if i is None or i + 1 + horizon >= len(c):
            continue                                       # not enough future to measure
        j = rng.integers(0, len(c) - horizon - 1, n_draws)
        rows.append(dict(date=pd.Timestamp(d), symbol=s,
                         ret=c[i + 1 + horizon] / c[i + 1] - 1,
                         same_stock_random=float(np.nanmean(c[j + horizon] / c[j] - 1)),
                         market=m[i + 1 + horizon] / m[i + 1] - 1))
    out = pd.DataFrame(rows)
    out["vs_random"] = out.ret - out.same_stock_random
    out["vs_market"] = out.ret - out.market
    return out


def by_horizon(entry_px: pd.Series, later_px: pd.DataFrame) -> pd.DataFrame:
    """The same entries judged at several horizons. `later_px` has one column per
    horizon (for example '15min', 'close'). Two intraday studies can be compared
    only at the same horizon."""
    ret = later_px.div(entry_px, axis=0) - 1
    return pd.DataFrame({"mean_pct": 100 * ret.mean(), "win_rate": (ret > 0).mean()})


def both_sides(gross: pd.Series, cost_rt: float) -> dict:
    """The same trades taken long and short, each paying its own costs. The mirror
    of a losing trade is not a winning one until it has paid for itself."""
    return dict(long_mean=float((gross - cost_rt).mean()),
                short_mean=float((-gross - cost_rt).mean()))


def skip_rule_effect(pnl: pd.Series, skip: pd.Series, worst: int = 10) -> dict:
    """What a rule that stands aside in some periods would have done to a strategy:
    the income it gave up, the losses it avoided, and how many of the strategy's
    `worst` periods it actually skipped. A forecast can be accurate and still fail
    this test; this is the test that matters."""
    skip = skip.reindex(pnl.index, fill_value=False).astype(bool)
    worst_idx = pnl.nsmallest(worst).index
    return dict(skipped=int(skip.sum()),
                income_given_up=float(pnl[skip & (pnl > 0)].sum()),
                losses_avoided=float(-pnl[skip & (pnl < 0)].sum()),
                worst_skipped=int(skip[worst_idx].sum()),
                total_before=float(pnl.sum()), total_after=float(pnl[~skip].sum()))


def iv_percentile(iv: pd.Series, window: int = 252) -> pd.Series:
    """Share of the previous `window` days on which implied volatility was lower
    than today. Today is compared with the past only; NaN until the window is full."""
    vals = iv.to_numpy(float)
    out = np.full(len(vals), np.nan)
    for t in range(window, len(vals)):
        out[t] = np.mean(vals[t - window:t] < vals[t])
    return pd.Series(out, index=iv.index)
