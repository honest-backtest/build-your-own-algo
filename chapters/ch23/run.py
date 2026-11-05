"""
Chapter 23 — Fundamentals pipelines and the compounder book.

  1. Floors, then ranks: 300 SYNTHETIC companies; the top 20 when floors are applied
     first, against ranking everyone and filtering afterwards; and how many names
     fall out only because a figure is missing.
  2. The base effect on Book 1's two illustrative companies.
  3. Same-quarter comparisons: a steadily growing, seasonal business, judged
     quarter on quarter and against the same quarter a year earlier.
  4. The long-term date for a holding.

Every setting is ILLUSTRATIVE.

    python chapters/ch23/run.py
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from algo_builder.strategies.compounder_screen import (base_effect, disqualify, long_term_from, score,
                                                       screen, yoy_declines)


def companies(n: int = 300, seed: int = 23) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "is_financial": rng.random(n) < 0.15,
        "years_listed": rng.integers(1, 30, n), "loss_years_5": rng.poisson(0.4, n),
        "interest_cover": rng.lognormal(1.8, 0.9, n), "cfo_to_profit_5y": rng.uniform(0, 1.5, n),
        "margin_cv": rng.uniform(0.05, 0.9, n), "promoter_change_4q": rng.normal(0, 2, n),
        "roce": rng.normal(0.16, 0.08, n), "roe": rng.normal(0.15, 0.07, n), "debt_equity": rng.uniform(0, 2, n),
        "profit_up_years": rng.integers(0, 6, n), "profit_cagr_5y": rng.normal(0.12, 0.1, n),
        "sales_cagr_5y": rng.normal(0.10, 0.07, n), "pe": rng.lognormal(3.2, 0.4, n)},
        index=[f"C{i:03d}" for i in range(n)])
    for col in ("interest_cover", "cfo_to_profit_5y", "promoter_change_4q"):
        df.loc[df.sample(frac=0.03, random_state=len(col)).index, col] = None   # a few gaps
    return df


def part1() -> None:
    df = companies()
    reasons = df.apply(lambda r: disqualify(r.to_dict()), axis=1)
    first = screen(df).head(20).index
    late = score(df)
    late = late[reasons.loc[late.index].str.len() == 0].head(20).index
    gaps = df[df.isna().any(axis=1)]
    lenient = gaps.fillna({"interest_cover": 99, "cfo_to_profit_5y": 1, "promoter_change_4q": 0})
    would_pass = sum(not disqualify(r.to_dict()) for _, r in lenient.iterrows())
    print(f"1. {len(df)} companies, {int((reasons.str.len() == 0).sum())} pass the floors; "
          f"{len(gaps)} have a missing figure, of which {would_pass} would have passed if a gap counted as a pass")
    print(f"   top 20 (floors first) vs top 20 (rank all, filter after): {len(set(first) & set(late))} in common")


def part2() -> None:
    a = pd.Series([100, 80, 50, 70, 100, 130, 145, 160.0], index=range(2018, 2026))
    b = pd.Series([60, 67, 75, 88, 103, 120, 140, 160.0], index=range(2018, 2026))
    print(f"\n2. Base effect (5-year minus 7-year growth): recovering A {base_effect(a):+.1%}, "
          f"compounder B {base_effect(b):+.1%}")


def part3() -> None:
    q = pd.period_range("2021Q1", "2025Q4", freq="Q")
    season = np.tile([0.9, 0.8, 1.05, 1.25], 5)                       # monsoon dip, festival peak
    profit = pd.Series(100 * 1.035 ** np.arange(20) * season, index=q)
    qoq = (profit.diff() < 0).astype(int)
    alarms = int(((qoq + qoq.shift()) == 2).sum())
    print(f"\n3. A business growing ~15% a year, with seasons: two quarter-on-quarter falls in a row "
          f"{alarms} times; latest run of year-on-year falls: {yoy_declines(profit)}")


def part4() -> None:
    print(f"\n4. Bought 4 Nov 2025: long-term from {long_term_from(date(2025, 11, 4))}; "
          f"bought 29 Feb 2024: long-term from {long_term_from(date(2024, 2, 29))}")


if __name__ == "__main__":
    part1()
    part2()
    part3()
    part4()
