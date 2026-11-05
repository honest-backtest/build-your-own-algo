"""
Book 2, Chapter 23 — Fundamentals pipelines and the compounder book.

Shows: hard disqualifiers (floors) applied BEFORE ranking, a composite
       percentile-rank score with separate weights for financials, telling
       annual from quarterly tables by DATE SPACING (fiscal year-ends change),
       and never caching a partially parsed page as a success.

Floors and weights are ILLUSTRATIVE — tune them to your own philosophy.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


def cagr(a, b, years):
    if a is None or b is None or a <= 0 or b <= 0 or years <= 0:
        return None
    return (b / a) ** (1 / years) - 1


def table_kind(column_dates: list[pd.Timestamp]) -> str:
    gaps = pd.Series(sorted(column_dates)).diff().dt.days.dropna()
    return "annual" if gaps.median() > 200 else "quarterly"


def disqualify(m: dict) -> list[str]:
    """Reasons a name must not enter a multi-year hold, whatever its score. ILLUSTRATIVE
    floors. A figure that is missing is a reason too: a gap in the data is not a pass."""
    bad = []

    def need(key: str):
        v = m.get(key)
        if v is None:
            bad.append(f"missing {key}")
        return v

    years, losses = need("years_listed"), need("loss_years_5")
    if years is not None and years < 3:
        bad.append("recent listing")
    if losses is not None and losses >= 2:
        bad.append("repeated losses")
    if not m["is_financial"]:
        cover, cash, cv = need("interest_cover"), need("cfo_to_profit_5y"), need("margin_cv")
        if cover is not None and cover < 3:
            bad.append("weak interest cover")
        if cash is not None and cash < 0.4:
            bad.append("poor cash conversion")
        if cv is not None and cv > 0.6:
            bad.append("cyclical margins")
    promoter = need("promoter_change_4q")
    if promoter is not None and promoter < -3.0:
        bad.append("promoter selling")
    return bad


def base_effect(profit_by_year: pd.Series, short: int = 5, long: int = 7) -> float | None:
    """Growth over the short window minus growth over the long one, both ending at the
    latest year. Large and positive: a recovery from a bad year, not compounding."""
    p = profit_by_year.sort_index()
    if len(p) <= long:
        return None
    g_short = cagr(p.iloc[-1 - short], p.iloc[-1], short)
    g_long = cagr(p.iloc[-1 - long], p.iloc[-1], long)
    return None if g_short is None or g_long is None else g_short - g_long


def pct_rank(s: pd.Series, invert: bool = False) -> pd.Series:
    r = s.rank(pct=True)
    return 1 - r if invert else r


def score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for mask in (~df.is_financial, df.is_financial):
        g = df[mask]
        if g.empty:
            continue
        if not g.is_financial.iloc[0]:
            quality = 0.5 * pct_rank(g.roce) + 0.3 * pct_rank(g.roe) + 0.2 * pct_rank(g.debt_equity, True)
        else:
            quality = 0.7 * pct_rank(g.roe) + 0.3 * pct_rank(g.profit_up_years)
        durability = 0.4 * pct_rank(g.profit_cagr_5y) + 0.3 * pct_rank(g.sales_cagr_5y) \
            + 0.3 * pct_rank(g.margin_cv, True)
        value = pct_rank(g.pe, True)
        df.loc[mask, "score"] = 0.4 * quality + 0.4 * durability + 0.2 * value
    return df.sort_values("score", ascending=False)


def cache_parse_result(cache: dict, symbol: str, parsed: dict | None, required: list[str]) -> None:
    """Partial parse = FAILED (refetch next run), never a cached success."""
    ok = parsed is not None and all(parsed.get(k) is not None for k in required)
    cache[symbol] = {"status": "ok" if ok else "failed", "data": parsed if ok else None}


def screen(df: pd.DataFrame) -> pd.DataFrame:
    """Floors FIRST, then ranks among the survivors only: a disqualified name must
    neither be bought nor shift anyone else's percentile."""
    reasons = df.apply(lambda r: disqualify(r.to_dict()), axis=1)
    survivors = df[reasons.str.len() == 0]
    return score(survivors) if len(survivors) else survivors.assign(score=pd.Series(dtype=float))


def known_on(results: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Quarterly results as they were known on `day`: only rows ANNOUNCED by then
    (columns quarter_end, announced, ...). Quarter-end dates are not publication dates."""
    return results[results.announced <= day]


def yoy_declines(quarterly_profit: pd.Series) -> int:
    """Consecutive latest quarters whose profit fell against the SAME quarter a year
    earlier. Comparing with the previous quarter mistakes seasons for trends."""
    yoy = quarterly_profit.sort_index().pct_change(4).dropna()
    n = 0
    for v in reversed(yoy.to_list()):
        if v >= 0:
            break
        n += 1
    return n


def long_term_from(buy: date) -> date:
    """First sale date on which a listed Indian share bought on `buy` is held for more
    than twelve months (as of 2026; check the current rules)."""
    try:
        anniversary = buy.replace(year=buy.year + 1)
    except ValueError:                                   # bought on 29 February
        anniversary = buy.replace(year=buy.year + 1, day=28)
    return anniversary + timedelta(days=1)
