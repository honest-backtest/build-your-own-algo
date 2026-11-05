from datetime import date

import pandas as pd

from algo_builder.strategies.compounder_screen import (base_effect, disqualify, known_on, long_term_from,
                                                       screen, yoy_declines)

BASE = dict(is_financial=False, years_listed=10, loss_years_5=0, interest_cover=8.0, cfo_to_profit_5y=0.9,
            margin_cv=0.2, promoter_change_4q=0.0)


def test_zero_and_missing_are_not_passes():
    assert "weak interest cover" in disqualify(dict(BASE, interest_cover=0.0))
    assert "poor cash conversion" in disqualify(dict(BASE, cfo_to_profit_5y=0.0))
    assert "missing interest_cover" in disqualify(dict(BASE, interest_cover=None))
    assert disqualify(BASE) == []


def test_floors_come_before_ranks():
    rows = [dict(BASE, roce=r, roe=0.15, debt_equity=0.5, profit_up_years=3, profit_cagr_5y=0.1,
                 sales_cagr_5y=0.1, pe=20) for r in (0.1, 0.2, 0.3)]
    rows[2]["loss_years_5"] = 3                                     # the best on paper, disqualified
    df = pd.DataFrame(rows, index=["A", "B", "C"])
    out = screen(df)
    assert list(out.index) == ["B", "A"] and "C" not in out.index


def test_base_effect_yoy_announcements_and_tax_date():
    a = pd.Series([100, 80, 50, 70, 100, 130, 145, 160.0], index=range(2018, 2026))
    assert base_effect(a) > 0.15 and base_effect(a.iloc[-5:]) is None
    q = pd.Series([10, 8, 11, 13, 9, 7, 10, 12.0], index=pd.period_range("2024Q1", periods=8, freq="Q"))
    assert yoy_declines(q) == 4
    res = pd.DataFrame({"quarter_end": pd.to_datetime(["2026-06-30", "2026-09-30"]),
                        "announced": pd.to_datetime(["2026-08-10", "2026-11-12"])})
    assert len(known_on(res, pd.Timestamp("2026-10-15"))) == 1
    assert long_term_from(date(2025, 11, 4)) == date(2026, 11, 5)
