from datetime import date

import pandas as pd

from algo_builder.strategies.ppn_quarterly_call import P, Cycle, nearest_call, run


def _chain():
    return pd.DataFrame({"strike": [10_000.0, 10_300.0, 12_000.0], "opt_type": "CE",
                         "close": [400.0, 280.0, 5.0], "contracts": [100, 100, 0]})


def test_untraded_strikes_do_not_exist():
    assert nearest_call(_chain(), 12_000, min_contracts=1) == (10_300.0, 280.0)


def test_only_interest_is_spent_in_whole_lots():
    c = Cycle(date(2026, 1, 1), date(2026, 3, 31), _chain(), 10_000.0, 11_000.0, 0.07, 75)
    r = run([c], capital=30_00_000, p=dict(P, otm=0.03, slip=0.0, cost_per_lot=0.0)).iloc[0]
    per_lot = 280.0 * 75
    assert r.lots == int(r.interest // per_lot) and r.spent <= r.interest
    assert r.payoff == r.lots * 75 * 700 and r.fund == 30_00_000 + r.interest - r.spent + r.payoff


def test_a_low_yield_quarter_buys_nothing_and_the_capital_is_kept():
    c = Cycle(date(2026, 1, 1), date(2026, 3, 31), _chain(), 10_000.0, 9_000.0, 0.03, 75)
    r = run([c], capital=30_00_000).iloc[0]
    assert r.lots == 0 and r.note.startswith("killed") and r.fund > 30_00_000
