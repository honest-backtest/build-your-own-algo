import numpy as np
import pandas as pd

from algo_builder.robustness.walk_forward import plateau_share, rank_correlation, walk_forward


def _yearly(pnl_by_year: dict, n: int = 40, exit_days: int = 0) -> pd.DataFrame:
    rows = []
    for y, p in pnl_by_year.items():
        e = pd.date_range(f"{y}-01-05", periods=n, freq="5D")
        rows.append(pd.DataFrame({"entry_ts": e, "exit_ts": e + pd.Timedelta(days=exit_days), "pnl": p}))
    return pd.concat(rows, ignore_index=True)


def test_training_trades_still_open_at_the_test_year_are_left_out():
    # One huge winner entered in December and closed in January must not train the January choice.
    def run(c):
        t = _yearly({2021: 0.0, 2022: 0.0}, n=40)
        if c["k"] == "peek":
            extra = pd.DataFrame({"entry_ts": [pd.Timestamp("2021-12-20")],
                                  "exit_ts": [pd.Timestamp("2022-01-10")], "pnl": [1e6]})
            t = pd.concat([t, extra], ignore_index=True)
        return t
    wf = walk_forward([{"k": "peek"}, {"k": "plain"}], run, 2022, 2022, min_trades=30)
    assert wf.loc[0, "train_score"] == 0.0


def test_rolling_window_only_sees_recent_years():
    def run(c):
        return _yearly({2018: 10.0 * c["a"], 2019: -1.0 * c["a"], 2020: 0.0})
    anchored = walk_forward([{"a": 1}, {"a": -1}], run, 2020, 2020, min_trades=30)
    rolling = walk_forward([{"a": 1}, {"a": -1}], run, 2020, 2020, train_years=1, min_trades=30)
    assert anchored.loc[0, "chosen"] == {"a": 1} and rolling.loc[0, "chosen"] == {"a": -1}


def test_rank_correlation_sees_a_flip():
    def run(c):
        return _yearly({2020: c["x"], 2021: -c["x"]})
    rho, table = rank_correlation([{"x": v} for v in (1.0, 2.0, 3.0)], run, range(2020, 2021), range(2021, 2022))
    assert np.isclose(rho, -1.0) and len(table) == 3


def test_plateau_share_counts_neighbours():
    grid = pd.DataFrame([dict(a=a, b=b, pnl=1.0) for a in (1, 2, 3) for b in (1, 2, 3)])
    grid.loc[(grid.a == 2) & (grid.b == 2), "pnl"] = 5.0
    grid.loc[(grid.a == 1) & (grid.b == 1), "pnl"] = -1.0
    share = plateau_share(grid, ["a", "b"])
    centre = share[(grid.a == 2) & (grid.b == 2)].iloc[0]
    assert np.isclose(centre, 7 / 8)
