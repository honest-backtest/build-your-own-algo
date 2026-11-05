"""
Chapter 10 — Walk-forward: tuning without fooling yourself.

Two SYNTHETIC strategies, each with a 5 x 5 grid of settings (25 combos), traded
2017-2026, about 60 trades a combo a year, results in points per trade:
  1. 'regime-fitted': no lasting edge. Which settings look best depends on a regime
     that changes every two or three years, plus luck;
  2. 'plateau': a small, stable edge across a broad region of the grid, plus the
     same luck.
For each: the best combo chosen in hindsight, the anchored walk-forward, the
in-sample/out-of-sample rank correlation, and the plateau check.

    python chapters/ch10/run.py
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from algo_builder.robustness.walk_forward import plateau_share, rank_correlation, walk_forward

FAST = (5, 10, 20, 40, 80)
STOP = (1.0, 1.5, 2.0, 2.5, 3.0)
YEARS = range(2017, 2027)
REGIME = {2017: 1, 2018: 1, 2019: -1, 2020: -1, 2021: -1, 2022: 1, 2023: 1, 2024: -1, 2025: -1, 2026: 1}
COMBOS = [dict(fast=f, stop=s) for f, s in product(FAST, STOP)]


def make_runner(kind: str, seed: int = 10):
    def run(c: dict) -> pd.DataFrame:
        i, j = FAST.index(c["fast"]), STOP.index(c["stop"])
        rng = np.random.default_rng(seed * 1000 + i * 10 + j)
        rows = []
        for y in YEARS:
            luck = rng.normal(0, 1.0)                                  # this combo, this year
            if kind == "regime-fitted":
                style = (i - 2) / 2                                    # -1 shortest average ... +1 longest
                edge = 5.0 * REGIME[y] * style + luck
            else:
                edge = 2.0 * (1 <= i <= 3) * (j >= 2) + luck          # a broad, stable region
            days = pd.bdate_range(f"{y}-01-01", f"{y}-12-31")
            n = 100
            rows.append(pd.DataFrame({"entry_ts": rng.choice(days, n, replace=False),
                                      "pnl": rng.normal(edge, 15.0, n)}))
        return pd.concat(rows, ignore_index=True).sort_values("entry_ts")
    return run


def report(kind: str) -> dict:
    run = make_runner(kind)
    totals = pd.DataFrame([dict(**c, pnl=run(c).pipe(lambda t: t[t.entry_ts.dt.year >= 2020]).pnl.sum())
                           for c in COMBOS])
    best = totals.loc[totals.pnl.idxmax()]
    wf = walk_forward(COMBOS, run, 2020, 2026, min_trades=100)
    rho, _ = rank_correlation(COMBOS, run, range(2017, 2022), range(2022, 2027))
    full = pd.DataFrame([dict(**c, pnl=run(c).pnl.sum()) for c in COMBOS])
    full["neighbours_positive"] = plateau_share(full, ["fast", "stop"])
    top = full.loc[full.pnl.idxmax()]
    return dict(strategy=kind,
                best_in_hindsight=f"fast {best.fast:.0f}, stop {best.stop}: {best.pnl:+,.0f}",
                walk_forward_oos=f"{wf.oos_pnl.sum():+,.0f}",
                years_oos_positive=f"{(wf.oos_pnl > 0).sum()} of {len(wf)}",
                rank_corr=round(rho, 2),
                combos_positive=f"{(full.pnl > 0).sum()} of {len(full)}",
                best_combo_neighbours_positive=f"{top.neighbours_positive:.0%}")


def main() -> None:
    pd.set_option("display.width", 160)
    out = pd.DataFrame([report("regime-fitted"), report("plateau")]).set_index("strategy").T
    print("Points, 2020-2026 test years; 25 combos (synthetic)")
    print(out.to_string())
    wf = walk_forward(COMBOS, make_runner("regime-fitted"), 2020, 2026, min_trades=100)
    print("\nRegime-fitted, year by year:")
    print(wf.assign(chosen=wf.chosen.map(lambda c: f"fast {c['fast']}, stop {c['stop']}"))
            .round(1).to_string(index=False))


if __name__ == "__main__":
    main()
