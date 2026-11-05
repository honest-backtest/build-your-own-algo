"""
Book 2, Chapter 11 — The default report.

Every backtest produces the same report, by default, and every report carries a
stamp saying exactly what produced it. A test that could not be run is reported as
NOT RUN, never silently left out, because a missing test is not a passed one.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from algo_builder.backtest.metrics import monte_carlo, trade_stats
from algo_builder.infra.candle_cache import write_atomic_json
from algo_builder.robustness.by_year_and_ex_top_n import by_year, concentration_by_day, ex_top_n
from algo_builder.robustness.drift_benchmark import drift_report, random_entry_means
from algo_builder.robustness.fill_delay_test import fill_delay_table
from algo_builder.robustness.gross_net import cost_or_signal, friction_sweep_points

PACKAGE = Path(__file__).resolve().parents[1]


def run_stamp(label: str, params: dict, data_from: str, data_to: str, costs: str,
              now: datetime) -> dict:
    """What produced a result: evidence label, settings, data window, cost profile,
    and a fingerprint of every source file in the package at the moment it ran."""
    h = hashlib.sha256()
    for f in sorted(PACKAGE.rglob("*.py")):
        h.update(f.relative_to(PACKAGE).as_posix().encode())
        h.update(f.read_bytes())
    cfg = json.dumps(params, sort_keys=True, default=str)
    return dict(label=label, run_at=now.isoformat(timespec="seconds"), params=params,
                params_hash=hashlib.sha256(cfg.encode()).hexdigest()[:12],
                code_hash=h.hexdigest()[:12], data_from=data_from, data_to=data_to, costs=costs)


def _check(name: str, ok: bool | None, detail: str) -> dict:
    return dict(check=name, status="NOT RUN" if ok is None else "PASS" if ok else "WARN", detail=detail)


def robustness_report(trades: pd.DataFrame, capital: float, *, cost_pts: float | None = None,
                      closes: pd.DataFrame | None = None, minute_bars: dict | None = None,
                      signals: pd.DataFrame | None = None, max_dd_limit: float = 0.25,
                      seed: int = 7) -> dict:
    """trades: entry_ts, exit_ts, pnl (net). Optional columns and inputs switch on the
    rest: gross_pts + cost_pts (costs), symbol/hold/ret + closes (drift), minute_bars +
    signals (fill delay). Returns the tables and a checklist."""
    rep: dict = {"stats": trade_stats(trades), "by_year": by_year(trades),
                 "ex_top": ex_top_n(trades, ns=(0, 3, 10, 20)),
                 "concentration": concentration_by_day(trades),
                 "monte_carlo": monte_carlo(trades, capital, n_sim=2000, seed=seed)}
    s, yr, ex, mc = rep["stats"], rep["by_year"], rep["ex_top"].set_index("removed_top"), rep["monte_carlo"]
    checks = [_check("sample", s["t_stat"] >= 2, f"{s['n']} trades, t = {s['t_stat']:.2f}"),
              _check("net result", trades.pnl.sum() > 0, f"total {trades.pnl.sum():,.0f}"),
              _check("years", yr["sum"].drop(yr["sum"].idxmax()).sum() > 0,
                     f"{int(yr.positive.sum())} of {len(yr)} years positive; total without the best year "
                     f"{yr['sum'].drop(yr['sum'].idxmax()).sum():,.0f}"),
              _check("ex-top-10", ex.loc[10, "total"] > 0, f"total without the 10 best trades {ex.loc[10, 'total']:,.0f}"),
              _check("drawdown", mc["worst_5pct_max_dd"] > -max_dd_limit,
                     f"median {mc['median_max_dd']:.0%}, worst 5% {mc['worst_5pct_max_dd']:.0%}, "
                     f"limit {-max_dd_limit:.0%}; P(lose half) {mc['p_lose_half']:.1%}")]

    if "gross_pts" in trades and cost_pts is not None:
        rep["friction"] = friction_sweep_points(trades.gross_pts)
        verdict = cost_or_signal(trades.gross_pts, cost_pts)
        checks.append(_check("costs", verdict.startswith("survives"), verdict))
    else:
        checks.append(_check("costs", None, "needs gross_pts and cost_pts"))

    if closes is not None and {"symbol", "hold", "ret"} <= set(trades.columns):
        rnd = random_entry_means(closes, trades, n_draws=1000, seed=seed)
        rep["drift"] = d = drift_report(trades, rnd)
        checks.append(_check("drift", d["excess_pct"] > 0 and d["p_value"] < 0.05,
                             f"excess {d['excess_pct']:+.2f}% a trade, p = {d['p_value']:.3f}"))
    else:
        checks.append(_check("drift", None, "needs closes and symbol/hold/ret"))

    if minute_bars is not None and signals is not None:
        rep["fill_delay"] = fd = fill_delay_table(minute_bars, signals, delays=(0, 1, 5))
        kept = fd.set_index("delay_min").edge_kept.get(1, np.nan)
        checks.append(_check("fill delay", bool(kept >= 0.5),
                             f"edge kept one minute late: {kept:.0%}" if np.isfinite(kept) else "no edge at the print"))
    else:
        checks.append(_check("fill delay", None, "needs minute_bars and signals"))

    rep["checks"] = pd.DataFrame(checks)
    return rep


def save_run(folder: str, stamp: dict, report: dict, trades: pd.DataFrame) -> Path:
    """One folder per run, never overwritten: the stamp, the checklist, the trades."""
    out = Path(folder) / f"{stamp['run_at'].replace(':', '')}_{stamp['params_hash']}"
    out.mkdir(parents=True, exist_ok=False)
    write_atomic_json(stamp, str(out / "stamp.json"))
    report["checks"].to_csv(out / "checks.csv", index=False)
    trades.to_csv(out / "trades.csv", index=False)
    return out
