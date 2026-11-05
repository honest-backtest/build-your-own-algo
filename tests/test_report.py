from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from algo_builder.robustness.report import robustness_report, run_stamp, save_run


def _trades(pnl):
    e = pd.date_range("2022-01-03 10:00", periods=len(pnl), freq="B")
    return pd.DataFrame({"entry_ts": e, "exit_ts": e, "pnl": pnl})


def test_missing_inputs_are_reported_not_skipped():
    rep = robustness_report(_trades(np.full(200, 100.0) + np.tile([50, -50], 100)), 100_000)
    status = rep["checks"].set_index("check").status
    assert status["costs"] == status["drift"] == status["fill delay"] == "NOT RUN"
    assert status["net result"] == "PASS"


def test_a_lottery_ticket_is_flagged():
    pnl = np.full(300, -20.0)
    pnl[:10] = 1_000.0
    status = robustness_report(_trades(pnl), 100_000)["checks"].set_index("check").status
    assert status["net result"] == "PASS" and status["ex-top-10"] == "WARN"


def test_stamp_fingerprints_settings_and_runs_are_never_overwritten(tmp_path):
    now = datetime(2026, 10, 30, 10, 0)
    a = run_stamp("Backtest", {"fast": 20}, "2021-01-01", "2026-06-30", "FUTURES 2026", now)
    b = run_stamp("Backtest", {"fast": 21}, "2021-01-01", "2026-06-30", "FUTURES 2026", now)
    assert a["code_hash"] == b["code_hash"] and a["params_hash"] != b["params_hash"]
    t = _trades(np.tile([30.0, -10.0], 25))
    rep = robustness_report(t, 100_000)
    out = save_run(str(tmp_path), a, rep, t)
    assert {p.name for p in out.iterdir()} == {"stamp.json", "checks.csv", "trades.csv"}
    with pytest.raises(FileExistsError):
        save_run(str(tmp_path), a, rep, t)
