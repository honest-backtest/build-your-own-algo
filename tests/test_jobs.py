"""Chapter 28: job wrapper, alerts and health check."""
from datetime import datetime, time

import pytest

from algo_builder.common import jobs
from algo_builder.common.notify import send_alert, transport_ready
from algo_builder.infra import kill_switches as ks


@pytest.fixture
def dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(ks, "CONTROL_DIR", tmp_path / "control")
    for n in ("HALT", "HALT_ALL", "KILL"):
        monkeypatch.setattr(ks, n, tmp_path / "control" / n)
    return tmp_path


def test_failures_are_logged_alerted_and_exit_non_zero(dirs):
    sent, now = [], datetime(2026, 11, 4, 9, 30)
    kw = dict(now=now, log_dir=dirs / "logs", heartbeat_dir=dirs / "hb", alert=sent.append)

    def typo():
        print("scanning")
        return object().no_such_attribute
    assert jobs.run_job("exit", typo, is_exit_job=True, **kw) == 1
    text = (dirs / "logs" / "exit" / "2026-11-04.log").read_text()
    assert "scanning" in text and "AttributeError" in text and "Traceback" in text
    assert len(sent) == 1 and "FAILED" in sent[0]

    def holiday():
        raise jobs.ExpectedSkip("market holiday")
    assert jobs.run_job("entry", holiday, is_exit_job=False, **kw) == 0 and len(sent) == 1
    assert jobs.run_job("entry", lambda: "2 orders", is_exit_job=False, **kw) == 0


def test_halt_all_alerts_on_every_run_and_halt_does_not_stop_exits(dirs):
    sent, ran = [], []
    kw = dict(now=datetime(2026, 11, 4, 9, 30), log_dir=dirs / "logs", heartbeat_dir=dirs / "hb", alert=sent.append)
    ks.CONTROL_DIR.mkdir(parents=True)
    ks.HALT.write_text("x")
    jobs.run_job("exit", lambda: ran.append(1), is_exit_job=True, **kw)
    jobs.run_job("entry", lambda: ran.append(2), is_exit_job=False, **kw)
    assert ran == [1] and sent == []
    ks.HALT_ALL.write_text("x")
    for _ in range(3):
        jobs.run_job("exit", lambda: ran.append(3), is_exit_job=True, **kw)
    assert ran == [1] and len(sent) == 3


def test_health_check_finds_missing_late_and_failed_jobs(tmp_path):
    hb = tmp_path / "hb"
    jobs.write_heartbeat(hb, "entry", "ok", datetime(2026, 11, 4, 9, 15))
    jobs.write_heartbeat(hb, "exit", "ok", datetime(2026, 11, 3, 9, 30))          # yesterday
    jobs.write_heartbeat(hb, "roll", "failed", datetime(2026, 11, 4, 9, 40), "KeyError: 'x'")
    sched = {"entry": time(9, 15), "exit": time(9, 30), "roll": time(9, 40), "net": time(14, 45), "eod": time(9, 0)}
    p = jobs.health_check(sched, hb, datetime(2026, 11, 4, 10, 0), trading_day=True)
    assert p == ["exit: did not run today (last 2026-11-03T09:30:00)", "roll: FAILED at 09:40: KeyError: 'x'",
                 "eod: no heartbeat at all"]
    assert jobs.health_check(sched, hb, datetime(2026, 11, 7, 10, 0), trading_day=False) == []


def test_undelivered_alerts_go_to_the_log(caplog):
    with caplog.at_level("ERROR"):
        assert send_alert("exit failed", transport=lambda t, q: False) is False
    assert "exit failed" in caplog.text
    assert transport_ready(lambda t, q: True) and not transport_ready(lambda t, q: 1 / 0)
