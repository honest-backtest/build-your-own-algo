"""
Book 2, Chapter 28 — Scheduling, logging and alerts.

Every scheduled job runs inside `run_job`, which:
  * writes everything the job logs or prints to a dated log file;
  * checks the kill switches first, and alerts on EVERY run while HALT_ALL exists;
  * lets a job say "nothing to do today" with ExpectedSkip, and treats anything
    else that escapes it as a failure: traceback to the log, an alert, and a
    non-zero exit code for the operating system's scheduler;
  * writes a heartbeat file recording when it last ran and how it ended.

`health_check` is a separate job, ideally on a different machine, that reads the
heartbeats against the schedule and sends one message a day even when all is
well: when that message does not arrive, a person notices. ILLUSTRATIVE.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
from datetime import datetime, time
from pathlib import Path
from typing import Callable, Optional

from algo_builder.common.notify import send_alert
from algo_builder.infra.kill_switches import job_gate

log = logging.getLogger("jobs")


class ExpectedSkip(Exception):
    """Raised by a job that has, correctly, nothing to do (a holiday, a filter)."""


class _ToLog:
    """A file-like object that sends print() output to a logger, line by line."""
    def __init__(self, logger: logging.Logger, level: int):
        self.logger, self.level = logger, level

    def write(self, s: str) -> None:
        for line in s.rstrip().splitlines():
            self.logger.log(self.level, line)

    def flush(self) -> None:
        pass


def write_heartbeat(folder: Path, name: str, status: str, now: datetime, detail: str = "") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f"{name}.json.tmp"
    tmp.write_text(json.dumps(dict(status=status, at=now.isoformat(timespec="seconds"), detail=detail[:500])))
    os.replace(tmp, folder / f"{name}.json")


def run_job(name: str, fn: Callable[[], object], *, is_exit_job: bool, now: datetime,
            log_dir: Path, heartbeat_dir: Path, system: Optional[str] = None,
            alert: Callable[[str], bool] = send_alert) -> int:
    """Run one scheduled job. Returns the exit code for the scheduler: 0 ran or
    skipped as expected, 1 failed."""
    path = log_dir / name / f"{now:%Y-%m-%d}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    old_level = root.level
    root.setLevel(min(old_level or logging.INFO, logging.INFO))
    out = logging.getLogger(f"job.{name}")
    try:
        with contextlib.redirect_stdout(_ToLog(out, logging.INFO)), \
             contextlib.redirect_stderr(_ToLog(out, logging.ERROR)):
            out.info("start (%s)", "exit job" if is_exit_job else "entry job")
            gate = job_gate(is_exit_job, system)
            if gate:
                out.warning("skipped: %s", gate)
                if "HALT_ALL" in gate:
                    alert(f"{name}: {gate}")
                write_heartbeat(heartbeat_dir, name, "skipped", now, gate)
                return 0
            try:
                result = fn()
            except ExpectedSkip as e:
                out.info("nothing to do: %s", e)
                write_heartbeat(heartbeat_dir, name, "skipped", now, str(e))
                return 0
            except Exception as e:
                out.exception("FAILED")
                alert(f"{name} FAILED at {now:%H:%M}: {type(e).__name__}: {e}. Log: {path.name}")
                write_heartbeat(heartbeat_dir, name, "failed", now, f"{type(e).__name__}: {e}")
                return 1
            out.info("done: %s", result)
            write_heartbeat(heartbeat_dir, name, "ok", now, str(result))
            return 0
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
        handler.close()


def health_check(schedule: dict[str, time], heartbeat_dir: Path, now: datetime,
                 trading_day: bool, grace_min: int = 15) -> list[str]:
    """Every job that should have run by now today and did not, or failed. The
    caller sends the result every day: problems as an alert, and otherwise one
    short all-clear. A missing all-clear is itself the alarm."""
    if not trading_day:
        return []
    problems = []
    for name, due in schedule.items():
        if (now - now.replace(hour=due.hour, minute=due.minute)).total_seconds() < grace_min * 60:
            continue                                    # not due yet
        f = heartbeat_dir / f"{name}.json"
        try:
            hb = json.loads(f.read_text())
        except (OSError, ValueError):
            problems.append(f"{name}: no heartbeat at all")
            continue
        if hb["at"][:10] != now.date().isoformat():
            problems.append(f"{name}: did not run today (last {hb['at']})")
        elif hb["status"] == "failed":
            problems.append(f"{name}: FAILED at {hb['at'][11:16]}: {hb['detail']}")
    return problems
