"""
Chapter 28 — Scheduling, logging and alerts.

  1. A typo in a signal step, twenty trading days, two ways: a broad try/except
     inside the step, against no inner catch and the job wrapper outside it.
  2. A week of health checks: a job that did not run, a job that failed, and a
     day when the machine was off and nothing ran at all.

Everything is SIMULATED; no message leaves the machine.

    python chapters/ch28/run.py
"""
from __future__ import annotations

import logging
import tempfile
from datetime import datetime, time, timedelta
from pathlib import Path

from algo_builder.common import jobs
from algo_builder.infra import kill_switches as ks


def enrich(signal: dict) -> dict:
    signal["day_type"] = signal["bars"].day_typ        # the typo: no such attribute
    return signal


def signals_swallowed(day_bars) -> list[dict]:
    out = []
    for kind in ("trend", "range"):
        s = dict(kind=kind, bars=day_bars)
        try:
            if kind == "trend":
                s = enrich(s)
            out.append(s)
        except Exception:                               # "keep going"
            pass
    return out


def signals_plain(day_bars) -> list[dict]:
    out = []
    for kind in ("trend", "range"):
        s = dict(kind=kind, bars=day_bars)
        out.append(enrich(s) if kind == "trend" else s)
    return out


def part1(base: Path) -> None:
    class Bars:
        day_type = "trend"
    for label, fn in (("try/except inside the step", signals_swallowed),
                      ("no inner catch, wrapper outside", signals_plain)):
        sent, trend, codes = [], 0, []
        for d in range(20):
            now = datetime(2026, 10, 1, 9, 20) + timedelta(days=d)
            box = []
            codes.append(jobs.run_job("signals", lambda: box.extend(fn(Bars())) or len(box), is_exit_job=False,
                                      now=now, log_dir=base / label[:5], heartbeat_dir=base / "hb",
                                      alert=sent.append))
            trend += sum(s["kind"] == "trend" for s in box)
        first = f"day {codes.index(1) + 1}" if 1 in codes else "never"
        print(f"1. {label:<32} trend signals {trend:>2} in 20 days; first alert: {first}")


def part2(base: Path) -> None:
    sched = {"entry": time(9, 15), "exit": time(9, 30), "safety_net_exit": time(14, 45)}
    week = {"Mon": "normal", "Tue": "exit did not run", "Wed": "exit failed",
            "Thu": "normal", "Fri": "machine off all day"}
    hb = base / "hb2"
    print("2. Messages received from the morning health check (10:00):")
    for i, (day, what) in enumerate(week.items()):
        d = datetime(2026, 11, 2 + i)
        if what == "machine off all day":
            print(f"   {day}: (nothing arrived)  <- the missing all-clear is the alarm")
            continue
        for name, t in sched.items():
            if what == "exit did not run" and name == "exit":
                continue
            ok = not (what == "exit failed" and name == "exit")
            jobs.run_job(name, (lambda: "ok") if ok else (lambda: 1 / 0), is_exit_job=name != "entry",
                         now=d.replace(hour=t.hour, minute=t.minute), log_dir=base / "logs2",
                         heartbeat_dir=hb, alert=lambda m: True)
        problems = jobs.health_check(sched, hb, d.replace(hour=10, minute=0), trading_day=True)
        print(f"   {day}: " + ("all clear" if not problems else "ALERT " + "; ".join(problems)))


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.CRITICAL)
    with tempfile.TemporaryDirectory() as d:
        ks.CONTROL_DIR = Path(d) / "control"
        ks.HALT, ks.HALT_ALL, ks.KILL = (ks.CONTROL_DIR / n for n in ("HALT", "HALT_ALL", "KILL"))
        part1(Path(d))
        part2(Path(d))
