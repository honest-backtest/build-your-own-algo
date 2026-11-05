"""
Chapter 27 — Rails and kill switches that never block exits.

  1. The switch matrix: which jobs each switch stops.
  2. An order burst through the rails, including the usual fat-finger mistakes.
  3. A bad day for an intraday option seller that re-enters after each stop,
     with the daily-loss rail checked at the close, on realised P&L during the
     day, and on marked-to-market P&L during the day.

Every number here is SYNTHETIC and every cap a PLACEHOLDER.

    python chapters/ch27/run.py
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from algo_builder.infra import kill_switches as ks


def use_dir(d: str) -> None:
    ks.CONTROL_DIR = Path(d)
    ks.HALT, ks.HALT_ALL, ks.KILL = (Path(d) / n for n in ("HALT", "HALT_ALL", "KILL"))


def part1() -> None:
    print("1. Switch matrix (system 'expiry'):")
    print(f"   {'switch present':<22}{'entry job':<34}exit job")
    cases = [("none", None), ("HALT", "HALT"), ("KILL", "KILL"), ("expiry/HALT", "expiry/HALT"),
             ("covered_call/HALT", "covered_call/HALT"), ("HALT_ALL", "HALT_ALL")]
    for label, f in cases:
        with tempfile.TemporaryDirectory() as d:
            use_dir(d)
            if f:
                p = Path(d) / f
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("x")
            e, x = ks.job_gate(False, "expiry"), ks.job_gate(True, "expiry")
            print(f"   {label:<22}{('runs' if e is None else 'skipped'):<34}{'runs' if x is None else 'FROZEN'}")


class Ledger:
    def __init__(self):
        self.n = 0

    def count_orders_today(self, sleeve):
        return self.n


def part2() -> None:
    with tempfile.TemporaryDirectory() as d:
        use_dir(d)
        led = Ledger()
        tries = [("normal share order", dict(sleeve="sleeve_a", qty=1, price=812.4, ltp=811.9)),
                 ("price typed ten times too high", dict(sleeve="sleeve_a", qty=1, price=8124.0, ltp=811.9)),
                 ("quantity 100 for 1", dict(sleeve="sleeve_a", qty=100, price=812.4, ltp=811.9)),
                 ("no last price", dict(sleeve="sleeve_a", qty=1, price=812.4, ltp=None)),
                 ("one lot of options", dict(sleeve="sleeve_b", qty=75, price=104.0, ltp=101.5, lot_size=75)),
                 ("lot size not known", dict(sleeve="sleeve_b", qty=75, price=104.0, ltp=101.5)),
                 ("70 for a 75-lot contract", dict(sleeve="sleeve_b", qty=70, price=104.0, ltp=101.5, lot_size=75)),
                 ("a sleeve with no caps", dict(sleeve="sleeve_x", qty=1, price=50.0, ltp=50.0))]
        print("2. Entries through the rails:")
        for label, kw in tries:
            ok, why = ks.check_order(led, kw.pop("sleeve"), kw.pop("qty"), **kw)
            print(f"   {label:<32}{'SENT' if ok else 'refused'}  ({why})")
        ks.HALT.write_text("x")
        for label, q, h in (("exit 75 of 75, HALT on", 75, 75), ("exit 150 of 75", 150, 75),
                            ("exit, positions unreadable", 75, None)):
            ok, why = ks.check_exit(q, h)
            print(f"   {label:<32}{'SENT' if ok else 'refused'}  ({why})")


def bad_day(policy: str, seed: int = 27) -> tuple[float, int, str]:
    """A seller that re-enters after every stop on a trending day. Returns the day's
    P&L, trades taken, and when the rail tripped."""
    rng = np.random.default_rng(seed)
    steps = 72                                          # five-minute steps, 09:20 to 15:20
    move = np.cumsum(rng.normal(9, 14, steps))          # the index trends all day
    realised, open_at, trades, tripped = 0.0, None, 0, "-"
    t0 = datetime(2026, 11, 4, 9, 20)
    with tempfile.TemporaryDirectory() as d:
        use_dir(d)
        for i in range(steps):
            now = t0 + timedelta(minutes=5 * i)
            mtm = 0.0
            if open_at is not None:
                mtm = 120 - 75 * abs(move[i] - move[open_at]) / 4   # premium in, delta-ish loss
                if mtm <= -1_500:                       # stop, filled 300 worse
                    realised += mtm - 300
                    open_at, mtm = None, 0.0
            pnl = {"realised": realised, "mtm": realised + mtm}.get(policy, None)
            if pnl is not None and ks.check_daily_loss({"sleeve_b": pnl}, now) and tripped == "-":
                tripped = f"{now:%H:%M}"
            if open_at is None and i < steps - 6 and ks.job_gate(False) is None \
                    and not (Path(d) / "KILL_sleeve_b").exists():
                open_at, trades = i, trades + 1
        if open_at is not None:
            realised += 120 - 75 * abs(move[-1] - move[open_at]) / 4
        if policy == "close":
            tripped = "15:30" if ks.check_daily_loss({"sleeve_b": realised}, t0.replace(hour=15, minute=30)) else "-"
    return realised, trades, tripped


def part3() -> None:
    print("3. A trending day, daily-loss cap 5,000 (sleeve_b):")
    for policy, label in (("close", "checked at the close"), ("realised", "realised, every run"),
                          ("mtm", "marked to market, every run")):
        pnl, n, t = bad_day(policy)
        print(f"   {label:<30} trades {n:>2}   day {pnl:>+9,.0f}   rail tripped {t}")


if __name__ == "__main__":
    part1()
    part2()
    part3()
