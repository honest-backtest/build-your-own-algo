"""
Book 2, Chapter 3 — The data layer ("cache closed sessions forever, never today").

The single most important cache rule: a session is cacheable only once it can
never produce another bar. Serving a stale partial bar from disk to a live loop
is a CORRECTNESS bug, not a performance one. Today is always fetched live.

Three more rules, each learned from data that went quietly missing:
  * an EMPTY answer is not evidence of a holiday: some broker clients return an
    empty frame, not an error, when rate-limited, so retry before believing it;
  * mark a day covered only on evidence for THAT day: it has bars, or it is in
    the exchange holiday calendar you supply. A truncated answer to a long
    request then leaves its missing days to be fetched again;
  * write every file via a temporary file and os.replace, so a crash never leaves
    a half-written file for the next reader.

Session times below are for NSE/BSE equity and F&O. Commodity sessions differ
(Chapter 12).
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import date, datetime, time as dtime, timedelta

import pandas as pd

from algo_builder.common.broker_adapter import BrokerAdapter

log = logging.getLogger("candle_cache")

MARKET_CLOSE = dtime(15, 30)
CACHE_DIR = os.environ.get("ALGO_CACHE_DIR", "./cache")


def session_closed(d: date, now: datetime | None = None) -> bool:
    now = now or datetime.now()
    return d < now.date() or (d == now.date() and now.time() >= MARKET_CLOSE)


def write_atomic_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_atomic_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _group_runs(days: list[date], gap: int = 3, max_days: int = 30) -> list[tuple[date, date]]:
    """Collapse missing days into contiguous blocks (fewer API calls), but never
    let one block span more calendar days than the broker allows per request."""
    blocks, start, prev = [], days[0], days[0]
    for d in days[1:]:
        if (d - prev).days > gap or (d - start).days >= max_days:
            blocks.append((start, prev))
            start = d
        prev = d
    blocks.append((start, prev))
    return blocks


def _fetch_with_retry(broker: BrokerAdapter, symbol: str, exchange: str, interval: str,
                      a: date, b: date, retries: int, pause: float) -> pd.DataFrame:
    """Some broker clients return an EMPTY frame, not an error, when rate-limited.
    Treat empty as 'maybe throttled' and retry before believing it."""
    for attempt in range(retries):
        df = broker.get_candles(symbol, exchange, interval,
                                datetime.combine(a, dtime(9, 0)),
                                datetime.combine(b, dtime(15, 45)))
        if df is not None and len(df):
            return df
        time.sleep(pause * (2 ** attempt))
    return pd.DataFrame()


def get_candles(broker: BrokerAdapter, symbol: str, exchange: str, interval: str,
                frm: datetime, to: datetime, holidays: frozenset[date] = frozenset(),
                retries: int = 3, pause: float = 1.0, max_days: int = 30,
                now: datetime | None = None) -> pd.DataFrame:
    """Candles for [frm, to], from disk where a session is closed and covered, from
    the broker otherwise. `holidays` is the exchange holiday calendar (you supply
    it); `max_days` is your broker's per-request limit for this interval; `now` is
    the clock: pass it explicitly in replays and tests."""
    key = f"{exchange}_{symbol}_{interval}".replace("/", "_")
    ppath = os.path.join(CACHE_DIR, key + ".parquet")
    cpath = os.path.join(CACHE_DIR, key + ".covered.json")

    cached = pd.read_parquet(ppath) if os.path.exists(ppath) else None
    covered: set[str] = set()
    if os.path.exists(cpath):
        with open(cpath) as f:
            covered = set(json.load(f))

    days = [frm.date() + timedelta(n) for n in range((to.date() - frm.date()).days + 1)]
    closed = [d for d in days if session_closed(d, now)]
    live = [d for d in days if not session_closed(d, now)]

    missing = [d for d in closed if d.isoformat() not in covered and d.weekday() < 5]
    if missing:
        got = []
        for a, b in _group_runs(missing, max_days=max_days):
            df = _fetch_with_retry(broker, symbol, exchange, interval, a, b, retries, pause)
            have = set(df.index.date) if len(df) else set()
            block = [d for d in missing if a <= d <= b]
            covered |= {d.isoformat() for d in block if d in have or d in holidays}
            unexplained = [d for d in block if d not in have and d not in holidays]
            if unexplained:
                log.warning("%s: no bars for %d day(s) from %s; will refetch next run",
                            key, len(unexplained), unexplained[0])
            if len(df):
                got.append(df)
        if got:
            cached = pd.concat(([cached] if cached is not None else []) + got)
            cached = cached[~cached.index.duplicated(keep="last")].sort_index()
            write_atomic_parquet(cached, ppath)
        write_atomic_json(sorted(covered), cpath)

    out = cached.loc[frm:to] if cached is not None else pd.DataFrame()
    if live:                                                  # never served from disk
        fresh = broker.get_candles(symbol, exchange, interval,
                                   max(frm, datetime.combine(min(live), dtime(9, 0))), to)
        out = pd.concat([out, fresh])
        out = out[~out.index.duplicated(keep="last")].sort_index()
    return out
