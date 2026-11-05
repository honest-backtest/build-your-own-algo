"""
Chapter 3 — The data layer.

Shows the candle cache's rules on synthetic data: closed sessions come from disk
on the second call; a throttled broker's empty answers are retried, never cached;
a truncated answer leaves its missing days to be fetched again; and two checks
run on every new source before any strategy reads it.

    python chapters/ch03/run.py
"""
from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime

import pandas as pd

os.environ.setdefault("ALGO_CACHE_DIR", tempfile.mkdtemp(prefix="ch03-cache-"))
logging.basicConfig(level=logging.WARNING, format="  log: %(message)s")

from algo_builder.common.broker_adapter import PaperBroker  # noqa: E402
from algo_builder.data.synthetic import minute_bars, resample, trading_days  # noqa: E402
from algo_builder.infra import candle_cache as cc  # noqa: E402
from algo_builder.infra.candle_checks import infer_bar_label, validate_candles  # noqa: E402

BARS = resample(minute_bars(trading_days("2026-03-02", "2026-03-06"), seed=5), "15min")
WEEK = (datetime(2026, 3, 2), datetime(2026, 3, 6, 15, 30))
NOW = datetime(2026, 3, 9, 10, 0)                       # the following Monday


class HistoryBroker(PaperBroker):
    """Serves BARS. Optionally answers the first `empties` calls with an empty frame
    (a throttled client) or returns only the first day of every request (truncation)."""

    def __init__(self, empties: int = 0, truncate: bool = False):
        super().__init__()
        self.empties, self.truncate, self.calls = empties, truncate, 0

    def get_candles(self, symbol, exchange, interval, start, end):
        self.calls += 1
        if self.calls <= self.empties:
            return pd.DataFrame()
        out = BARS[(BARS.index >= start) & (BARS.index <= end)]
        return out[out.index.normalize() == out.index[0].normalize()] if self.truncate and len(out) else out


def main() -> None:
    print("1. First call fetches; second call is served from disk")
    b = HistoryBroker()
    first = cc.get_candles(b, "IDX", "NSE", "15min", *WEEK, now=NOW, pause=0)
    second = cc.get_candles(b, "IDX", "NSE", "15min", *WEEK, now=NOW, pause=0)
    print(f"   bars: {len(first)} then {len(second)}; broker calls: {b.calls}")

    print("2. A throttled broker answers empty twice, then with data")
    cc.CACHE_DIR = tempfile.mkdtemp(prefix="ch03-b-")
    b = HistoryBroker(empties=2)
    got = cc.get_candles(b, "IDX", "NSE", "15min", *WEEK, now=NOW, retries=3, pause=0)
    print(f"   bars: {len(got)} after {b.calls} calls (two empties retried, not cached)")

    print("3. A truncated answer returns only Monday")
    cc.CACHE_DIR = tempfile.mkdtemp(prefix="ch03-c-")
    cc.get_candles(HistoryBroker(truncate=True), "IDX", "NSE", "15min", *WEEK, now=NOW, retries=1, pause=0)
    b = HistoryBroker()
    full = cc.get_candles(b, "IDX", "NSE", "15min", *WEEK, now=NOW, pause=0)
    print(f"   next run asks again for the missing days: {b.calls} call, {len(full)} bars in all")

    print("4. Checks on a new source")
    shifted = BARS.copy()
    shifted.index = shifted.index + pd.Timedelta("15min")
    print(f"   bar label: {infer_bar_label(BARS, '15min')} (ours), {infer_bar_label(shifted, '15min')} (shifted copy)")
    print(f"   validate candles: {validate_candles(BARS, '15min') or 'ok'}")
    print(f"   validate a web page: {validate_candles('<html>Page not found</html>', '15min')}")


if __name__ == "__main__":
    main()
