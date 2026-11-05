"""
Chapter 2 — The seven parts of a trading system.

Runs the same signal twice on synthetic data: once as a backtest, and once as a
paper run in which a scheduler's clock steps through the week and each cycle
fetches candles and quotes through the BrokerAdapter interface. The two produce
the same trades, because they run the same decision code.

    python chapters/ch02/run.py
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("ALGO_CACHE_DIR", tempfile.mkdtemp(prefix="ch02-cache-"))

from algo_builder.common.broker_adapter import PaperBroker  # noqa: E402
from algo_builder.data.replay import ReplayFeed  # noqa: E402
from algo_builder.data.synthetic import minute_bars, resample, trading_days  # noqa: E402
from algo_builder.engine.one_engine import backtest, crossover_signal, cycle  # noqa: E402

SYMBOL, EXCHANGE, LOT, INTERVAL = "IDX-FUT", "NFO", 50, "15min"


def main() -> None:
    days = trading_days("2026-03-02", "2026-03-06")             # an illustrative week
    bars = resample(minute_bars(days, seed=3), INTERVAL)

    bt = backtest(bars, crossover_signal, SYMBOL, EXCHANGE, LOT)

    feed, paper, state, pt = ReplayFeed(bars), PaperBroker(), {}, []
    for ts in bars.index[1:]:                                     # the scheduler's clock
        feed.now = ts.to_pydatetime()
        pt += cycle(feed, paper, state, crossover_signal, SYMBOL, EXCHANGE, INTERVAL, LOT,
                    now=feed.now, lookback=ts - bars.index[0])    # never ask before the data starts

    key = lambda f: (f["ts"], f["action"], f["fill_px"], f["qty"])
    print(f"backtest fills: {len(bt)}   paper fills: {len(pt)}   "
          f"identical: {[key(f) for f in bt] == [key(f) for f in pt]}")


if __name__ == "__main__":
    main()
