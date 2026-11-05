"""
Book 2, Chapter 2 — a feed that replays stored bars as if they were arriving live.
Used to drive paper-mode code from history, and in tests.
"""
from __future__ import annotations

import pandas as pd

from algo_builder.common.broker_adapter import BrokerAdapter


class ReplayFeed(BrokerAdapter):
    """At clock time `now`, reveals only bars that have already started, and quotes
    the open of the bar starting at `now`. Set `now` before each call. It is a data
    source only: it refuses orders."""

    def __init__(self, bars: pd.DataFrame):
        self.bars, self.now = bars, None

    def get_candles(self, symbol, exchange, interval, start, end):
        b = self.bars
        return b[(b.index >= start) & (b.index < min(end, self.now))]

    def get_quote(self, symbol, exchange):
        row = self.bars[self.bars.index >= self.now].head(1)
        return None if row.empty else {"ltp": float(row.open.iloc[0])}

    def place_order(self, req):
        raise RuntimeError("a data feed never places orders")

    def order_status(self, order_id):
        return None

    def positions(self):
        return None
