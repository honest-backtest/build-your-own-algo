"""
A skeleton for YOUR broker adapter (Book 2, Chapter 2 and Appendix D).

Copy this file, fill in each method with your broker's own client library, and
keep every broker-specific detail inside it. Strategy code never imports a
broker SDK; it only ever sees the BrokerAdapter interface.

Rules every adapter must keep:
  * Credentials come from environment variables (see algo_builder/common/config_env.py).
  * Return None when you could not read something. None means UNKNOWN, never
    "no position" or "order gone". Callers depend on that distinction.
  * Normalise order statuses to OPEN / COMPLETE / REJECTED / CANCELLED / UNKNOWN.
  * Return candles with a tz-naive DatetimeIndex labelled by bar START, and
    columns open/high/low/close/volume. Raise or retry on rate limits rather than
    returning an empty frame.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from algo_builder.common.broker_adapter import (BrokerAdapter, OrderRequest,
                                                   OrderStatus, Position)
from algo_builder.common.config_env import BROKER_API_KEY_ENV, require_env


class MyBrokerAdapter(BrokerAdapter):
    def __init__(self) -> None:
        self.api_key = require_env(BROKER_API_KEY_ENV)
        # self.client = <your broker's client>(api_key=self.api_key, ...)

    def get_candles(self, symbol: str, exchange: str, interval: str,
                    start: datetime, end: datetime) -> pd.DataFrame:
        raise NotImplementedError

    def get_quote(self, symbol: str, exchange: str) -> Optional[dict]:
        raise NotImplementedError

    def place_order(self, req: OrderRequest) -> Optional[str]:
        raise NotImplementedError

    def order_status(self, order_id: str) -> Optional[OrderStatus]:
        raise NotImplementedError

    def positions(self) -> Optional[list[Position]]:
        raise NotImplementedError

    # Optional extensions (Chapter 26). Add them only if your broker supports them;
    # without them the code takes the cautious path.
    #
    # def cancel_order(self, order_id: str) -> bool: ...
    #
    # def orders_with_tag(self, tag: str) -> Optional[list[OrderStatus]]:
    #     """Orders carrying `tag`: [] if none, None if the order book is unreadable."""
