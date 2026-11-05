"""
Book 2, Chapter 1 — completed bars only (Listing 1.1). Illustrative.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd


def completed_bars(bars: pd.DataFrame, now: datetime, interval: str) -> pd.DataFrame:
    """Keep only bars whose period has ended by `now`.
    Assumes each bar is labelled with its START time, as most broker APIs do."""
    bar_end = bars.index + pd.Timedelta(interval)
    return bars[bar_end <= now]
