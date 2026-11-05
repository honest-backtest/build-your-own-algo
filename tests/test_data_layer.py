from datetime import date, datetime

import pandas as pd

from algo_builder.common.broker_adapter import PaperBroker
from algo_builder.infra import candle_cache as cc


class CountingBroker(PaperBroker):
    """Returns empty frames for the first `empties` calls (a throttled client), then data."""

    def __init__(self, empties: int):
        super().__init__()
        self.empties, self.calls = empties, 0

    def get_candles(self, symbol, exchange, interval, start, end):
        self.calls += 1
        if self.calls <= self.empties:
            return pd.DataFrame()
        idx = pd.date_range(start.replace(hour=9, minute=15), periods=3, freq="1min")
        return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1},
                            index=idx)


RANGE = (datetime(2026, 10, 5), datetime(2026, 10, 5, 15, 30))   # a closed Monday


def test_empty_answers_are_retried_not_cached(monkeypatch):
    monkeypatch.setattr(cc, "CACHE_DIR", cc.os.environ["ALGO_CACHE_DIR"])
    b = CountingBroker(empties=99)
    out = cc.get_candles(b, "IDX", "NSE", "minute", *RANGE, retries=2, pause=0)
    assert out.empty and b.calls == 2
    b2 = CountingBroker(empties=0)                       # next run: data is back
    out = cc.get_candles(b2, "IDX", "NSE", "minute", *RANGE, retries=2, pause=0)
    assert len(out) == 3 and b2.calls == 1


def test_known_holiday_is_cached_as_covered(monkeypatch):
    monkeypatch.setattr(cc, "CACHE_DIR", cc.os.environ["ALGO_CACHE_DIR"])
    hol = frozenset({date(2026, 10, 5)})
    cc.get_candles(CountingBroker(99), "IDX", "NSE", "minute", *RANGE, holidays=hol, retries=1, pause=0)
    b = CountingBroker(0)
    cc.get_candles(b, "IDX", "NSE", "minute", *RANGE, holidays=hol, retries=1, pause=0)
    assert b.calls == 0                                   # never re-requested


def test_closed_session_served_from_disk(monkeypatch):
    monkeypatch.setattr(cc, "CACHE_DIR", cc.os.environ["ALGO_CACHE_DIR"])
    cc.get_candles(CountingBroker(0), "IDX", "NSE", "minute", *RANGE, pause=0)
    b = CountingBroker(0)
    out = cc.get_candles(b, "IDX", "NSE", "minute", *RANGE, pause=0)
    assert b.calls == 0 and len(out) == 3


def test_session_closed_rule():
    # not final until the closing auction (15:30-15:40) and its publication are over
    assert not cc.session_closed(date(2026, 10, 5), now=datetime(2026, 10, 5, 15, 35))
    assert not cc.session_closed(date(2026, 10, 5), now=datetime(2026, 10, 5, 15, 59))
    assert cc.session_closed(date(2026, 10, 5), now=datetime(2026, 10, 5, 16, 0))
    assert cc.session_closed(date(2026, 10, 4), now=datetime(2026, 10, 5, 9, 15))


class TruncatingBroker(PaperBroker):
    """Returns bars only for the first day of any request: a silent truncation."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def get_candles(self, symbol, exchange, interval, start, end):
        self.calls += 1
        idx = pd.date_range(start.replace(hour=9, minute=15), periods=3, freq="1min")
        return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1},
                            index=idx)


def test_truncated_answer_does_not_cover_the_missing_days(monkeypatch):
    monkeypatch.setattr(cc, "CACHE_DIR", cc.os.environ["ALGO_CACHE_DIR"])
    week = (datetime(2026, 10, 5), datetime(2026, 10, 9, 15, 30))      # Mon..Fri, closed
    b = TruncatingBroker()
    cc.get_candles(b, "IDX", "NSE", "minute", *week, retries=1, pause=0)
    b2 = TruncatingBroker()
    cc.get_candles(b2, "IDX", "NSE", "minute", *week, retries=1, pause=0)
    assert b2.calls == 1                       # Tuesday..Friday asked for again, Monday not


def test_blocks_respect_the_per_request_limit():
    days = [date(2026, 1, 1) + cc.timedelta(n) for n in range(90)]
    blocks = cc._group_runs(days, max_days=30)
    assert all((b - a).days < 30 for a, b in blocks) and len(blocks) == 3


def test_bar_label_and_payload_checks():
    from algo_builder.data.synthetic import minute_bars, resample, trading_days
    from algo_builder.infra.candle_checks import infer_bar_label, validate_candles
    bars = resample(minute_bars(trading_days("2026-03-02", "2026-03-04")), "15min")
    assert infer_bar_label(bars, "15min") == "start"
    shifted = bars.copy()
    shifted.index = shifted.index + pd.Timedelta("15min")
    assert infer_bar_label(shifted, "15min") == "end"
    assert validate_candles(bars, "15min") == []
    assert validate_candles("<html>Page not found</html>", "15min")[0].startswith("not a table")
    bad = bars.copy()
    bad.iloc[3, bad.columns.get_loc("high")] = 1.0
    assert "high/low inconsistent with open/close" in validate_candles(bad, "15min")
