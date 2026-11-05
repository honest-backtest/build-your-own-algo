from datetime import datetime
from functools import partial

import pandas as pd

from algo_builder.common.broker_adapter import PaperBroker
from algo_builder.data.replay import ReplayFeed
from algo_builder.data.synthetic import minute_bars, resample, trading_days
from algo_builder.engine.one_engine import Intent, backtest, crossover_signal, cycle, step
from algo_builder.infra import candle_cache as cc


def _bars():
    return resample(minute_bars(trading_days("2026-03-02", "2026-03-06"), seed=3), "15min")


def test_backtest_and_paper_replay_trade_identically(monkeypatch):
    monkeypatch.setattr(cc, "CACHE_DIR", cc.os.environ["ALGO_CACHE_DIR"])
    bars = _bars()
    sig = partial(crossover_signal, symbol="F", exchange="NFO")
    bt = backtest(bars, sig, "F", "NFO", 50)
    feed, paper, state, pt = ReplayFeed(bars), PaperBroker(), {}, []
    for ts in bars.index[1:]:
        feed.now = ts.to_pydatetime()
        pt += cycle(feed, paper, state, sig, "F", "NFO", "15min", 50,
                    now=feed.now, lookback=ts - bars.index[0])
    assert len(bt) >= 2 and bt == pt


def test_signal_is_pure():
    bars = _bars().iloc[:200]
    before = bars.copy()
    assert crossover_signal(bars, 0) == crossover_signal(bars, 0)
    pd.testing.assert_frame_equal(bars, before)


def test_fill_comes_from_the_bar_after_the_decision():
    bars = _bars()
    fills = backtest(bars, partial(crossover_signal, symbol="F"), "F", "NFO", 50)
    assert fills
    for f in fills:
        nxt = bars.index[bars.index.get_loc(f["ts"]) + 1]
        assert f["fill_px"] == bars.open.loc[nxt]


def test_unconfirmed_order_leaves_state_unchanged():
    class Blind(PaperBroker):
        def order_status(self, order_id):
            return None

    b, state = Blind(), {"lots": 0}
    b.set_quote("F", "NFO", 100.0)
    buy = lambda bars, lots: [Intent("BUY", "F", "NFO", 1, "test", pd.Timestamp("2026-03-02"), 100.0)]
    assert step(pd.DataFrame(), state, buy, b, 50) == [] and state["lots"] == 0


def test_cache_uses_the_injected_clock_not_the_wall_clock(monkeypatch):
    monkeypatch.setattr(cc, "CACHE_DIR", cc.os.environ["ALGO_CACHE_DIR"])
    bars = _bars()
    feed = ReplayFeed(bars)
    feed.now = datetime(2026, 3, 2, 10, 0)                       # Monday, mid-morning
    cc.get_candles(feed, "F", "NFO", "15min", datetime(2026, 3, 2), feed.now, now=feed.now, pause=0)
    feed.now = datetime(2026, 3, 3, 10, 0)                       # next day: Monday is now closed
    out = cc.get_candles(feed, "F", "NFO", "15min", datetime(2026, 3, 2), feed.now, now=feed.now, pause=0)
    assert len(out[out.index.normalize() == "2026-03-02"]) == 25     # the full day, not 3 bars
