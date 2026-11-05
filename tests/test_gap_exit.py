from datetime import date

from algo_builder.infra import state_store as ss
from algo_builder.strategies.buy_on_gap import log_row


def test_exit_job_waits_sells_only_holdings_and_never_stacks(monkeypatch, tmp_path):
    monkeypatch.setattr(ss, "STATE_DIR", str(tmp_path))
    from chapters.ch22.run import Afternoon
    today = date(2026, 11, 3)
    ss.save_state("gap", {today.isoformat(): dict(status="open", positions=[
        dict(symbol=s, exchange="NSE", qty=1) for s in ("A", "B", "C")])})
    b = Afternoon()
    runs = [ss.exit_tracked_longs(b, "gap", today, live=True) for _ in range(5)]
    assert runs[0].startswith("positions unreadable") and runs[3] == "closed"
    assert len(b.reqs) == 2 and {r.symbol for r in b.reqs.values()} == {"A", "C"}   # B was already flat


def test_dry_run_changes_no_status(monkeypatch, tmp_path):
    monkeypatch.setattr(ss, "STATE_DIR", str(tmp_path))
    from algo_builder.common.broker_adapter import PaperBroker, OrderRequest
    b = PaperBroker()
    b.set_quote("A", "NSE", 10.0)
    b.place_order(OrderRequest("A", "NSE", "BUY", 1))
    today = date(2026, 11, 3)
    ss.save_state("gap", {today.isoformat(): dict(status="open", positions=[dict(symbol="A", exchange="NSE", qty=1)])})
    ss.exit_tracked_longs(b, "gap", today, live=False)
    assert ss.load_state("gap")[today.isoformat()]["status"] == "open" and b.positions()[0].net_qty == 1
    assert round(log_row("A", 100.0, 100.34, None)["slip_pct"], 2) == 0.34
