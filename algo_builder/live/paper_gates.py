"""
Book 2, Chapter 20 (and 29) — paper go/no-go gates, written down before the test.

Before a paper test starts, record the minimum number of trades before judging it,
the minimum average trade and the maximum drawdown that would end it. The gates are
saved with a fingerprint, so a quiet edit later is detected rather than believed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from algo_builder.infra.candle_cache import write_atomic_json


@dataclass(frozen=True)
class Gates:
    min_trades: int
    min_mean_return: float         # per trade, as a fraction
    max_drawdown: float            # as a positive fraction of the paper account


def register(path: str, gates: Gates, now: datetime) -> dict:
    """Write the gates once, with the time and a fingerprint. Refuses to overwrite."""
    if Path(path).exists():
        raise FileExistsError(f"gates already registered at {path}")
    body = asdict(gates)
    rec = dict(gates=body, registered_at=now.isoformat(timespec="seconds"),
               fingerprint=hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12])
    write_atomic_json(rec, path)
    return rec


def decide(path: str, ledger: pd.DataFrame, equity: pd.Series) -> str:
    """'tampered', 'no-go', 'keep testing' or 'go'. ledger: one row per closed paper
    trade with a `ret` column; equity: the paper account, marked to market daily."""
    rec = json.loads(Path(path).read_text())
    if hashlib.sha256(json.dumps(rec["gates"], sort_keys=True).encode()).hexdigest()[:12] != rec["fingerprint"]:
        return "tampered"
    g = Gates(**rec["gates"])
    dd = float(-(equity / equity.cummax() - 1).min()) if len(equity) else 0.0
    if dd > g.max_drawdown:
        return "no-go"                                # a drawdown gate ends the test at any time
    if len(ledger) < g.min_trades:
        return "keep testing"
    return "go" if ledger.ret.mean() >= g.min_mean_return else "no-go"
