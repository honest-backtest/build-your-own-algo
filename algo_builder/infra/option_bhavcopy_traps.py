"""
Book 2, Chapter 4 — Ten traps that quietly corrupt your data.
Also used in Chapters 16 (expiry day) and 18 (DIY principal-protected note).

Daily exchange option files (bhavcopies) are the cheapest source of REAL option
prices — and they contain traps that silently wreck a backtest:

  * On expiry day some files overwrite an option's close (and settle) with the
    UNDERLYING's settlement value. A 3-rupee option suddenly "closes" at 72,000.
    Detect the signature (close == underlying) instead of trusting either exchange.
  * Settlement can be recovered from the option panel itself via put-call parity.
  * Most strikes barely trade: a liquidity gate matters more than any signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_PAIRS = 5     # minimum CE/PE strike pairs for a parity estimate (illustrative)


def leg_exit_price(close: float, underlying: float, strike: float, is_call: bool) -> float:
    """Closing value of one leg, working around the expiry-day overwrite.
    If close == underlying, the row was overwritten: use intrinsic vs settlement."""
    if np.isfinite(underlying) and abs(close - underlying) < 0.01:
        return max(0.0, underlying - strike) if is_call else max(0.0, strike - underlying)
    return float(close)


def settlement_from_parity(chain: pd.DataFrame) -> float:
    """S ~ median(K + C - P) across strikes, preferring strikes where both legs traded.
    Pairs whose call and put show the SAME close are skipped: that is the expiry-day
    overwrite's signature, and K + S - S would return the strike, not the index.
    chain columns: strike, opt_type ('CE'/'PE'), close, contracts."""
    ce = chain[chain.opt_type == "CE"].drop_duplicates("strike").set_index("strike")
    pe = chain[chain.opt_type == "PE"].drop_duplicates("strike").set_index("strike")
    k = ce.index.intersection(pe.index)
    c, p = ce.loc[k, "close"].to_numpy(float), pe.loc[k, "close"].to_numpy(float)
    honest = ~np.isclose(c, p, rtol=0.0, atol=0.01)
    k, c, p = k[honest], c[honest], p[honest]
    if len(k) < MIN_PAIRS:
        return np.nan
    est = k.to_numpy(float) + c - p
    traded = ((ce.loc[k, "contracts"] > 0) & (pe.loc[k, "contracts"] > 0)).to_numpy()
    use = est[traded] if traded.sum() >= MIN_PAIRS else est
    use = use[np.isfinite(use) & (use > 0)]
    return float(np.median(use)) if len(use) >= MIN_PAIRS else np.nan


def liquid_strikes(chain: pd.DataFrame, min_contracts: int, min_premium: float) -> pd.DataFrame:
    """Keep only strikes you could realistically have traded that day.
    Thresholds are the caller's choice — set them from your own fill experience."""
    return chain[(chain.contracts >= min_contracts) & (chain.close >= min_premium)]
