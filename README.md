# build-your-own-algo

Companion code for ***Build Your Own Algo Trading System: Designing, Testing and Running Systematic Strategies in Indian Markets*** (The Honest Backtest, Book 2) by Harish Kumar Krishnappa.

The book is about building, testing and running a trading system without fooling yourself. This repository holds every listing printed in the book, the tests that check them, and one runnable script per chapter.

> **Read this first.** This is educational code. It is not a trading system, not investment advice and not a recommendation to trade any instrument. It comes with no warranty of any kind (see `LICENSE`). Futures and options can lose more than the money deposited. If you connect any of this code to a real broker account, you are responsible for every order it places and for meeting SEBI's rules on retail algorithmic trading. Run everything in paper mode first.

## Five-minute start

You need Python 3.10 or later. No broker account or paid data is required: every example runs on an in-memory paper broker and on synthetic data the code generates.

```bash
git clone https://github.com/honest-backtest/build-your-own-algo.git
cd build-your-own-algo
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q                          # the whole test suite, a few seconds
python chapters/ch01/run.py        # Chapter 1: the report card as code
```

## How the code is organised

```
algo_builder/        the importable package
  common/               BrokerAdapter interface and PaperBroker, configuration, alerts,
                        completed bars, the Contract record
  infra/                candle cache, instrument-master download, exchange-file traps,
                        cost model, execution, state store, kill switches
  backtest/             the bar-by-bar engine and the report-card metrics
  robustness/           friction sweep, drift benchmark, fill-delay test, walk-forward,
                        by-year and ex-top-N
  strategies/           the core logic of each case study, by chapter
  engine/               one engine, three modes: intents, the decision step, backtest and live drivers
  data/                 synthetic data generators and a replay feed
adapters/               template for writing your own broker adapter
chapters/chNN/run.py    one runnable script per chapter
tests/                  pytest suite
tools/                  check_listings.py (book vs code), scan_secrets.py
```

## Chapter to code

| Chapter | Code |
|---|---|
| 1 Trading ideas as data | `backtest/metrics.py`, `common/bars.py`, `common/contract.py` |
| Front matter: what you need | `common/config_env.py` |
| 2 The seven parts of a trading system | `common/broker_adapter.py`, `engine/one_engine.py`, `data/replay.py` |
| 3 The data layer | `infra/candle_cache.py`, `infra/candle_checks.py`, `infra/instrument_master_lock.py`, `infra/ranged_download.py` |
| 4 Ten traps that quietly corrupt your data | `infra/option_bhavcopy_traps.py`, `infra/data_traps.py` |
| 5 A backtester that doesn't lie | `backtest/event_engine.py`, `backtest/sizing.py`, `infra/cost_model.py` |
| 6 Gross, net and the friction sweep | `robustness/friction_sweep.py`, `robustness/gross_net.py`, `strategies/rsi_macd_futures.py` |
| 7 Benchmarks that fight back | `robustness/drift_benchmark.py`, `robustness/benchmarks.py` |
| 8 Slicing the result | `robustness/by_year_and_ex_top_n.py`, `robustness/regime.py`, `strategies/rsi_macd_dipbuy.py` (`regime_tag`) |
| 9 The one-minute test | `robustness/fill_delay_test.py` |
| 10 Walk-forward | `robustness/walk_forward.py` |
| 11 The default report | `backtest/metrics.py` (Monte Carlo) |
| 12 Market Profile day types | `strategies/market_profile_framework.py`, `strategies/mcx_session_vwap.py` |
| 13 The opening-range breakout | `strategies/opening_range_options.py` |
| 14 Working with option contracts | `strategies/option_chart_momentum.py`, `strategies/supertrend_rsi.py` |
| 15 Multi-leg selling | `strategies/option_selling_adx_switch.py`, `strategies/expiry_mispricing.py` |
| 16 Expiry day | `strategies/expiry_straddle_framework.py` |
| 17 The covered-call overlay | `strategies/covered_call_framework.py` |
| 18 The DIY principal-protected note | `strategies/ppn_quarterly_call.py` |
| 19 One signal, three vehicles | `strategies/rsi_macd_dipbuy.py`, `strategies/hourly_signal_to_options.py` |
| 20 VCP and SEPA | `strategies/vcp_breakout.py`, `strategies/sepa_template_violations.py` |
| 21 The pullback in an uptrend | `strategies/pullback_in_uptrend.py` |
| 22 Buy-on-gap | `strategies/buy_on_gap.py` |
| 23 The compounder book | `strategies/compounder_screen.py` |
| 24 Design exercises | `lab/design_exercises.py` |
| 25 Post-mortems | `lab/post_mortems.py` (with `infra/cost_model.py`, `robustness/`) |
| 26 State and reconciliation (send once, morning check) | `infra/state_store.py`, `infra/execution.py` |
| 27 Rails and kill switches | `infra/kill_switches.py` |
| 28 Scheduling, logging and alerts | `common/jobs.py`, `common/notify.py` |
| 29 From backtest to paper to live | `live/reconcile.py`, `live/paper_gates.py`, `common/broker_adapter.py` (`PaperBroker`) |

Every chapter from 1 to 29 has a script in `chapters/`; Chapter 30 has none.

## Rules the code keeps

- **Broker access goes only through `BrokerAdapter`** (`get_candles`, `get_quote`, `place_order`, `order_status`, `positions`). No file imports a broker SDK. To trade with your broker, copy `adapters/template_adapter.py` and fill it in.
- **`None` means unknown, never "flat" or "gone".** An adapter that cannot read positions or an order book returns `None`, and the calling code treats that as "don't act yet".
- **Secrets come only from environment variables.** Copy `.env.example` to `.env` (which git ignores) and fill in your own values.
- **Numbers marked `ILLUSTRATIVE` are not anyone's live settings.** They exist so the code runs and to show the shape of a configuration.
- **Three systems are framework only:** the Market Profile day-type system, the expiry-day two-index rotation and the covered-call overlay. Their files contain the structure, not the rules, thresholds or schedules. You supply those.
- **No market data is included.** Exchange files have their own terms of use. The code reads files you download yourself, and synthetic data stands in for everything else.

## Keeping the book and the code in step

`tools/check_listings.py` compares every function and class printed in the book with the code here, and fails if any differ. The printed book matches the release tagged `v1.0-book`. Later fixes are listed in `ERRATA.md`.

## Contributing

Bug reports and corrections are welcome as issues. Please never paste API keys, account details, order IDs or screenshots of your broker account into an issue. Every commit is checked by `tools/scan_secrets.py`; install the hooks with `pre-commit install`.

## Licence

Code: MIT (see `LICENSE`). The book's text is not part of this repository and remains under the book's copyright.
