# MIDAS 🪙

**M**ulti-agent **I**nvestment **D**iversification and **A**llocation **S**ystem

A hierarchical multi-agent deep reinforcement learning framework for portfolio allocation, benchmarked against traditional diversification techniques — Markowitz mean-variance optimisation and the equal-weight (1/N) portfolio — across a diversified universe of S&P 100 equities, bond and commodity ETFs, and REITs.

MSc dissertation project, AI and Machine Learning in Sciences (QMUL).

## Motivation

Classical mean-variance optimisation (Markowitz, 1952) relies on expected returns and covariances estimated from historical data, which makes it an "estimation-error maximiser" (Michaud, 1989): unstable, concentrated portfolios that often fail to beat even the naive equal-weight portfolio out of sample (DeMiguel et al., 2009). Reinforcement learning instead treats allocation as a sequential decision problem — an agent learns an allocation policy directly from market data and can adapt as conditions change. MIDAS asks whether a hierarchical multi-agent RL approach can outperform these traditional techniques.

## Architecture

MIDAS is organised as a layered pipeline running from raw data to executed trades:

```
Layer 0 · Data collection   →  Collect and format raw inputs
Layer 1 · Experts           →  Condense data into signals ──┐
Layer 2 · Class agents      →  RL agent + VSN per class     │ raw expert signals
Layer 3 · Aggregation       →  Merge signals, constraints ◄─┘ bypass to aggregation
Layer 4 · Manager           →  Allocate across asset classes
Layer 5 · Constraints       →  Apply hard user limits
Layer 6 · Execution         →  Execute final allocation
```

- **Layer 0 — Data collection.** Gathers and formats daily OHLCV prices, fundamentals, news text and macroeconomic series from the sources listed below. Large-scale preprocessing (e.g. the 15.7M-record FNSPID news corpus) is handled with PySpark; downstream series work uses pandas/NumPy.
- **Layer 1 — Expert signals.** Condenses raw data into three groups of signals:
  - *Technical and statistical indicators* — trend and momentum (3/6/12-month momentum, moving-average crossovers, MACD, ADX), mean reversion (RSI, stochastic, Bollinger bands), risk and volatility (realised and EWMA volatility, HAR-RV, maximum drawdown, downside deviation, beta), valuation (P/E, P/B, EV/EBITDA, dividend yield, P/FFO for REITs), factor exposures (size, value, quality, momentum) and liquidity (Amihud illiquidity, dollar volume).
  - *Model-based signals* — financial-text sentiment from **FinBERT**, a **graph neural network** (PyTorch Geometric) capturing relationships between assets, and a **hidden Markov model** producing market-regime probabilities.
  - *Raw external inputs* — macroeconomic series (interest rates, CPI, GDP, unemployment, VIX) and raw fundamentals (earnings, book value) passed to the agents without transformation.
- **Layer 2 — Per-class RL agents.** One agent per asset class (equities, bond ETFs, commodity ETFs, REITs), each preceded by a **variable selection network (VSN)** that weights incoming features. Each agent assigns continuous weights within its class.
- **Layer 3 — Aggregation.** Merges the class agents' outputs with the raw expert signals (routed directly from Layer 1) and any market constraints.
- **Layer 4 — Manager.** A higher-level agent that sets the capital allocation across asset classes.
- **Layer 5 — Constraints.** Applies the investor's hard limits to the proposed allocation.
- **Layer 6 — Execution.** Carries out the resulting trades.

Decomposing the problem this way reduces the action space each agent must learn and improves stability over a large asset universe, following recent hierarchical frameworks (Rahimi et al., 2025; Coriat & Benhamou, 2025 — HARLF; Ren et al., 2025 — SAMP-HDRL). MIDAS differs in that its lower-level agents assign continuous weights within classes defined by asset type rather than clustering, and draws on a broader pool of expert signals.

## Models and training

| Component | Model / method |
|---|---|
| Per-class agents & manager | **SAC** (Soft Actor-Critic), trained with Stable-Baselines3 on PyTorch |
| Feature weighting | Variable selection network per class agent |
| Sentiment | FinBERT (Araci, 2019) |
| Asset relationships | Graph neural network (PyTorch Geometric) |
| Market regimes | Hidden Markov model |

## Benchmarks

- **Markowitz mean-variance portfolio**
- **Equal-weight (1/N) portfolio**

## Data sources

| Data | Source | Coverage |
|---|---|---|
| Daily OHLCV — S&P 100 equities, bond/commodity/REIT ETFs | yfinance, Stooq | Several decades |
| Fundamentals & filing text | SEC EDGAR | From 1993 |
| News & sentiment | Alpha Vantage | From 2022 (articles + sentiment scores) |
| Historical news | FNSPID (Dong et al., 2024) | 15.7M records, 1999–2023 |
| Additional news coverage | Finnhub, NewsAPI, Tiingo | — |
| Macro (T-bill, rates, CPI, GDP, unemployment, VIX) | FRED / ALFRED | As-announced vintages via ALFRED |

## Evaluation

Cumulative and annualised return, Sharpe and Sortino ratios, maximum drawdown, volatility and turnover, with statistical testing of return differences against the benchmarks.

## Stack

Python · PyTorch · Stable-Baselines3 · PyTorch Geometric · Transformers (FinBERT) · PySpark · pandas · NumPy · matplotlib · Docker

## Setup

### Local

```bash
git clone https://github.com/YOUR_USERNAME/midas.git
cd midas
pip install -r requirements.txt
```

### Docker (recommended)

The full environment — Python, PyTorch, Stable-Baselines3, PySpark and the Java runtime Spark requires — is containerised so results are reproducible on any machine.

```bash
docker compose build
docker compose run midas            # interactive shell inside the container
```

Data, models and results are mounted as volumes, so outputs persist on the host.

API keys (Alpha Vantage, Finnhub, NewsAPI, Tiingo) go in a `.env` file — never committed. Docker Compose loads it automatically.

## Related work

MIDAS builds on recent hierarchical RL portfolio frameworks:
[HARLF](https://arxiv.org/abs/2507.18560) (Coriat & Benhamou, 2025),
[Rahimi et al., 2025](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5598002) and
[SAMP-HDRL](https://arxiv.org/abs/2512.22895) (Ren et al., 2025),
with sentiment via [FinBERT](https://arxiv.org/abs/1908.10063) (Araci, 2019).
