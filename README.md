# Attention Exchange

**Trading Attention as a Liquid Asset via Continuous Double Auction**

A novel market mechanism (AXP — Attention Exchange Protocol) that tokenizes user attention as a fungible, decaying digital asset and trades it via a continuous double auction. This repository contains the complete agent-based simulation framework and research paper.

## Key Findings

| Metric | Exchange (μ ± σ) | Ad-Driven (μ ± σ) | Cohen's d | p-value |
|---|---|---|---|---|
| Total Surplus | 28.90 ± 8.77 | 1967.42 ± 53.53 | −50.54 | < 0.001 |
| Producer Surplus | **+13.74 ± 7.68** | **−368.91 ± 26.63** | +19.53 | < 0.001 |
| Consumer Surplus | 15.16 ± 3.96 | 382.80 ± 45.22 | −11.45 | < 0.001 |

> The ad-driven model systematically extracts value from content creators (negative producer surplus), while the Attention Exchange ensures positive surplus for all participant types.

## Project Structure

```
attention-exchange/
├── config.py              # Simulation parameters & experiment configs
├── attention_token.py     # Token class with exponential decay
├── order_book.py          # CDA order book (price-time priority)
├── agents.py              # 5 agent types × 3 strategies (ZI, ZIP, RL)
├── exchange_model.py      # AXP protocol implementation
├── ad_model.py            # Ad-driven baseline (Vickrey auctions)
├── welfare.py             # Welfare metrics engine
├── visualizations.py      # Publication-quality figure generation
├── run_all.py             # Master experiment runner
├── figures/               # Generated figures (7 PNG @ 300 DPI)
├── results/               # Experiment results (JSON)
└── paper/                 # Research paper (LaTeX)
```

## Quick Start

```bash
pip install numpy scipy matplotlib seaborn pandas
python run_all.py
```

Runs 5 experiments × 30 replications (~25 min). Results saved to `results/`, figures to `figures/`.

## Experiments

1. **Market Efficiency** — CDA price convergence within 200 timesteps
2. **Welfare Comparison** — Exchange vs Ad-driven surplus distribution
3. **Price Discovery** — Token prices reflect content quality
4. **Fairness** — Gini coefficient analysis (quality-driven vs compressed)
5. **Manipulation Resistance** — Circuit breakers & position limits

## Authors

- Vedant Agarwal
- Shreyas B S
- Dhruv Agrawal

## License

MIT
