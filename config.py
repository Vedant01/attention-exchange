"""
Attention Exchange — Global Configuration
==========================================
Central configuration for all simulation parameters, experiment sweeps,
and reproducibility controls.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SimulationConfig:
    """Master configuration for a single simulation run."""

    # --- Time & Scale ---
    num_timesteps: int = 1000
    num_replications: int = 30
    random_seed: int = 42

    # --- Agent Population ---
    num_producers: int = 50
    num_consumers: int = 200
    num_advertisers: int = 30
    num_speculators: int = 20  # Exchange model only

    # --- Attention Token Parameters ---
    initial_token_supply: int = 10000
    token_decay_rate: float = 0.005        # λ in e^(-λt)
    token_mint_rate: float = 0.1           # tokens minted per unit engagement
    min_token_value: float = 0.01          # floor value before token expires
    max_token_age: int = 200               # max lifetime in timesteps

    # --- Content Quality ---
    quality_distribution: str = "beta"     # "beta", "uniform", "normal"
    quality_alpha: float = 2.0             # Beta distribution shape α
    quality_beta: float = 5.0              # Beta distribution shape β (right-skewed)

    # --- Consumer Behavior ---
    consumer_attention_budget_mean: float = 10.0
    consumer_attention_budget_std: float = 3.0
    consumer_quality_sensitivity: float = 0.7   # weight on content quality
    consumer_price_sensitivity: float = 0.3     # weight on price

    # --- Producer Behavior ---
    producer_cost_mean: float = 1.0
    producer_cost_std: float = 0.5
    producer_content_rate: float = 1.0     # content pieces per timestep

    # --- Advertiser Behavior ---
    advertiser_budget_mean: float = 50.0
    advertiser_budget_std: float = 20.0
    advertiser_valuation_per_click: float = 0.5
    base_ctr: float = 0.02                 # base click-through rate

    # --- Exchange Model Parameters ---
    tick_size: float = 0.01                # minimum price increment
    position_limit: int = 500              # max tokens per agent (anti-manipulation)
    circuit_breaker_pct: float = 0.15      # halt trading if price moves > 15%
    market_maker_spread: float = 0.05      # MM bid-ask spread
    market_maker_depth: int = 50           # MM order quantity

    # --- Ad Model Parameters ---
    platform_cut: float = 0.30             # platform intermediary fee
    ad_auction_type: str = "second_price"  # "first_price" or "second_price"
    ad_slots_per_timestep: int = 20        # available impression slots
    quality_score_weight: float = 0.4      # Google-style quality factor

    # --- Welfare Metrics ---
    optimal_allocation_method: str = "hungarian"  # for computing allocative efficiency

    # --- Output ---
    output_dir: str = "results"
    figures_dir: str = "figures"
    save_trade_tape: bool = True
    verbose: bool = False


@dataclass
class ExperimentConfig:
    """Configuration for parameter sweep experiments."""

    name: str = ""
    description: str = ""
    base_config: SimulationConfig = field(default_factory=SimulationConfig)
    sweep_params: Dict[str, List] = field(default_factory=dict)


# ── Pre-defined experiment configurations ─────────────────────────────────

def get_experiment_configs() -> Dict[str, ExperimentConfig]:
    """Return all 5 experiment configurations."""

    configs = {}

    # Experiment 1: Market Efficiency Convergence
    configs["efficiency"] = ExperimentConfig(
        name="Market Efficiency Convergence",
        description="Compare price convergence speed and bid-ask spread dynamics",
        base_config=SimulationConfig(num_timesteps=1000, num_replications=30),
        sweep_params={"num_consumers": [50, 100, 200, 400]}
    )

    # Experiment 2: Welfare Comparison
    configs["welfare"] = ExperimentConfig(
        name="Welfare Comparison",
        description="Compare total surplus, consumer/producer surplus, and DWL",
        base_config=SimulationConfig(num_timesteps=500, num_replications=30),
        sweep_params={"platform_cut": [0.1, 0.2, 0.3, 0.4, 0.5]}
    )

    # Experiment 3: Price Discovery & Quality
    configs["price_discovery"] = ExperimentConfig(
        name="Price Discovery & Quality Correlation",
        description="Test if token prices reflect true content quality",
        base_config=SimulationConfig(num_timesteps=500, num_replications=30),
        sweep_params={"consumer_quality_sensitivity": [0.3, 0.5, 0.7, 0.9]}
    )

    # Experiment 4: Fairness & Gini
    configs["fairness"] = ExperimentConfig(
        name="Fairness & Attention Inequality",
        description="Compare Gini coefficients and bottom-quartile shares",
        base_config=SimulationConfig(num_timesteps=500, num_replications=30),
        sweep_params={"quality_alpha": [1.0, 2.0, 5.0]}  # Vary quality skew
    )

    # Experiment 5: Manipulation Resistance
    configs["manipulation"] = ExperimentConfig(
        name="Manipulation Resistance",
        description="Test AXP safeguards against wash trading and spoofing",
        base_config=SimulationConfig(num_timesteps=500, num_replications=30),
        sweep_params={
            "circuit_breaker_pct": [0.05, 0.10, 0.15, 0.25, 1.0],  # 1.0 = disabled
            "position_limit": [100, 250, 500, 10000]  # 10000 ≈ unlimited
        }
    )

    return configs


def set_global_seed(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)


# ── Utility ────────────────────────────────────────────────────────────────

def ensure_dirs(config: SimulationConfig):
    """Create output directories if they don't exist."""
    import os
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.figures_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "trade_tapes"), exist_ok=True)
