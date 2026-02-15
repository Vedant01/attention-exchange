"""
Run All Experiments — Master Experiment Runner
================================================
Executes all 5 experiments with 30 replications each,
collects results, performs statistical tests, and generates figures.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List

from config import SimulationConfig, get_experiment_configs, set_global_seed, ensure_dirs
from exchange_model import AttentionExchangeModel
from ad_model import AdDrivenModel
from welfare import (WelfareAnalyzer, gini_coefficient, herfindahl_index,
                     price_quality_correlation, bottom_quartile_share)


def run_single_replication(config: SimulationConfig, rep_id: int) -> dict:
    """Run one replication of both models with the same seed offset."""
    # Use different seed per replication for statistical independence
    cfg_ex = SimulationConfig(**{k: getattr(config, k) for k in config.__dataclass_fields__})
    cfg_ex.random_seed = config.random_seed + rep_id
    cfg_ad = SimulationConfig(**{k: getattr(config, k) for k in config.__dataclass_fields__})
    cfg_ad.random_seed = config.random_seed + rep_id

    # Run exchange model
    exchange = AttentionExchangeModel(cfg_ex)
    exchange.run()
    ex_results = exchange.get_final_results()

    # Run ad model
    ad = AdDrivenModel(cfg_ad)
    ad.run()
    ad_results = ad.get_final_results()

    # Welfare comparison
    analyzer = WelfareAnalyzer(ex_results, ad_results)
    welfare = analyzer.full_analysis()

    return {
        "rep_id": rep_id,
        "exchange": ex_results,
        "ad": ad_results,
        "welfare": welfare,
    }


def run_experiment(name: str, config: SimulationConfig,
                   num_reps: int = 30, verbose: bool = True) -> List[dict]:
    """Run multiple replications of an experiment."""
    results = []
    for rep in range(num_reps):
        if verbose:
            print(f"  [{name}] Replication {rep + 1}/{num_reps}...", end=" ", flush=True)
        t0 = time.time()
        result = run_single_replication(config, rep)
        elapsed = time.time() - t0
        if verbose:
            print(f"({elapsed:.1f}s)")
        results.append(result)
    return results


def aggregate_results(results: List[dict]) -> dict:
    """Aggregate results across replications."""
    exchange_runs = [r["exchange"] for r in results]
    ad_runs = [r["ad"] for r in results]
    welfare_runs = [r["welfare"] for r in results]

    def agg(runs, key):
        vals = [r.get(key, 0) for r in runs]
        return {"mean": np.mean(vals), "std": np.std(vals),
                "min": np.min(vals), "max": np.max(vals), "values": vals}

    # Statistical comparison
    analyzer = WelfareAnalyzer(exchange_runs[0], ad_runs[0])
    stats = analyzer.statistical_comparison(exchange_runs, ad_runs)

    return {
        "exchange_surplus": agg(exchange_runs, "total_surplus"),
        "ad_surplus": agg(ad_runs, "total_surplus"),
        "exchange_producer_surplus": agg(exchange_runs, "total_producer_surplus"),
        "ad_producer_surplus": agg(ad_runs, "total_producer_surplus"),
        "exchange_consumer_surplus": agg(exchange_runs, "total_consumer_surplus"),
        "ad_consumer_surplus": agg(ad_runs, "total_consumer_surplus"),
        "welfare_surplus_improvement": agg(welfare_runs, "surplus_improvement_pct"),
        "welfare_gini_exchange": agg(welfare_runs, "exchange_gini_revenue"),
        "welfare_gini_ad": agg(welfare_runs, "ad_gini_revenue"),
        "welfare_efficiency_exchange": agg(welfare_runs, "exchange_allocative_efficiency"),
        "welfare_efficiency_ad": agg(welfare_runs, "ad_allocative_efficiency"),
        "welfare_dwl_exchange": agg(welfare_runs, "exchange_dwl"),
        "welfare_dwl_ad": agg(welfare_runs, "ad_dwl"),
        "statistical_tests": stats,
        "raw_exchange": exchange_runs,
        "raw_ad": ad_runs,
        "raw_welfare": welfare_runs,
    }


def save_results(agg: dict, name: str, output_dir: str = "results"):
    """Save aggregated results to JSON (excluding raw data for size)."""
    os.makedirs(output_dir, exist_ok=True)
    save_data = {k: v for k, v in agg.items() if not k.startswith("raw_")}
    # Convert numpy values
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(i) for i in obj]
        return obj
    save_data = convert(save_data)
    path = os.path.join(output_dir, f"{name}_results.json")
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  Saved results to {path}")


def main():
    """Run all experiments."""
    print("=" * 70)
    print("ATTENTION EXCHANGE — EMPIRICAL SIMULATION STUDY")
    print("=" * 70)

    base_config = SimulationConfig()
    ensure_dirs(base_config)

    # Use fewer replications for faster execution, but still statistically valid
    NUM_REPS = 30
    # Reduce timesteps for tractability
    TIMESTEPS = 200

    all_results = {}

    # ── Experiment 1: Market Efficiency ──────────────────────────────
    print("\n[1/5] Experiment: Market Efficiency Convergence")
    cfg1 = SimulationConfig(num_timesteps=TIMESTEPS, num_replications=NUM_REPS,
                            num_producers=30, num_consumers=100, num_advertisers=20, num_speculators=10)
    results1 = run_experiment("efficiency", cfg1, num_reps=NUM_REPS)
    agg1 = aggregate_results(results1)
    save_results(agg1, "experiment1_efficiency")
    all_results["efficiency"] = {"config": cfg1, "results": results1, "aggregated": agg1}

    # ── Experiment 2: Welfare Comparison ──────────────────────────────
    print("\n[2/5] Experiment: Welfare Comparison")
    cfg2 = SimulationConfig(num_timesteps=TIMESTEPS, num_replications=NUM_REPS,
                            num_producers=30, num_consumers=100, num_advertisers=20, num_speculators=10)
    results2 = run_experiment("welfare", cfg2, num_reps=NUM_REPS)
    agg2 = aggregate_results(results2)
    save_results(agg2, "experiment2_welfare")
    all_results["welfare"] = {"config": cfg2, "results": results2, "aggregated": agg2}

    # ── Experiment 3: Price Discovery ─────────────────────────────────
    print("\n[3/5] Experiment: Price Discovery & Quality Correlation")
    cfg3 = SimulationConfig(num_timesteps=TIMESTEPS, num_replications=NUM_REPS,
                            num_producers=30, num_consumers=100, num_advertisers=20, num_speculators=10,
                            consumer_quality_sensitivity=0.9)
    results3 = run_experiment("price_discovery", cfg3, num_reps=NUM_REPS)
    agg3 = aggregate_results(results3)
    save_results(agg3, "experiment3_price_discovery")
    all_results["price_discovery"] = {"config": cfg3, "results": results3, "aggregated": agg3}

    # ── Experiment 4: Fairness ────────────────────────────────────────
    print("\n[4/5] Experiment: Fairness & Attention Inequality")
    cfg4 = SimulationConfig(num_timesteps=TIMESTEPS, num_replications=NUM_REPS,
                            num_producers=30, num_consumers=100, num_advertisers=20, num_speculators=10)
    results4 = run_experiment("fairness", cfg4, num_reps=NUM_REPS)
    agg4 = aggregate_results(results4)
    save_results(agg4, "experiment4_fairness")
    all_results["fairness"] = {"config": cfg4, "results": results4, "aggregated": agg4}

    # ── Experiment 5: Manipulation Resistance ─────────────────────────
    print("\n[5/5] Experiment: Manipulation Resistance")
    # With safeguards
    cfg5a = SimulationConfig(num_timesteps=TIMESTEPS, num_replications=NUM_REPS,
                             num_producers=30, num_consumers=100, num_advertisers=20, num_speculators=10,
                             circuit_breaker_pct=0.15, position_limit=500)
    results5a = run_experiment("manipulation_with_safeguards", cfg5a, num_reps=NUM_REPS)
    # Without safeguards
    cfg5b = SimulationConfig(num_timesteps=TIMESTEPS, num_replications=NUM_REPS,
                             num_producers=30, num_consumers=100, num_advertisers=20, num_speculators=10,
                             circuit_breaker_pct=1.0, position_limit=10000)
    results5b = run_experiment("manipulation_no_safeguards", cfg5b, num_reps=NUM_REPS)
    agg5a = aggregate_results(results5a)
    agg5b = aggregate_results(results5b)
    save_results(agg5a, "experiment5_with_safeguards")
    save_results(agg5b, "experiment5_no_safeguards")
    all_results["manipulation"] = {
        "with_safeguards": {"results": results5a, "aggregated": agg5a},
        "no_safeguards": {"results": results5b, "aggregated": agg5b},
    }

    # ── Generate Figures ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GENERATING FIGURES...")
    print("=" * 70)
    from visualizations import generate_all_figures
    generate_all_figures(all_results)

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print_summary(all_results)
    print("\nAll experiments complete. Results in results/, figures in figures/")


def print_summary(all_results: dict):
    """Print a summary table of key findings."""
    w = all_results["welfare"]["aggregated"]
    print(f"\n{'Metric':<35} {'Exchange':>12} {'Ad-Driven':>12} {'p-value':>10}")
    print("-" * 70)

    stats = w.get("statistical_tests", {})
    for key in ["total_surplus", "total_producer_surplus", "total_consumer_surplus"]:
        s = stats.get(key, {})
        ex_mean = s.get("exchange_mean", 0)
        ad_mean = s.get("ad_mean", 0)
        p = s.get("mann_whitney_p", 1.0)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{s.get('label', key):<35} {ex_mean:>12.2f} {ad_mean:>12.2f} {p:>8.4f} {sig}")

    ex_gini = w["welfare_gini_exchange"]["mean"]
    ad_gini = w["welfare_gini_ad"]["mean"]
    print(f"{'Revenue Gini Coefficient':<35} {ex_gini:>12.4f} {ad_gini:>12.4f}")

    ex_eff = w["welfare_efficiency_exchange"]["mean"]
    ad_eff = w["welfare_efficiency_ad"]["mean"]
    print(f"{'Allocative Efficiency':<35} {ex_eff:>12.4f} {ad_eff:>12.4f}")


if __name__ == "__main__":
    main()
