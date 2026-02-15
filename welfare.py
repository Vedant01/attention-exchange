"""
Welfare Metrics — Economic Analysis Engine
=============================================
Computes welfare economics metrics for comparing the Attention Exchange
vs Ad-Driven models: surplus, efficiency, Gini, deadweight loss.
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy import stats as scipy_stats


def gini_coefficient(values: List[float]) -> float:
    """Compute Gini coefficient. 0 = perfect equality, 1 = max inequality."""
    arr = np.array(values, dtype=float)
    arr = arr[arr >= 0]  # Filter negatives
    if len(arr) == 0 or arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr))


def herfindahl_index(shares: List[float]) -> float:
    """Herfindahl-Hirschman Index for market concentration."""
    arr = np.array(shares, dtype=float)
    total = arr.sum()
    if total == 0:
        return 0.0
    proportions = arr / total
    return float(np.sum(proportions ** 2))


def allocative_efficiency(actual_surplus: float, optimal_surplus: float) -> float:
    """Ratio of actual to theoretical maximum surplus."""
    if optimal_surplus <= 0:
        return 0.0
    return min(actual_surplus / optimal_surplus, 1.0)


def bottom_quartile_share(values: List[float]) -> float:
    """Share of total value held by bottom 25%."""
    arr = np.sort(np.array(values, dtype=float))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    q1_count = max(1, n // 4)
    return float(arr[:q1_count].sum() / arr.sum())


def compute_deadweight_loss(actual_surplus: float, optimal_surplus: float) -> float:
    """Deadweight loss = optimal surplus - actual surplus."""
    return max(0, optimal_surplus - actual_surplus)


def price_quality_correlation(prices: List[float], qualities: List[float]) -> Tuple[float, float]:
    """Spearman rank correlation between price and quality. Returns (rho, p-value)."""
    if len(prices) < 3 or len(qualities) < 3:
        return 0.0, 1.0
    min_len = min(len(prices), len(qualities))
    rho, p = scipy_stats.spearmanr(prices[:min_len], qualities[:min_len])
    return float(rho), float(p)


def compute_optimal_surplus(producer_qualities: List[float],
                            consumer_budgets: List[float],
                            production_costs: List[float]) -> float:
    """
    Compute theoretical maximum surplus via optimal matching.
    Uses greedy assignment: highest-quality producers matched to
    highest-budget consumers.
    """
    sorted_q = sorted(enumerate(producer_qualities), key=lambda x: -x[1])
    sorted_b = sorted(consumer_budgets, reverse=True)
    total = 0.0
    for i, (pidx, q) in enumerate(sorted_q):
        if i >= len(sorted_b):
            break
        valuation = sorted_b[i] * q * 0.7
        cost = production_costs[pidx] if pidx < len(production_costs) else 0
        surplus = valuation - cost
        if surplus > 0:
            total += surplus
    return total


class WelfareAnalyzer:
    """Comprehensive welfare analysis comparing two market models."""

    def __init__(self, exchange_results: dict, ad_results: dict):
        self.exchange = exchange_results
        self.ad = ad_results

    def full_analysis(self) -> dict:
        """Run all welfare comparisons."""
        # Producer revenue Gini
        ex_gini_rev = gini_coefficient(self.exchange.get("producer_revenues", []))
        ad_gini_rev = gini_coefficient(self.ad.get("producer_revenues", []))

        # Producer surplus
        ex_ps = self.exchange.get("total_producer_surplus", 0)
        ad_ps = self.ad.get("total_producer_surplus", 0)

        # Consumer surplus
        ex_cs = self.exchange.get("total_consumer_surplus", 0)
        ad_cs = self.ad.get("total_consumer_surplus", 0)

        # Total surplus
        ex_total = self.exchange.get("total_surplus", 0)
        ad_total = self.ad.get("total_surplus", 0)

        # HHI
        ex_hhi = herfindahl_index(self.exchange.get("producer_revenues", []))
        ad_hhi = herfindahl_index(self.ad.get("producer_revenues", []))

        # Bottom quartile
        ex_bq = bottom_quartile_share(self.exchange.get("producer_revenues", []))
        ad_bq = bottom_quartile_share(self.ad.get("producer_revenues", []))

        # Optimal surplus estimate
        qualities = self.exchange.get("producer_qualities", [])
        costs = [0.5] * len(qualities)
        budgets = [10.0] * 200
        optimal = compute_optimal_surplus(qualities, budgets, costs)

        ex_efficiency = allocative_efficiency(ex_total, optimal)
        ad_efficiency = allocative_efficiency(ad_total, optimal)
        ex_dwl = compute_deadweight_loss(ex_total, optimal)
        ad_dwl = compute_deadweight_loss(ad_total, optimal)

        return {
            "exchange_producer_surplus": ex_ps,
            "ad_producer_surplus": ad_ps,
            "exchange_consumer_surplus": ex_cs,
            "ad_consumer_surplus": ad_cs,
            "exchange_total_surplus": ex_total,
            "ad_total_surplus": ad_total,
            "exchange_gini_revenue": ex_gini_rev,
            "ad_gini_revenue": ad_gini_rev,
            "exchange_hhi": ex_hhi,
            "ad_hhi": ad_hhi,
            "exchange_bottom_quartile": ex_bq,
            "ad_bottom_quartile": ad_bq,
            "exchange_allocative_efficiency": ex_efficiency,
            "ad_allocative_efficiency": ad_efficiency,
            "exchange_dwl": ex_dwl,
            "ad_dwl": ad_dwl,
            "optimal_surplus": optimal,
            "surplus_improvement_pct": ((ex_total - ad_total) / max(abs(ad_total), 1)) * 100,
            "gini_improvement": ad_gini_rev - ex_gini_rev,
        }

    def statistical_comparison(self, exchange_runs: List[dict],
                                ad_runs: List[dict]) -> dict:
        """
        Statistical tests across multiple replications.
        Uses Mann-Whitney U test (non-parametric).
        """
        results = {}

        metrics_to_compare = [
            ("total_surplus", "Total Surplus"),
            ("total_producer_surplus", "Producer Surplus"),
            ("total_consumer_surplus", "Consumer Surplus"),
        ]

        for key, label in metrics_to_compare:
            ex_vals = [r.get(key, 0) for r in exchange_runs]
            ad_vals = [r.get(key, 0) for r in ad_runs]

            if len(ex_vals) < 2 or len(ad_vals) < 2:
                continue

            u_stat, p_value = scipy_stats.mannwhitneyu(
                ex_vals, ad_vals, alternative='two-sided'
            )
            t_stat, t_p = scipy_stats.ttest_ind(ex_vals, ad_vals)

            effect_size = (np.mean(ex_vals) - np.mean(ad_vals)) / max(
                np.sqrt((np.std(ex_vals)**2 + np.std(ad_vals)**2) / 2), 1e-10
            )

            results[key] = {
                "label": label,
                "exchange_mean": np.mean(ex_vals),
                "exchange_std": np.std(ex_vals),
                "ad_mean": np.mean(ad_vals),
                "ad_std": np.std(ad_vals),
                "mann_whitney_u": u_stat,
                "mann_whitney_p": p_value,
                "t_statistic": t_stat,
                "t_test_p": t_p,
                "cohens_d": effect_size,
                "significant_005": p_value < 0.05,
                "significant_001": p_value < 0.01,
            }

        # Gini comparison
        ex_ginis = [gini_coefficient(r.get("producer_revenues", [])) for r in exchange_runs]
        ad_ginis = [gini_coefficient(r.get("producer_revenues", [])) for r in ad_runs]
        if len(ex_ginis) >= 2 and len(ad_ginis) >= 2:
            u, p = scipy_stats.mannwhitneyu(ex_ginis, ad_ginis, alternative='two-sided')
            results["gini"] = {
                "label": "Revenue Gini",
                "exchange_mean": np.mean(ex_ginis),
                "ad_mean": np.mean(ad_ginis),
                "mann_whitney_p": p,
                "significant_005": p < 0.05,
            }

        return results
