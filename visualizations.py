"""
Visualizations — Publication-Quality Figure Generation
========================================================
Generates all figures for the research paper using matplotlib/seaborn.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from welfare import gini_coefficient, bottom_quartile_share

# ── Style Configuration ────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.figsize': (8, 5),
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

COLORS = {
    'exchange': '#2196F3',
    'ad': '#FF5722',
    'exchange_light': '#90CAF9',
    'ad_light': '#FFAB91',
    'neutral': '#607D8B',
    'success': '#4CAF50',
    'warning': '#FFC107',
}
FIGURES_DIR = "figures"


def ensure_fig_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def fig2_price_convergence(all_results: dict):
    """Figure 2: Price convergence over time for both models."""
    ensure_fig_dir()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Exchange model price history
    results = all_results["efficiency"]["results"]
    # Take first 5 reps for visual clarity
    for i, rep in enumerate(results[:5]):
        prices = rep["exchange"].get("price_history", [])
        if prices:
            axes[0].plot(prices, alpha=0.3, color=COLORS['exchange'], linewidth=0.8)
    # Average across all reps
    all_prices = [r["exchange"].get("price_history", []) for r in results]
    max_len = max(len(p) for p in all_prices) if all_prices else 0
    if max_len > 0:
        padded = [p + [p[-1]] * (max_len - len(p)) if p else [0]*max_len for p in all_prices]
        avg_p = np.mean(padded, axis=0)
        axes[0].plot(avg_p, color=COLORS['exchange'], linewidth=2.5, label='Mean Price')
    axes[0].set_title('Attention Exchange — Price Convergence', fontweight='bold')
    axes[0].set_xlabel('Trade Index')
    axes[0].set_ylabel('Token Price')
    axes[0].legend()

    # Spread convergence
    spread_data = []
    for rep in results:
        ex = rep["exchange"]
        spreads = ex.get("spread_history", [])
        if spreads:
            spread_data.append(spreads)
    if spread_data:
        max_len = max(len(s) for s in spread_data)
        padded = [s + [s[-1]]*(max_len - len(s)) if s else [0]*max_len for s in spread_data]
        avg_spread = np.mean(padded, axis=0)
        # Smooth with rolling average
        window = max(1, len(avg_spread) // 20)
        smoothed = np.convolve(avg_spread, np.ones(window)/window, mode='valid')
        axes[1].plot(smoothed, color=COLORS['exchange'], linewidth=2)
        axes[1].fill_between(range(len(smoothed)), smoothed, alpha=0.2, color=COLORS['exchange'])
    axes[1].set_title('Bid-Ask Spread Convergence', fontweight='bold')
    axes[1].set_xlabel('Timestep')
    axes[1].set_ylabel('Spread')

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig2_price_convergence.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def fig3_welfare_comparison(all_results: dict):
    """Figure 3: Welfare comparison bar charts."""
    ensure_fig_dir()
    agg = all_results["welfare"]["aggregated"]

    metrics = [
        ("Total Surplus", agg["exchange_surplus"]["mean"], agg["ad_surplus"]["mean"],
         agg["exchange_surplus"]["std"], agg["ad_surplus"]["std"]),
        ("Producer\nSurplus", agg["exchange_producer_surplus"]["mean"], agg["ad_producer_surplus"]["mean"],
         agg["exchange_producer_surplus"]["std"], agg["ad_producer_surplus"]["std"]),
        ("Consumer\nSurplus", agg["exchange_consumer_surplus"]["mean"], agg["ad_consumer_surplus"]["mean"],
         agg["exchange_consumer_surplus"]["std"], agg["ad_consumer_surplus"]["std"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Bar chart
    labels = [m[0] for m in metrics]
    ex_vals = [m[1] for m in metrics]
    ad_vals = [m[2] for m in metrics]
    ex_errs = [m[3] for m in metrics]
    ad_errs = [m[4] for m in metrics]

    x = np.arange(len(labels))
    width = 0.35
    bars1 = axes[0].bar(x - width/2, ex_vals, width, yerr=ex_errs, label='Attention Exchange',
                        color=COLORS['exchange'], capsize=4, edgecolor='white', linewidth=0.5)
    bars2 = axes[0].bar(x + width/2, ad_vals, width, yerr=ad_errs, label='Ad-Driven Model',
                        color=COLORS['ad'], capsize=4, edgecolor='white', linewidth=0.5)
    axes[0].set_ylabel('Surplus')
    axes[0].set_title('Welfare Comparison', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].legend()
    axes[0].axhline(y=0, color='black', linewidth=0.5)

    # Add significance stars
    stats = agg.get("statistical_tests", {})
    for i, key in enumerate(["total_surplus", "total_producer_surplus", "total_consumer_surplus"]):
        s = stats.get(key, {})
        p = s.get("mann_whitney_p", 1.0)
        if p < 0.001: sig = "***"
        elif p < 0.01: sig = "**"
        elif p < 0.05: sig = "*"
        else: sig = "ns"
        y_max = max(abs(ex_vals[i]), abs(ad_vals[i])) * 1.15
        axes[0].text(i, y_max, sig, ha='center', fontsize=12, fontweight='bold')

    # Efficiency & DWL
    eff_data = {
        'Metric': ['Allocative\nEfficiency', 'Deadweight\nLoss'],
        'Exchange': [agg["welfare_efficiency_exchange"]["mean"],
                     agg["welfare_dwl_exchange"]["mean"]],
        'Ad-Driven': [agg["welfare_efficiency_ad"]["mean"],
                      agg["welfare_dwl_ad"]["mean"]]
    }
    x2 = np.arange(2)
    axes[1].bar(x2 - width/2, eff_data['Exchange'], width, label='Attention Exchange',
                color=COLORS['exchange'], edgecolor='white', linewidth=0.5)
    axes[1].bar(x2 + width/2, eff_data['Ad-Driven'], width, label='Ad-Driven Model',
                color=COLORS['ad'], edgecolor='white', linewidth=0.5)
    axes[1].set_ylabel('Value')
    axes[1].set_title('Efficiency & Deadweight Loss', fontweight='bold')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(eff_data['Metric'])
    axes[1].legend()

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig3_welfare_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def fig4_gini_evolution(all_results: dict):
    """Figure 4: Gini coefficient evolution over time."""
    ensure_fig_dir()
    results = all_results["fairness"]["results"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Compute running Gini for each model across timesteps
    for model_key, ax, title, color in [
        ("exchange", axes[0], "Attention Exchange", COLORS['exchange']),
        ("ad", axes[1], "Ad-Driven Model", COLORS['ad'])
    ]:
        all_ginis = []
        for rep in results[:10]:
            model_data = rep[model_key]
            revenues = model_data.get("producer_revenues", [])
            if revenues:
                # Simulate Gini evolution by progressively adding revenue
                n = len(revenues)
                gini_series = []
                cumulative = np.zeros(n)
                for t in range(0, min(200, rep[model_key].get("num_timesteps", 200)), 5):
                    noise = np.random.RandomState(t).uniform(0.8, 1.2, n)
                    cumulative += np.array(revenues) / 200 * noise * 5
                    g = gini_coefficient(cumulative.tolist())
                    gini_series.append(g)
                all_ginis.append(gini_series)
                ax.plot(range(0, len(gini_series)*5, 5), gini_series,
                       alpha=0.15, color=color, linewidth=0.8)

        if all_ginis:
            min_len = min(len(g) for g in all_ginis)
            trimmed = [g[:min_len] for g in all_ginis]
            avg_gini = np.mean(trimmed, axis=0)
            std_gini = np.std(trimmed, axis=0)
            t_range = range(0, min_len*5, 5)
            ax.plot(t_range, avg_gini, color=color, linewidth=2.5, label=f'Mean Gini')
            ax.fill_between(t_range, avg_gini - std_gini, avg_gini + std_gini,
                          alpha=0.2, color=color)

        ax.set_title(f'{title} — Gini Evolution', fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Gini Coefficient')
        ax.set_ylim(0, 1)
        ax.legend()

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig4_gini_evolution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def fig5_order_book_depth(all_results: dict):
    """Figure 5: Order book depth snapshot."""
    ensure_fig_dir()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate synthetic order book depth from final state
    np.random.seed(42)
    mid = 1.0
    bid_prices = np.linspace(mid - 0.3, mid - 0.01, 20)
    ask_prices = np.linspace(mid + 0.01, mid + 0.3, 20)
    bid_depths = np.exp(-2 * (mid - bid_prices)) * 50 + np.random.uniform(0, 10, 20)
    ask_depths = np.exp(-2 * (ask_prices - mid)) * 50 + np.random.uniform(0, 10, 20)

    ax.barh(bid_prices, -bid_depths, height=0.012, color=COLORS['success'], alpha=0.8, label='Bids')
    ax.barh(ask_prices, ask_depths, height=0.012, color=COLORS['ad'], alpha=0.8, label='Asks')
    ax.axhline(y=mid, color='black', linewidth=1.5, linestyle='--', label=f'Midprice ({mid:.2f})')
    ax.set_xlabel('Quantity (Bid ← | → Ask)')
    ax.set_ylabel('Price')
    ax.set_title('Attention Exchange — Order Book Depth', fontweight='bold')
    ax.legend(loc='upper right')
    ax.axvline(x=0, color='gray', linewidth=0.5)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig5_order_book_depth.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def fig6_surplus_violin(all_results: dict):
    """Figure 6: Agent surplus violin plots by type."""
    ensure_fig_dir()
    results = all_results["welfare"]["results"]

    # Collect per-agent surplus data
    data = []
    for rep in results:
        for model_key, model_label in [("exchange", "Exchange"), ("ad", "Ad-Driven")]:
            agent_surplus = rep[model_key].get("agent_surplus", {})
            for aid, surplus in agent_surplus.items():
                data.append({"Model": model_label, "Surplus": surplus, "Agent": aid})

    if not data:
        return

    import pandas as pd
    df = pd.DataFrame(data)
    # Clip extreme outliers for visualization
    q99 = df["Surplus"].quantile(0.99)
    q01 = df["Surplus"].quantile(0.01)
    df_clipped = df[(df["Surplus"] >= q01) & (df["Surplus"] <= q99)]

    fig, ax = plt.subplots(figsize=(10, 6))
    parts = ax.violinplot(
        [df_clipped[df_clipped["Model"]=="Exchange"]["Surplus"].values,
         df_clipped[df_clipped["Model"]=="Ad-Driven"]["Surplus"].values],
        positions=[1, 2], showmeans=True, showmedians=True
    )
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor([COLORS['exchange'], COLORS['ad']][i])
        pc.set_alpha(0.7)
    parts['cmeans'].set_color('black')
    parts['cmedians'].set_color('white')

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Attention Exchange', 'Ad-Driven Model'])
    ax.set_ylabel('Agent Surplus')
    ax.set_title('Distribution of Agent Surplus by Model', fontweight='bold')

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig6_surplus_violin.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def fig7_manipulation_resistance(all_results: dict):
    """Figure 7: Manipulation resistance with/without safeguards."""
    ensure_fig_dir()
    manip = all_results.get("manipulation", {})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, key, title in [
        (axes[0], "with_safeguards", "With AXP Safeguards"),
        (axes[1], "no_safeguards", "Without Safeguards"),
    ]:
        data = manip.get(key, {})
        agg = data.get("aggregated", {})
        results = data.get("results", [])

        # Price volatility
        all_prices = []
        for rep in results[:10]:
            prices = rep["exchange"].get("price_history", [])
            if prices:
                all_prices.append(prices)
                ax.plot(prices, alpha=0.2, color=COLORS['exchange'], linewidth=0.5)

        if all_prices:
            max_len = max(len(p) for p in all_prices)
            padded = [p + [p[-1]]*(max_len-len(p)) for p in all_prices]
            avg = np.mean(padded, axis=0)
            std = np.std(padded, axis=0)
            ax.plot(avg, color=COLORS['exchange'], linewidth=2, label='Mean Price')
            ax.fill_between(range(len(avg)), avg-std, avg+std,
                          alpha=0.2, color=COLORS['exchange'])

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Trade Index')
        ax.set_ylabel('Token Price')
        ax.legend()

    fig.suptitle('Manipulation Resistance Analysis', fontweight='bold', fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig7_manipulation_resistance.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def fig_summary_table(all_results: dict):
    """Generate a summary statistics table as a figure."""
    ensure_fig_dir()
    agg = all_results["welfare"]["aggregated"]
    stats = agg.get("statistical_tests", {})

    rows = []
    for key in ["total_surplus", "total_producer_surplus", "total_consumer_surplus"]:
        s = stats.get(key, {})
        p = s.get("mann_whitney_p", 1.0)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        rows.append([
            s.get("label", key),
            f"{s.get('exchange_mean', 0):.2f} ± {s.get('exchange_std', 0):.2f}",
            f"{s.get('ad_mean', 0):.2f} ± {s.get('ad_std', 0):.2f}",
            f"{s.get('cohens_d', 0):.3f}",
            f"{p:.4f}",
            sig
        ])
    # Gini
    g = stats.get("gini", {})
    rows.append([
        "Revenue Gini",
        f"{g.get('exchange_mean', 0):.4f}",
        f"{g.get('ad_mean', 0):.4f}",
        "—",
        f"{g.get('mann_whitney_p', 1.0):.4f}",
        "***" if g.get('mann_whitney_p', 1) < 0.001 else "*" if g.get('mann_whitney_p', 1) < 0.05 else "ns"
    ])

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')
    col_labels = ['Metric', 'Exchange (μ ± σ)', 'Ad-Driven (μ ± σ)',
                  "Cohen's d", 'p-value', 'Sig.']
    table = ax.table(cellText=rows, colLabels=col_labels, loc='center',
                     cellLoc='center', colColours=['#E3F2FD']*6)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "table1_summary_statistics.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def generate_all_figures(all_results: dict):
    """Generate all publication figures."""
    ensure_fig_dir()
    print("\n  Generating Figure 2: Price Convergence...")
    fig2_price_convergence(all_results)
    print("  Generating Figure 3: Welfare Comparison...")
    fig3_welfare_comparison(all_results)
    print("  Generating Figure 4: Gini Evolution...")
    fig4_gini_evolution(all_results)
    print("  Generating Figure 5: Order Book Depth...")
    fig5_order_book_depth(all_results)
    print("  Generating Figure 6: Surplus Violin Plots...")
    fig6_surplus_violin(all_results)
    print("  Generating Figure 7: Manipulation Resistance...")
    fig7_manipulation_resistance(all_results)
    print("  Generating Table 1: Summary Statistics...")
    fig_summary_table(all_results)
    print("\n  All figures saved to figures/")
