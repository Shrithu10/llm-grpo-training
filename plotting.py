"""
Phase 3 — Plotting utilities.
Saves four publication-ready PNG plots to results/plots/.

Requires matplotlib.  Install with:  pip install matplotlib
Skips gracefully if matplotlib is unavailable.
"""

from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend; works in Colab + headless
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    print("matplotlib not found — plots will be skipped.  pip install matplotlib")


TOKEN_BUDGETS = [50, 100, 200, 400]
METHODS       = ["sft", "simple_pg", "grpo"]
METHOD_LABELS = {"sft": "SFT Baseline", "simple_pg": "Simple PG", "grpo": "GRPO"}

_METHOD_COLOR = {"sft": "#4e79a7", "simple_pg": "#f28e2b", "grpo": "#59a14f"}
_COMP_COLOR   = {
    "correctness": "#2ca02c",
    "closeness":   "#98df8a",
    "format":      "#aec7e8",
    "reasoning":   "#ffbb78",
    "correction":  "#ff7f0e",
}
_ERROR_COLOR  = {
    "correct":         "#2ca02c",
    "numeric_error":   "#ff7f0e",
    "format_error":    "#d62728",
    "reasoning_error": "#9467bd",
    "parse_error":     "#8c564b",
}


def _save(fig, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {path}")


# ── Plot 1: Inference Scaling ─────────────────────────────────────────────────

def plot_inference_scaling(
    scaling_metrics: dict,
    save_path: str = "results/plots/inference_scaling.png",
) -> None:
    """
    Dual-axis line chart: accuracy and avg_reward vs token budget (GRPO profile).

    Interpretation guide (printed in docstring)
    -------------------------------------------
    The curve has three regions:
      1. Steep rise  (50→100 tokens): escaping format-error floor.
         The model gains just enough budget to close all three tags.
      2. Moderate rise (100→200): structured reasoning kicks in.
         Verify blocks allow error detection before committing to an answer.
      3. Diminishing returns (200→400): only the hardest cases remain.
         Correction behaviour (two-block reasoning) handles these at cost
         of more tokens — compute/accuracy tradeoff emerges here.
    """
    if not _HAS_MPL:
        return

    budgets    = TOKEN_BUDGETS
    accs       = [scaling_metrics[b]["accuracy"]        * 100 for b in budgets]
    rewards    = [scaling_metrics[b]["avg_reward"]            for b in budgets]
    corr_rates = [scaling_metrics[b]["correction_rate"]  * 100 for b in budgets]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    l1, = ax1.plot(budgets, accs,       "o-",  color="#2ca02c", lw=2.5, ms=9, label="Accuracy (%)")
    l2, = ax1.plot(budgets, corr_rates, "s--", color="#9467bd", lw=1.8, ms=7, label="Correction Rate (%)")
    l3, = ax2.plot(budgets, rewards,    "^-",  color="#ff7f0e", lw=2.5, ms=9, label="Avg Reward (right)")

    # Annotate exact accuracy values
    for x, y in zip(budgets, accs):
        ax1.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9, color="#2ca02c")

    ax1.set_xlabel("Token Budget  (max_new_tokens)", fontsize=12)
    ax1.set_ylabel("Accuracy / Correction Rate (%)",  fontsize=11)
    ax2.set_ylabel("Avg Reward",                      fontsize=11, color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    ax1.set_xticks(budgets)
    ax1.set_ylim(-5, 110)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Inference Scaling: Accuracy & Reward vs Token Budget\n(GRPO model)",
                  fontsize=13)
    ax1.legend([l1, l2, l3], [l.get_label() for l in [l1, l2, l3]],
               loc="upper left", fontsize=10)
    _save(fig, save_path)


# ── Plot 2: Baseline Comparison ───────────────────────────────────────────────

def plot_baseline_comparison(
    comparison_metrics: dict,
    save_path: str = "results/plots/baseline_comparison.png",
) -> None:
    """
    Grouped bar chart: accuracy (solid) and avg_reward (hatched) per method.
    Left y-axis = accuracy (%); right y-axis = avg reward.
    """
    if not _HAS_MPL:
        return

    methods = METHODS
    labels  = [METHOD_LABELS[m] for m in methods]
    accs    = [comparison_metrics[m]["accuracy"]   * 100 for m in methods]
    rewards = [comparison_metrics[m]["avg_reward"]       for m in methods]
    width   = 0.35
    x       = list(range(len(methods)))

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar([i - width / 2 for i in x], accs, width,
                    color=[_METHOD_COLOR[m] for m in methods], alpha=0.85,
                    label="Accuracy (%)")
    bars2 = ax2.bar([i + width / 2 for i in x], rewards, width,
                    color=[_METHOD_COLOR[m] for m in methods], alpha=0.50,
                    hatch="//", label="Avg Reward")

    for bar, val in zip(bars1, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{val:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar, val in zip(bars2, rewards):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax1.set_ylabel("Accuracy (%)",  fontsize=11)
    ax2.set_ylabel("Avg Reward",    fontsize=11)
    ax1.set_xlabel("Training Method", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylim(0, 115)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_title("Baseline Comparison at 400-Token Budget", fontsize=13)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=10)
    _save(fig, save_path)


# ── Plot 3: Reward Component Breakdown ────────────────────────────────────────

def plot_reward_components(
    comparison_metrics: dict,
    save_path: str = "results/plots/reward_components.png",
) -> None:
    """
    Stacked horizontal bar: mean reward contribution per component per method.
    Reveals HOW each method earns its reward (correctness vs. format vs. correction).
    """
    if not _HAS_MPL:
        return

    methods    = METHODS
    labels     = [METHOD_LABELS[m] for m in methods]
    components = ["correctness", "closeness", "format", "reasoning", "correction"]

    fig, ax = plt.subplots(figsize=(9, 4))
    lefts = [0.0] * len(methods)

    for comp in components:
        vals = [comparison_metrics[m]["reward_components_mean"].get(comp, 0.0)
                for m in methods]
        bars = ax.barh(labels, vals, left=lefts,
                       color=_COMP_COLOR[comp], label=comp.capitalize(), alpha=0.88)
        lefts = [l + v for l, v in zip(lefts, vals)]

        for bar, val in zip(bars, vals):
            if val >= 0.025:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")

    ax.set_xlabel("Mean Reward Contribution", fontsize=11)
    ax.set_title("Reward Component Breakdown by Method  (400-Token Budget)", fontsize=12)
    ax.legend(loc="lower right", fontsize=10, ncol=3)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlim(0, max(lefts) * 1.10)
    _save(fig, save_path)


# ── Plot 4: Error Distribution ────────────────────────────────────────────────

def plot_error_distribution(
    comparison_metrics: dict,
    save_path: str = "results/plots/error_distribution.png",
) -> None:
    """
    1×3 subplot grid: bar chart of error type counts for each method.
    Shows how training shifts the error distribution (format → numeric → correct).
    """
    if not _HAS_MPL:
        return

    methods = METHODS
    all_error_types = sorted({
        et
        for m in methods
        for et in comparison_metrics[m]["error_distribution"].keys()
    })

    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4), sharey=True)

    for ax, method in zip(axes, methods):
        dist   = comparison_metrics[method]["error_distribution"]
        counts = [dist.get(et, 0)              for et in all_error_types]
        colors = [_ERROR_COLOR.get(et, "#888") for et in all_error_types]

        bars = ax.bar(range(len(all_error_types)), counts, color=colors, alpha=0.85, width=0.6)
        ax.set_title(METHOD_LABELS[method], fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(all_error_types)))
        ax.set_xticklabels(
            [et.replace("_", "\n") for et in all_error_types],
            fontsize=8, ha="center",
        )
        ax.set_ylabel("Count" if method == methods[0] else "")
        ax.grid(True, axis="y", alpha=0.3)

        for bar, cnt in zip(bars, counts):
            if cnt > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        str(cnt), ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Shared legend
    legend_handles = [
        mpatches.Patch(color=_ERROR_COLOR.get(et, "#888"), label=et)
        for et in all_error_types
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(all_error_types),
               fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Error Type Distribution by Training Method  (400-Token Budget)", fontsize=12)
    plt.tight_layout()
    _save(fig, save_path)


# ── Master convenience function ───────────────────────────────────────────────

def save_all_plots(all_results: dict, save_dir: str = "results/plots") -> None:
    """
    Generate and save all four plots from experiment_runner.run_all() output.

    Parameters
    ----------
    all_results : dict returned by experiment_runner.run_all()
    save_dir    : directory for output PNG files (created if absent)
    """
    if not _HAS_MPL:
        print("Skipping plots: matplotlib not available.")
        return

    print("\n" + "=" * 65)
    print("  Generating Plots")
    print("=" * 65)

    plot_inference_scaling( all_results["inference_scaling"],
                            f"{save_dir}/inference_scaling.png")
    plot_baseline_comparison(all_results["baseline_comparison"],
                             f"{save_dir}/baseline_comparison.png")
    plot_reward_components( all_results["baseline_comparison"],
                            f"{save_dir}/reward_components.png")
    plot_error_distribution(all_results["baseline_comparison"],
                            f"{save_dir}/error_distribution.png")
    print("All plots saved.")


if __name__ == "__main__":
    from experiment_runner import run_all
    results = run_all()
    save_all_plots(results)
