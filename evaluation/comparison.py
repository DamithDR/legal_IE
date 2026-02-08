"""Aggregate results across models and generate comparison reports."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

import config


def load_all_results() -> dict:
    """Load all *_results.json files from the results directory."""
    results = {}
    for path in config.RESULTS_DIR.glob("*_results.json"):
        with open(path) as f:
            data = json.load(f)
        model_name = data["model_name"]
        results[model_name] = data
    return results


def generate_comparison_report(results: dict = None):
    """
    Generate a unified comparison report across all evaluated models.

    Outputs:
        - Console table
        - results/overall_comparison.csv
        - results/per_entity_comparison.csv
        - results/detailed_results.json
        - results/comparison_chart.png
    """
    if results is None:
        results = load_all_results()

    if not results:
        print("No results found in results/ directory.")
        return

    # --- Overall comparison table ---
    overall_rows = []
    for model_name, data in results.items():
        overall_rows.append({
            "Model": model_name,
            "Type": data.get("model_type", "unknown"),
            "Samples": data.get("num_samples_evaluated", "?"),
            "Failures": data.get("num_failures", 0),
            "Precision": data["overall"]["precision"],
            "Recall": data["overall"]["recall"],
            "F1": data["overall"]["f1"],
            "Macro F1": data["macro_avg"]["f1"],
        })

    overall_df = pd.DataFrame(overall_rows).sort_values("F1", ascending=False)

    print("\n" + "=" * 80)
    print("OVERALL MODEL COMPARISON")
    print("=" * 80)
    print(tabulate(overall_df, headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))

    overall_df.to_csv(config.RESULTS_DIR / "overall_comparison.csv", index=False)

    # --- Per-entity F1 comparison ---
    all_entity_types = set()
    for data in results.values():
        all_entity_types.update(data.get("per_entity", {}).keys())
    all_entity_types = sorted(all_entity_types)

    per_entity_rows = []
    for model_name, data in results.items():
        row = {"Model": model_name}
        for etype in all_entity_types:
            etype_data = data.get("per_entity", {}).get(etype, {})
            row[etype] = etype_data.get("f1", 0.0)
        per_entity_rows.append(row)

    per_entity_df = pd.DataFrame(per_entity_rows)

    print("\n" + "=" * 80)
    print("PER-ENTITY F1 COMPARISON")
    print("=" * 80)
    print(tabulate(per_entity_df, headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))

    per_entity_df.to_csv(config.RESULTS_DIR / "per_entity_comparison.csv", index=False)

    # --- Save detailed results ---
    with open(config.RESULTS_DIR / "detailed_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- Comparison chart ---
    _generate_chart(overall_df, per_entity_df, all_entity_types)

    print(f"\nAll reports saved to {config.RESULTS_DIR}/")


def _generate_chart(overall_df, per_entity_df, entity_types):
    """Generate comparison bar charts."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Chart 1: Overall P/R/F1 by model
    models = overall_df["Model"].tolist()
    x = range(len(models))
    width = 0.25

    axes[0].bar(
        [i - width for i in x],
        overall_df["Precision"],
        width,
        label="Precision",
        color="#2196F3",
    )
    axes[0].bar(x, overall_df["Recall"], width, label="Recall", color="#4CAF50")
    axes[0].bar(
        [i + width for i in x],
        overall_df["F1"],
        width,
        label="F1",
        color="#FF9800",
    )

    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Overall NER Performance")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha="right")
    axes[0].legend()
    axes[0].set_ylim(0, 1.0)

    # Chart 2: F1 per entity type (heatmap-style grouped bar)
    if len(entity_types) > 0 and len(models) > 0:
        num_models = len(per_entity_df)
        x_entities = range(len(entity_types))
        bar_width = 0.8 / max(num_models, 1)

        colors = plt.cm.Set2.colors
        for i, (_, row) in enumerate(per_entity_df.iterrows()):
            offsets = [x + i * bar_width - 0.4 + bar_width / 2 for x in x_entities]
            values = [row.get(et, 0) for et in entity_types]
            axes[1].bar(
                offsets,
                values,
                bar_width,
                label=row["Model"],
                color=colors[i % len(colors)],
            )

        axes[1].set_xlabel("Entity Type")
        axes[1].set_ylabel("F1 Score")
        axes[1].set_title("Per-Entity F1 Comparison")
        axes[1].set_xticks(range(len(entity_types)))
        axes[1].set_xticklabels(entity_types, rotation=45, ha="right", fontsize=8)
        axes[1].legend(fontsize=7, loc="upper right")
        axes[1].set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "comparison_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Chart saved to results/comparison_chart.png")
