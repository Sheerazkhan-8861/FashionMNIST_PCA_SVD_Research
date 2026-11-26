from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

RESULTS = Path("outputs/full_experiment/full_classifier_results.csv")
TABLE_DIR = Path("outputs/tables")
FIGURE_DIR = Path("outputs/figures")

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RESULTS)

required = {
    "Preprocessing", "Representation", "Components",
    "Is_Grid_Point", "Selection_Methods",
    "Validation_Accuracy", "Training_Time_Seconds",
    "Convergence_Warning"
}
missing = required.difference(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df["Is_Grid_Point"] = (
    df["Is_Grid_Point"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
)
if df["Is_Grid_Point"].isna().any():
    raise ValueError("Could not interpret Is_Grid_Point values.")

arms = ["Scaled and Centered", "Standardised"]
comparison_rows = []
grid_results = {}

for arm in arms:
    subset = df[
        (df["Preprocessing"] == arm)
        & (df["Representation"] == "PCA")
    ]

    grid = subset[subset["Is_Grid_Point"]]
    if grid.empty:
        raise ValueError(f"No PCA grid results found for {arm}.")

    best_grid = grid.loc[grid["Validation_Accuracy"].idxmax()]
    grid_results[arm] = best_grid

    selected = subset[subset["Selection_Methods"].notna()]

    for _, row in selected.iterrows():
        comparison_rows.append({
            "Preprocessing": arm,
            "Selection_Method": row["Selection_Methods"],
            "Selected_k": int(row["Components"]),
            "Validation_Accuracy": row["Validation_Accuracy"],
            "Accuracy_Difference_pp_vs_Best_Grid":
                (row["Validation_Accuracy"]
                 - best_grid["Validation_Accuracy"]) * 100,
            "Best_Grid_k": int(best_grid["Components"]),
            "Best_Grid_Accuracy": best_grid["Validation_Accuracy"],
            "Training_Time_Seconds": row["Training_Time_Seconds"],
            "Convergence_Warning": row["Convergence_Warning"],
        })

comparison = pd.DataFrame(comparison_rows)
comparison = comparison.sort_values(
    ["Preprocessing", "Validation_Accuracy"],
    ascending=[True, False]
)

comparison_path = TABLE_DIR / "validation_selection_rule_comparison.csv"
comparison.to_csv(comparison_path, index=False)

# Figure 1: PCA grid accuracy against component count, with rule-selected
# configurations marked separately.
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, arm in zip(axes, arms):
    subset = df[
        (df["Preprocessing"] == arm)
        & (df["Representation"] == "PCA")
    ]
    grid = subset[subset["Is_Grid_Point"]].sort_values("Components")
    selected = subset[subset["Selection_Methods"].notna()]
    best = grid_results[arm]

    ax.plot(
        grid["Components"],
        grid["Validation_Accuracy"],
        marker="o",
        label="PCA component grid"
    )
    ax.scatter(
        selected["Components"],
        selected["Validation_Accuracy"],
        marker="x",
        s=65,
        label="Rule-selected configurations"
    )
    ax.scatter(
        [best["Components"]],
        [best["Validation_Accuracy"]],
        marker="*",
        s=180,
        label="Best grid configuration"
    )

    ax.set_title(arm)
    ax.set_xlabel("Number of principal components (k)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

axes[0].set_ylabel("Validation accuracy")
fig.suptitle("Fashion-MNIST: Validation Accuracy Across PCA Dimensions")
fig.tight_layout()
fig.savefig(
    FIGURE_DIR / "validation_accuracy_by_components.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

# Figure 2: Difference in percentage points between each rule and
# the best grid configuration in the same preprocessing arm.
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

for ax, arm in zip(axes, arms):
    part = comparison[comparison["Preprocessing"] == arm].sort_values(
        "Accuracy_Difference_pp_vs_Best_Grid"
    )

    ax.barh(
        part["Selection_Method"],
        part["Accuracy_Difference_pp_vs_Best_Grid"]
    )
    ax.axvline(0, linewidth=1)
    ax.set_title(arm)
    ax.set_xlabel("Accuracy difference vs best grid (percentage points)")
    ax.grid(axis="x", alpha=0.3)

fig.suptitle("Dimension-Selection Rules Compared with the Best Grid Result")
fig.tight_layout()
fig.savefig(
    FIGURE_DIR / "selection_rule_accuracy_difference.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

print("VALIDATION ANALYSIS COMPLETED")
print("=" * 65)
print("Total experiment rows:", len(df))
print("Selection-rule configurations:", len(comparison))

for arm in arms:
    best = grid_results[arm]
    print(f"\n{arm}")
    print("Best grid k:", int(best["Components"]))
    print("Best grid accuracy:", f'{best["Validation_Accuracy"]:.4f}')

    print(
        comparison[comparison["Preprocessing"] == arm][
            [
                "Selection_Method", "Selected_k",
                "Validation_Accuracy",
                "Accuracy_Difference_pp_vs_Best_Grid"
            ]
        ].to_string(index=False)
    )

print("\nSaved table:", comparison_path)
print("Saved figure:", FIGURE_DIR / "validation_accuracy_by_components.png")
print("Saved figure:", FIGURE_DIR / "selection_rule_accuracy_difference.png")
print("\nNo classifier was trained. The official test set was not accessed.")
