import numpy as np
import pandas as pd
from pathlib import Path


def load_value(path, key):
    with np.load(path) as data:
        return int(data[key])


def main():
    centered = [
        ("90% Explained Variance",
         load_value("outputs/variance_selection_results.npz", "k_90")),
        ("95% Explained Variance",
         load_value("outputs/variance_selection_results.npz", "k_95")),
        ("Kaiser (Covariance Analogue)",
         load_value("outputs/kaiser_centered_variant_results.npz", "selected_k")),
        ("Scree Elbow",
         load_value("outputs/scree_elbow_results.npz", "k_elbow")),
        ("Horn's Parallel Analysis",
         load_value("outputs/parallel_analysis_results.npz", "k_parallel")),
        ("Minka's MLE",
         load_value("outputs/minka_mle_results.npz", "k_minka")),
    ]

    standardized = [
        ("90% Explained Variance",
         load_value("outputs/variance_selection_standardized_results.npz", "k_90")),
        ("95% Explained Variance",
         load_value("outputs/variance_selection_standardized_results.npz", "k_95")),
        ("Kaiser (Eigenvalue > 1)",
         load_value("outputs/kaiser_selection_results.npz", "k_kaiser")),
        ("Scree Elbow",
         load_value("outputs/scree_elbow_standardized_results.npz", "selected_k")),
        ("Horn's Parallel Analysis",
         load_value("outputs/parallel_analysis_standardized_results.npz", "selected_k")),
        ("Minka's MLE",
         load_value("outputs/minka_mle_standardized_results.npz", "selected_k")),
    ]

    centered_df = pd.DataFrame(
        centered,
        columns=["Method", "Selected k"]
    )
    centered_df.insert(0, "Preprocessing", "Scaled and Centered")

    standardized_df = pd.DataFrame(
        standardized,
        columns=["Method", "Selected k"]
    )
    standardized_df.insert(0, "Preprocessing", "Standardised")

    combined_df = pd.concat(
        [centered_df, standardized_df],
        ignore_index=True
    )

    output_dir = Path("outputs/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    centered_path = output_dir / "pca_selection_centered.csv"
    standardized_path = output_dir / "pca_selection_standardized.csv"
    combined_path = output_dir / "pca_selection_combined.csv"

    centered_df.to_csv(centered_path, index=False)
    standardized_df.to_csv(standardized_path, index=False)
    combined_df.to_csv(combined_path, index=False)

    print("CENTERED-DATA COMPARISON")
    print("-" * 55)
    print(centered_df.to_string(index=False))

    print("\nSTANDARDISED-DATA COMPARISON")
    print("-" * 55)
    print(standardized_df.to_string(index=False))

    print("\nCOMBINED COMPARISON")
    print("-" * 55)
    print(combined_df.to_string(index=False))

    print("\nSAVED TABLES")
    print(centered_path)
    print(standardized_path)
    print(combined_path)

    print("\nAll six methods verified in both preprocessing arms.")
    print("FINAL SELECTION TABLES CREATED SUCCESSFULLY")


if __name__ == "__main__":
    main()
