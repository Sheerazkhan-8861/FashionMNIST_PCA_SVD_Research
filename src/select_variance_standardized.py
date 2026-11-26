import numpy as np
from pathlib import Path


def main():
    input_path = Path("outputs/pca_svd_standardized_results.npz")
    output_path = Path("outputs/variance_selection_standardized_results.npz")

    data = np.load(input_path)
    cumulative_variance = data["cumulative_variance"]

    k_90 = int(np.searchsorted(cumulative_variance, 0.90) + 1)
    k_95 = int(np.searchsorted(cumulative_variance, 0.95) + 1)

    variance_90 = cumulative_variance[k_90 - 1]
    variance_95 = cumulative_variance[k_95 - 1]

    print("STANDARDISED EXPLAINED-VARIANCE SELECTION")
    print("-" * 50)
    print(f"90% variance: k = {k_90}")
    print(f"Actual variance retained: {variance_90:.6f}")
    print()
    print(f"95% variance: k = {k_95}")
    print(f"Actual variance retained: {variance_95:.6f}")

    np.savez_compressed(
        output_path,
        k_90=k_90,
        k_95=k_95,
        variance_90=variance_90,
        variance_95=variance_95
    )

    print("Saved file:", output_path)
    print("STANDARDISED VARIANCE SELECTION COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
