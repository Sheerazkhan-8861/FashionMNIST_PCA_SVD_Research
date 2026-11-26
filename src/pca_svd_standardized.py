import numpy as np
from pathlib import Path
from threadpoolctl import threadpool_limits


def main():
    input_path = Path("data/processed/fashion_mnist_standardized.npz")
    output_path = Path("outputs/pca_svd_standardized_results.npz")

    data = np.load(input_path)
    X_fit = data["X_fit"]

    print("STANDARDISED-DATA PCA/SVD")
    print("-" * 45)
    print("Input fitting shape:", X_fit.shape)

    with threadpool_limits(limits=1):
        U, singular_values, Vt = np.linalg.svd(
            X_fit,
            full_matrices=False
        )

    explained_variance = singular_values**2 / (X_fit.shape[0] - 1)

    explained_variance_ratio = (
        explained_variance / explained_variance.sum()
    )

    cumulative_variance = np.cumsum(explained_variance_ratio)

    print("U shape:", U.shape)
    print("Singular values shape:", singular_values.shape)
    print("Vt shape:", Vt.shape)
    print("Number of principal components:", len(singular_values))
    print("First 5 explained variance ratios:")
    print(explained_variance_ratio[:5])
    print("Cumulative explained variance:")
    print("First 5 components:", cumulative_variance[4])
    print("First 100 components:", cumulative_variance[99])
    print("All components:", cumulative_variance[-1])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        components=Vt,
        singular_values=singular_values,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_variance_ratio,
        cumulative_variance=cumulative_variance
    )

    print("Saved file:", output_path)
    print("STANDARDISED PCA/SVD COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
