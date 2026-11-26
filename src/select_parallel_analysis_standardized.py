import numpy as np
from pathlib import Path
from threadpoolctl import threadpool_limits


def main():
    input_path = Path("data/processed/fashion_mnist_standardized.npz")
    output_path = Path(
        "outputs/parallel_analysis_standardized_results.npz"
    )

    data = np.load(input_path)
    X_fit = data["X_fit"]

    n_permutations = 100
    random_seed = 42
    percentile = 95

    print("STANDARDISED-DATA HORN'S PARALLEL ANALYSIS")
    print("-" * 55)
    print("Fitting data shape:", X_fit.shape)
    print("Number of permutations:", n_permutations)
    print("Random seed:", random_seed)
    print("Reference percentile:", percentile)

    # Observed covariance eigenvalues.
    with threadpool_limits(limits=1):
        observed_covariance = np.cov(X_fit, rowvar=False)
        observed_eigenvalues = np.linalg.eigvalsh(
            observed_covariance
        )[::-1]

    rng = np.random.default_rng(random_seed)

    random_eigenvalues = np.empty(
        (n_permutations, X_fit.shape[1])
    )

    # Independently permute each feature column.
    with threadpool_limits(limits=1):
        for permutation in range(n_permutations):
            X_random = np.empty_like(X_fit)

            for column in range(X_fit.shape[1]):
                X_random[:, column] = rng.permutation(
                    X_fit[:, column]
                )

            random_covariance = np.cov(
                X_random,
                rowvar=False
            )

            random_eigenvalues[permutation] = np.linalg.eigvalsh(
                random_covariance
            )[::-1]

            print(
                f"Completed permutation "
                f"{permutation + 1}/{n_permutations}"
            )

    # 95th-percentile random reference eigenvalues.
    reference_eigenvalues = np.percentile(
        random_eigenvalues,
        percentile,
        axis=0
    )

    # Retain consecutive components exceeding the reference.
    exceeds_reference = (
        observed_eigenvalues > reference_eigenvalues
    )

    selected_k = 0

    for condition in exceeds_reference:
        if condition:
            selected_k += 1
        else:
            break

    print()
    print("Selected components (k):", selected_k)

    if selected_k > 0:
        print(
            "Observed eigenvalue at boundary:",
            observed_eigenvalues[selected_k - 1]
        )
        print(
            "Reference eigenvalue at boundary:",
            reference_eigenvalues[selected_k - 1]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        observed_eigenvalues=observed_eigenvalues,
        reference_eigenvalues=reference_eigenvalues,
        random_eigenvalues=random_eigenvalues,
        selected_k=selected_k,
        n_permutations=n_permutations,
        random_seed=random_seed,
        percentile=percentile
    )

    print("Saved results:", output_path)
    print(
        "STANDARDISED PARALLEL ANALYSIS "
        "COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()
