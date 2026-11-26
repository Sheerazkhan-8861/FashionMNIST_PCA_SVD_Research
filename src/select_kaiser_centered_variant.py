import numpy as np
from pathlib import Path
from threadpoolctl import threadpool_limits


def main():
    input_path = Path("data/processed/fashion_mnist_scaled_centered.npz")
    output_path = Path("outputs/kaiser_centered_variant_results.npz")

    data = np.load(input_path)
    X_fit = data["X_fit"]

    print("CENTERED-DATA KAISER VARIANT")
    print("-" * 45)
    print("Fitting data shape:", X_fit.shape)

    # Calculate covariance matrix and its eigenvalues.
    with threadpool_limits(limits=1):
        covariance = np.cov(X_fit, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(covariance)

    # Sort eigenvalues in descending order.
    eigenvalues = eigenvalues[::-1]

    # Protocol-defined covariance analogue.
    threshold = eigenvalues.mean()
    selected_k = int(np.sum(eigenvalues > threshold))

    print("Covariance matrix shape:", covariance.shape)
    print("Number of eigenvalues:", len(eigenvalues))
    print("Mean eigenvalue threshold:", threshold)
    print("Selected components (k):", selected_k)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        eigenvalues=eigenvalues,
        threshold=threshold,
        selected_k=selected_k,
        method="Centered-data Kaiser covariance analogue",
        description="Retain eigenvalues greater than the mean eigenvalue."
    )

    print("Saved file:", output_path)
    print("CENTERED KAISER VARIANT COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
