import numpy as np
import sklearn
from sklearn.decomposition import PCA
from pathlib import Path
from threadpoolctl import threadpool_limits


def main():
    input_path = Path("data/processed/fashion_mnist_standardized.npz")
    output_path = Path("outputs/minka_mle_standardized_results.npz")

    data = np.load(input_path)
    X_fit = data["X_fit"]

    print("STANDARDISED-DATA MINKA'S MLE")
    print("-" * 45)
    print("Fitting data shape:", X_fit.shape)
    print("Scikit-learn version:", sklearn.__version__)

    model = PCA(
        n_components="mle",
        svd_solver="full"
    )

    with threadpool_limits(limits=1):
        model.fit(X_fit)

    selected_k = model.n_components_

    print("Selected components (k):", selected_k)
    print(
        "Explained variance retained:",
        model.explained_variance_ratio_.sum()
    )
    print("Components shape:", model.components_.shape)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        selected_k=selected_k,
        explained_variance_ratio_sum=(
            model.explained_variance_ratio_.sum()
        ),
        components=model.components_,
        explained_variance=model.explained_variance_,
        sklearn_version=sklearn.__version__
    )

    print("Saved results:", output_path)
    print("STANDARDISED MINKA'S MLE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
