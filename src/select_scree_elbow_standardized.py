import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    input_path = Path("outputs/pca_svd_standardized_results.npz")
    output_path = Path("outputs/scree_elbow_standardized_results.npz")
    figure_path = Path("outputs/figures/scree_plot_standardized.png")

    data = np.load(input_path)
    eigenvalues = data["explained_variance"]

    n_components = len(eigenvalues)

    # Normalize component indices and eigenvalues.
    x = np.linspace(0, 1, n_components)
    y = (eigenvalues - eigenvalues.min()) / (
        eigenvalues.max() - eigenvalues.min()
    )

    # Straight line joining the first and last points.
    reference_line = 1 - x

    # Maximum perpendicular distance from the reference line.
    distances = np.abs(y - reference_line) / np.sqrt(2)

    elbow_index = int(np.argmax(distances))
    selected_k = elbow_index + 1

    print("STANDARDISED-DATA SCREE ELBOW")
    print("-" * 45)
    print("Total components:", n_components)
    print("Selected elbow k:", selected_k)
    print("Eigenvalue at elbow:", eigenvalues[elbow_index])
    print("Maximum distance:", distances[elbow_index])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        eigenvalues=eigenvalues,
        selected_k=selected_k,
        elbow_eigenvalue=eigenvalues[elbow_index],
        maximum_distance=distances[elbow_index]
    )

    plt.figure(figsize=(10, 6))
    plt.plot(
        np.arange(1, n_components + 1),
        eigenvalues
    )
    plt.axvline(
        selected_k,
        linestyle="--",
        label=f"Selected k = {selected_k}"
    )
    plt.xlabel("Principal Component")
    plt.ylabel("Eigenvalue")
    plt.title("Scree Plot - Standardised Fashion-MNIST")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close()

    print("Saved results:", output_path)
    print("Saved figure:", figure_path)
    print("STANDARDISED SCREE ELBOW COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
