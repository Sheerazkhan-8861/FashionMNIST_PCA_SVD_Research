import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load PCA eigenvalues
pca_data = np.load("outputs/pca_svd_results.npz")
eigenvalues = pca_data["explained_variance"]

# Component numbers
components = np.arange(1, len(eigenvalues) + 1)

# Calculate elbow using maximum distance from the straight line
# joining the first and last points of the scree curve.
x = (components - components.min()) / (components.max() - components.min())
y = (eigenvalues - eigenvalues.min()) / (
    eigenvalues.max() - eigenvalues.min()
)

line_y = 1 - x
distances = np.abs(y - line_y) / np.sqrt(2)

k_elbow = int(np.argmax(distances) + 1)

# Create output directories
output_dir = Path("outputs")
figure_dir = output_dir / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)

# Plot scree curve
plt.figure(figsize=(10, 6))
plt.plot(components, eigenvalues)
plt.axvline(k_elbow, linestyle="--", label=f"Elbow: k={k_elbow}")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance")
plt.title("Fashion-MNIST Scree Plot")
plt.legend()
plt.grid(True)
plt.tight_layout()

figure_file = figure_dir / "scree_plot.png"
plt.savefig(figure_file, dpi=300)
plt.close()

# Save results
np.savez_compressed(
    output_dir / "scree_elbow_results.npz",
    k_elbow=k_elbow,
    eigenvalues=eigenvalues,
    distances=distances
)

print("SCREE ELBOW RESULTS")
print("-" * 40)
print("Total components:", len(eigenvalues))
print("Selected elbow component:", k_elbow)
print("Eigenvalue at elbow:", eigenvalues[k_elbow - 1])
print("Saved figure:", figure_file)
print("Saved results: outputs/scree_elbow_results.npz")
print("\nSCREE ELBOW ANALYSIS COMPLETED")
