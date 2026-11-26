import numpy as np
import sklearn
from pathlib import Path
from sklearn.decomposition import PCA

# Load scaled and centered fitting data only
data = np.load("data/processed/fashion_mnist_scaled_centered.npz")
X_fit = data["X_fit"]

print("MINKA'S MAXIMUM LIKELIHOOD ESTIMATION")
print("-" * 45)
print("Input fitting shape:", X_fit.shape)
print("Scikit-learn version:", sklearn.__version__)
print("Starting PCA with Minka's MLE...")

# Apply Minka's MLE
pca = PCA(
    n_components="mle",
    svd_solver="full"
)

pca.fit(X_fit)

# Extract selected number of components
k_minka = pca.n_components_

# Save results
output_file = Path("outputs/minka_mle_results.npz")

np.savez_compressed(
    output_file,
    k_minka=k_minka,
    explained_variance=pca.explained_variance_,
    explained_variance_ratio=pca.explained_variance_ratio_,
    cumulative_variance=np.cumsum(pca.explained_variance_ratio_),
    sklearn_version=sklearn.__version__
)

# Display results
print("\nMINKA'S MLE RESULTS")
print("-" * 45)
print("Selected components:", k_minka)
print("Explained variance retained:",
      np.sum(pca.explained_variance_ratio_))
print("Components shape:", pca.components_.shape)

print("\nSaved file:", output_file)
print("\nMINKA'S MLE COMPLETED SUCCESSFULLY")
