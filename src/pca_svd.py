import numpy as np
from pathlib import Path

INPUT_FILE = Path("data/processed/fashion_mnist_scaled_centered.npz")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the fitting dataset only
data = np.load(INPUT_FILE)
X_fit = data["X_fit"]

print("PCA USING SINGULAR VALUE DECOMPOSITION")
print("-" * 45)
print("Input fitting shape:", X_fit.shape)

# Perform SVD
U, singular_values, Vt = np.linalg.svd(
    X_fit,
    full_matrices=False
)

# Calculate explained variance
explained_variance = singular_values**2 / (X_fit.shape[0] - 1)

# Calculate explained variance ratio
explained_variance_ratio = (
    explained_variance / explained_variance.sum()
)

# Calculate cumulative explained variance
cumulative_variance = np.cumsum(explained_variance_ratio)

# Save PCA results (U is not needed for our later experiments)
output_file = OUTPUT_DIR / "pca_svd_results.npz"

np.savez_compressed(
    output_file,
    components=Vt,
    singular_values=singular_values,
    explained_variance=explained_variance,
    explained_variance_ratio=explained_variance_ratio,
    cumulative_variance=cumulative_variance
)

# Verification
print("\nSVD OUTPUT SHAPES")
print("U shape:", U.shape)
print("Singular values shape:", singular_values.shape)
print("Vt shape:", Vt.shape)

print("\nPCA INFORMATION")
print("Number of principal components:", len(singular_values))
print("First 5 explained variance ratios:")
print(explained_variance_ratio[:5])

print("\nCumulative explained variance:")
print("First 5 components:", cumulative_variance[4])
print("First 100 components:", cumulative_variance[99])
print("All components:", cumulative_variance[-1])

print("\nSaved file:", output_file)
print("\nPCA DECOMPOSITION COMPLETED SUCCESSFULLY")
