import numpy as np
from pathlib import Path

# Load preprocessed datasets
data = np.load("data/processed/fashion_mnist_scaled_centered.npz")

X_fit = data["X_fit"]
X_val = data["X_val"]

# Load previously calculated PCA components
pca_data = np.load("outputs/pca_svd_results.npz")
components = pca_data["components"]

# Select the first 100 principal components
k = 100
selected_components = components[:k, :]

# Transform fitting and validation datasets
X_fit_pca = X_fit @ selected_components.T
X_val_pca = X_val @ selected_components.T

# Save transformed datasets
output_file = Path("data/processed/fashion_mnist_pca_100.npz")

np.savez_compressed(
    output_file,
    X_fit_pca=X_fit_pca,
    X_val_pca=X_val_pca,
    y_fit=data["y_fit"],
    y_val=data["y_val"],
    k=k
)

# Verification
print("PCA TRANSFORMATION VERIFICATION")
print("-" * 45)

print("Original fitting shape:", X_fit.shape)
print("Original validation shape:", X_val.shape)

print("Transformed fitting shape:", X_fit_pca.shape)
print("Transformed validation shape:", X_val_pca.shape)

print("Components retained:", k)
print("Labels unchanged:",
      np.array_equal(data["y_fit"], np.load(output_file)["y_fit"]) and
      np.array_equal(data["y_val"], np.load(output_file)["y_val"]))

print("\nSaved file:", output_file)
print("\nPCA TRANSFORMATION COMPLETED SUCCESSFULLY")
