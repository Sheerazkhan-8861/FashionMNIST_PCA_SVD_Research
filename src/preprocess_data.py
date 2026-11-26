import numpy as np
from pathlib import Path

INPUT_FILE = Path("data/processed/fashion_mnist_split.npz")
OUTPUT_DIR = Path("data/processed")

# Load original datasets
data = np.load(INPUT_FILE)

X_fit = data["X_fit"].astype(np.float64) / 255.0
X_val = data["X_val"].astype(np.float64) / 255.0

y_fit = data["y_fit"]
y_val = data["y_val"]

# Calculate fitting-set feature means
feature_means = X_fit.mean(axis=0)

# Center both datasets using fitting-set means
X_fit_centered = X_fit - feature_means
X_val_centered = X_val - feature_means

# Calculate fitting-set standard deviations
feature_stds = X_fit.std(axis=0, ddof=0)
safe_stds = np.where(feature_stds == 0, 1.0, feature_stds)

# Standardise both datasets using fitting-set statistics
X_fit_standardized = X_fit_centered / safe_stds
X_val_standardized = X_val_centered / safe_stds

# Save scaled and centered datasets
np.savez_compressed(
    OUTPUT_DIR / "fashion_mnist_scaled_centered.npz",
    X_fit=X_fit_centered,
    X_val=X_val_centered,
    y_fit=y_fit,
    y_val=y_val
)

# Save standardised datasets
np.savez_compressed(
    OUTPUT_DIR / "fashion_mnist_standardized.npz",
    X_fit=X_fit_standardized,
    X_val=X_val_standardized,
    y_fit=y_fit,
    y_val=y_val
)

# Verification
print("FINAL PREPROCESSING VERIFICATION")
print("-" * 45)

print("Scaled and centered fitting shape:", X_fit_centered.shape)
print("Scaled and centered validation shape:", X_val_centered.shape)

print("\nStandardised fitting shape:", X_fit_standardized.shape)
print("Standardised validation shape:", X_val_standardized.shape)

print("\nScaled and centered fitting means (first 5):")
print(X_fit_centered.mean(axis=0)[:5])

print("\nStandardised fitting means (first 5):")
print(X_fit_standardized.mean(axis=0)[:5])

print("\nStandardised fitting std (first 5):")
print(X_fit_standardized.std(axis=0)[:5])

print("\nConstant features:", np.sum(feature_stds == 0))

print("\nLabels unchanged:",
      np.array_equal(y_fit, data["y_fit"]) and
      np.array_equal(y_val, data["y_val"]))

print("\nPREPROCESSING COMPLETED SUCCESSFULLY")
