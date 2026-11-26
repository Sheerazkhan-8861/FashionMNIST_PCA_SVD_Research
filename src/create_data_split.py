import numpy as np
from sklearn.model_selection import train_test_split

from load_fashion_mnist import load_images, load_labels, DATA_DIR

# Load only the original training data
X_images = load_images(DATA_DIR / "train-images-idx3-ubyte.gz")
y = load_labels(DATA_DIR / "train-labels-idx1-ubyte.gz")

# Flatten images: 28 x 28 = 784 features
X = X_images.reshape(X_images.shape[0], 784)

# Preserve original sample indices
all_indices = np.arange(len(y))

fit_indices, val_indices = train_test_split(
    all_indices,
    test_size=10000,
    random_state=42,
    stratify=y
)

# Create fitting and validation datasets
X_fit = X[fit_indices]
X_val = X[val_indices]

y_fit = y[fit_indices]
y_val = y[val_indices]

# Save datasets and original indices
output_dir = DATA_DIR.parent.parent / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "fashion_mnist_split.npz"

np.savez_compressed(
    output_file,
    X_fit=X_fit,
    X_val=X_val,
    y_fit=y_fit,
    y_val=y_val,
    fit_indices=fit_indices,
    val_indices=val_indices
)

# Verification
print("FASHION-MNIST SPLIT VERIFICATION")
print("-" * 40)

print("Fitting shape:", X_fit.shape)
print("Validation shape:", X_val.shape)

print("Fitting class counts:", np.bincount(y_fit))
print("Validation class counts:", np.bincount(y_val))

print("Total samples:", len(fit_indices) + len(val_indices))
print("Overlapping original indices:",
      len(np.intersect1d(fit_indices, val_indices)))

print("All original samples accounted for:",
      len(np.unique(np.concatenate([fit_indices, val_indices]))) == 60000)

print("Pixel range:", X.min(), "to", X.max())
print("Data type:", X.dtype)

print("\nSaved file:", output_file)
