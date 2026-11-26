import numpy as np
from pathlib import Path

# Load the standardised fitting dataset
data = np.load("data/processed/fashion_mnist_standardized.npz")
X_fit = data["X_fit"]

# Calculate covariance matrix
covariance_matrix = np.cov(X_fit, rowvar=False)

# Calculate eigenvalues
eigenvalues = np.linalg.eigvalsh(covariance_matrix)

# Sort eigenvalues from largest to smallest
eigenvalues = eigenvalues[::-1]

# Apply Kaiser criterion
selected = eigenvalues > 1.0
k_kaiser = int(np.sum(selected))

# Save results
output_file = Path("outputs/kaiser_selection_results.npz")

np.savez_compressed(
    output_file,
    eigenvalues=eigenvalues,
    selected=selected,
    k_kaiser=k_kaiser
)

# Verification
print("KAISER CRITERION RESULTS")
print("-" * 40)

print("Covariance matrix shape:", covariance_matrix.shape)
print("Total eigenvalues:", len(eigenvalues))
print("Components with eigenvalue > 1:", k_kaiser)

print("\nFirst 10 eigenvalues:")
print(eigenvalues[:10])

print("\nLast 10 eigenvalues:")
print(eigenvalues[-10:])

print("\nSaved file:", output_file)
print("\nKAISER SELECTION COMPLETED SUCCESSFULLY")
