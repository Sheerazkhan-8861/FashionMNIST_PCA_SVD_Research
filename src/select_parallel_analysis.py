import numpy as np
from pathlib import Path

# Reproducibility settings
N_PERMUTATIONS = 100
RANDOM_SEED = 42

# Load scaled and centered fitting data only
data = np.load("data/processed/fashion_mnist_scaled_centered.npz")
X_fit = data["X_fit"]

# Load observed eigenvalues from SVD
pca_data = np.load("outputs/pca_svd_results.npz")
observed_eigenvalues = pca_data["explained_variance"]

n_samples, n_features = X_fit.shape

# Generate random reference eigenvalues
rng = np.random.default_rng(RANDOM_SEED)

random_eigenvalues = np.zeros((N_PERMUTATIONS, n_features))

print("HORN'S PARALLEL ANALYSIS")
print("-" * 45)
print("Dataset shape:", X_fit.shape)
print("Number of permutations:", N_PERMUTATIONS)

for i in range(N_PERMUTATIONS):
    # Independently shuffle each feature column
    X_random = np.empty_like(X_fit)

    for j in range(n_features):
        X_random[:, j] = rng.permutation(X_fit[:, j])

    # Calculate covariance eigenvalues
    covariance = np.cov(X_random, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]

    random_eigenvalues[i] = eigenvalues

    if (i + 1) % 10 == 0:
        print(f"Completed {i + 1}/{N_PERMUTATIONS} permutations")

# Calculate the 95th percentile reference eigenvalue
reference_eigenvalues = np.percentile(
    random_eigenvalues,
    95,
    axis=0
)

# Retain consecutive components exceeding the reference
comparison = observed_eigenvalues > reference_eigenvalues

if np.any(~comparison):
    k_parallel = int(np.argmax(~comparison))
else:
    k_parallel = n_features

# Save results
output_file = Path("outputs/parallel_analysis_results.npz")

np.savez_compressed(
    output_file,
    observed_eigenvalues=observed_eigenvalues,
    random_eigenvalues=random_eigenvalues,
    reference_eigenvalues=reference_eigenvalues,
    comparison=comparison,
    k_parallel=k_parallel,
    n_permutations=N_PERMUTATIONS,
    random_seed=RANDOM_SEED
)

print("\nPARALLEL ANALYSIS RESULTS")
print("-" * 45)
print("Selected components:", k_parallel)
print("Observed eigenvalue at selection boundary:",
      observed_eigenvalues[k_parallel - 1] if k_parallel > 0 else "None")
print("Random reference eigenvalue at boundary:",
      reference_eigenvalues[k_parallel - 1] if k_parallel > 0 else "None")
print("Saved file:", output_file)
print("\nPARALLEL ANALYSIS COMPLETED")
