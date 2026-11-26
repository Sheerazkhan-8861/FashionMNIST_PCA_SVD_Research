import numpy as np

# Load previously calculated PCA information
pca_data = np.load("outputs/pca_svd_results.npz")

cumulative_variance = pca_data["cumulative_variance"]

# Find the minimum number of components for each threshold
k_90 = np.argmax(cumulative_variance >= 0.90) + 1
k_95 = np.argmax(cumulative_variance >= 0.95) + 1

# Display results
print("EXPLAINED VARIANCE COMPONENT SELECTION")
print("-" * 45)

print("90% variance threshold:")
print("Components required:", k_90)
print("Actual variance retained:", cumulative_variance[k_90 - 1])

print("\n95% variance threshold:")
print("Components required:", k_95)
print("Actual variance retained:", cumulative_variance[k_95 - 1])

# Save selected component counts
np.savez_compressed(
    "outputs/variance_selection_results.npz",
    k_90=k_90,
    k_95=k_95,
    variance_90=cumulative_variance[k_90 - 1],
    variance_95=cumulative_variance[k_95 - 1]
)

print("\nSaved: outputs/variance_selection_results.npz")
print("\nVARIANCE SELECTION COMPLETED SUCCESSFULLY")
