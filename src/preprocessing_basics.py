import numpy as np

# 1. Create a sample feature matrix
X = np.array([
    [50, 100, 150],
    [100, 150, 200],
    [150, 200, 250]
])

print("Original Matrix:")
print(X)

# 2. Scale pixel values to [0, 1]
X_scaled = X / 255.0

print("\nScaled Matrix:")
print(X_scaled)

# 3. Calculate feature means
feature_means = np.mean(X_scaled, axis=0)

print("\nFeature Means:")
print(feature_means)

# 4. Center the scaled matrix
X_centered = X_scaled - feature_means

print("\nCentered Matrix:")
print(X_centered)

# 5. Verify feature means after centering
print("\nMeans After Centering:")
print(np.mean(X_centered, axis=0))