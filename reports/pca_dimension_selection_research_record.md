# PCA Dimension-Selection Research Record

## 1. Project
Title: Do Rules of Thumb Pick the Right k? Comparing PCA Dimension-Selection Rules Against Downstream Accuracy on Fashion-MNIST.

## 2. Dataset and experimental protocol
- Dataset: Fashion-MNIST.
- Original training samples: 60,000.
- Fitting samples: 50,000.
- Validation samples: 10,000.
- Features per image: 784.
- Image dimensions: 28 x 28 grayscale.
- Number of classes: 10.
- Stratified split random seed: 42.
- Official test set: 10,000 samples; reserved for final evaluation.
- All PCA decompositions and dimension-selection calculations in this phase used fitting data only.

## 3. Preprocessing arms

### Arm A: Scaled and centered
Pixel values were scaled to [0, 1]. Feature means were calculated using fitting data only and subtracted from both fitting and validation data.

Formula:
X_centered = X_scaled - mu_fit

### Arm B: Standardised
Feature means and standard deviations were calculated using fitting data only.

Formula:
X_standardized = (X_scaled - mu_fit) / sigma_fit

The fitting-set statistics were reused for validation data. No validation or test statistics were used to fit preprocessing.

## 4. PCA and SVD foundations

For a centered fitting matrix X with n samples and p features:

X = U Sigma V^T

Where:
- U contains left singular vectors.
- Sigma contains singular values.
- V^T contains principal directions.

The covariance eigenvalues are:

lambda_i = sigma_i^2 / (n - 1)

Explained variance ratio:

r_i = lambda_i / sum(lambda_j)

Cumulative explained variance through component k:

C_k = sum(r_i), for i = 1 to k

## 5. Dimension-selection methods

### 5.1 90% and 95% explained variance

Select the smallest k satisfying:

C_k >= threshold

Thresholds:
- 90%: 0.90
- 95%: 0.95

### 5.2 Kaiser criterion: standardised arm

Retain components satisfying:

lambda_i > 1

This is the conventional Kaiser eigenvalue-greater-than-one rule for standardised features.

### 5.3 Centered-data Kaiser covariance analogue

For the scaled-and-centered arm, retain components satisfying:

lambda_i > mean(lambda)

The mean eigenvalue is:

mean(lambda) = (1 / p) * sum(lambda_i)

This is a covariance-based analogue specified by the project protocol. It is not the conventional standardised Kaiser criterion.

### 5.4 Scree elbow

The implementation:
1. Uses covariance eigenvalues from the fitting-data PCA decomposition.
2. Min-max normalises the eigenvalues and component indices.
3. Constructs the reference line y = 1 - x between the normalised endpoints.
4. Calculates perpendicular distance from each point to this line.
5. Selects the component index with maximum distance.

This is a heuristic elbow detector. It is not a universally unique or objectively guaranteed elbow definition.

### 5.5 Horn's parallel analysis

Procedure:
1. Calculate observed covariance eigenvalues from fitting data.
2. Independently permute each feature column to destroy cross-feature relationships while preserving each column's empirical distribution.
3. Calculate covariance eigenvalues for each permuted dataset.
4. Repeat for 100 permutations.
5. Calculate the 95th percentile of random eigenvalues for each component.
6. Retain consecutive observed components whose eigenvalues exceed their corresponding random reference eigenvalues.

Settings:
- Number of permutations: 100.
- Random seed: 42.
- Reference percentile: 95.
- Data used: fitting data only.

Selection condition:

lambda_observed_i > lambda_random_i_95th_percentile

The selected count is conditional on these settings.

### 5.6 Minka's MLE

Implementation:

sklearn.decomposition.PCA(
    n_components="mle",
    svd_solver="full"
)

Fitted separately on each preprocessing arm's fitting data.

Scikit-learn version: 1.9.1.

Both arms selected k = 783. This is the maximum permitted component count under this MLE configuration for 784 features. It is a boundary result and must not be interpreted as proof that exactly 783 meaningful dimensions exist.

## 6. Verified dimension-selection results

| Method | Scaled and centered | Standardised |
|---|---:|---:|
| 90% explained variance | 84 | 137 |
| 95% explained variance | 187 | 256 |
| Kaiser | 58* | 80 |
| Scree elbow | 15 | 21 |
| Horn's parallel analysis | 39 | 67 |
| Minka's MLE | 783 | 783 |

*The centered value is the covariance analogue using the mean eigenvalue threshold. The standardised value uses the conventional eigenvalue-greater-than-one criterion.

## 7. Verified PCA/SVD observations

### Scaled-and-centered arm
- Fitting shape: 50,000 x 784.
- First five components explained approximately 61.5982% of variance.
- First 100 components explained approximately 91.2295%.
- Total explained variance approximately 100%.

### Standardised arm
- Fitting shape: 50,000 x 784.
- First five components explained approximately 51.1011% of variance.
- First 100 components explained approximately 87.1173%.
- Total explained variance approximately 100%.

## 8. Interpretation and limitations

1. Dimension-selection outcomes depend on preprocessing.
2. The six methods produce substantially different component counts.
3. Explained variance alone does not establish downstream classification usefulness.
4. The centered Kaiser method is a protocol-defined covariance analogue and must be named accordingly.
5. The scree elbow depends on the specified maximum-distance heuristic.
6. Horn's parallel-analysis result depends on the permutation count, random seed, and reference percentile.
7. Minka's MLE selected a boundary value in both arms and requires cautious interpretation.
8. No classifier-performance conclusion can be drawn from these component counts alone.
9. These results are dimension-selection results, not final test-set results.

## 9. Saved experimental artifacts

### PCA/SVD
- outputs/pca_svd_results.npz
- outputs/pca_svd_standardized_results.npz

### Scaled-and-centered selection
- outputs/variance_selection_results.npz
- outputs/kaiser_centered_variant_results.npz
- outputs/scree_elbow_results.npz
- outputs/parallel_analysis_results.npz
- outputs/minka_mle_results.npz

### Standardised selection
- outputs/variance_selection_standardized_results.npz
- outputs/kaiser_selection_results.npz
- outputs/scree_elbow_standardized_results.npz
- outputs/parallel_analysis_standardized_results.npz
- outputs/minka_mle_standardized_results.npz

### Comparison tables
- outputs/tables/pca_selection_centered.csv
- outputs/tables/pca_selection_standardized.csv
- outputs/tables/pca_selection_combined.csv

### Figures
- outputs/figures/scree_plot.png
- outputs/figures/scree_plot_standardized.png

## 10. Next experimental phase

Evaluate downstream Logistic Regression performance using comparable settings across both preprocessing arms.

The official test set must remain untouched until the final authorised evaluation.
