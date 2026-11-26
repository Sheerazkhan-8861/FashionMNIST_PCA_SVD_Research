# Mathematical Formulas and Notation Register

## Research Project

**Title:** Do Rules of Thumb Pick the Right k? Comparing PCA
Dimension-Selection Rules Against Downstream Accuracy on Fashion-MNIST.

**Purpose:** This document is the central mathematical reference for
the research project. Every formula used in the methodology,
implementation, experiments, and evaluation must be recorded here.

Formulas must match the actual implementation. Any approximation,
variant, assumption, or implementation-specific convention must be
identified explicitly.

---

# 1. Notation and Dataset

Let:

- \(n\): number of observations.
- \(p\): number of original features.
- \(d\): number of retained principal components.
- \(X\): data matrix.
- \(x_i\): feature vector for observation \(i\).
- \(x_{ij}\): value of feature \(j\) for observation \(i\).
- \(\mu_j\): fitting-set mean of feature \(j\).
- \(\sigma_j\): fitting-set standard deviation of feature \(j\).
- \(V\): matrix of principal-component directions.
- \(k\): number of retained principal components.
- \(y_i\): true class label of observation \(i\).

For Fashion-MNIST:

- Original image dimensions: \(28 \times 28\).
- Original feature count: \(p=784\).
- Number of classes: \(10\).
- Fitting observations: \(n=50,000\).
- Validation observations: \(10,000\).
- Official test observations: \(10,000\).

The fitting set is used to estimate preprocessing statistics and
principal-component directions. Validation observations use those
same fitted transformations. The official test set remains untouched
until authorised final evaluation.

---

# 2. Preprocessing Formulas

## 2.1 Pixel Scaling

Original pixel intensities range from 0 to 255.

The scaled feature value is:

\[
x_{ij}^{(\mathrm{scaled})}=\frac{x_{ij}^{(\mathrm{raw})}}{255}
\]

This maps pixel intensities to the interval \([0,1]\).

## 2.2 Fitting-Set Feature Mean

For feature \(j\), the mean is estimated using fitting observations:

\[
\mu_j=\frac{1}{n}\sum_{i=1}^{n}x_{ij}
\]

The validation set is not used to calculate this mean.

## 2.3 Mean Centering

Each scaled feature is centered by subtracting its fitting-set mean:

\[
x_{ij}^{(\mathrm{centered})}
=x_{ij}^{(\mathrm{scaled})}-\mu_j
\]

The same fitting-set mean is used to center validation observations.

## 2.4 Fitting-Set Standard Deviation

The population standard deviation used by the preprocessing
implementation is:

\[
\sigma_j=
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}(x_{ij}-\mu_j)^2
}
\]

The exact implementation convention must be checked against the
preprocessing source code before final paper reporting.

## 2.5 Standardisation

A feature is standardised using:

\[
x_{ij}^{(\mathrm{standardized})}
=
\frac{x_{ij}^{(\mathrm{scaled})}-\mu_j}{\sigma_j}
\]

For a feature with zero standard deviation, the implementation uses
a safe divisor of 1.0.

The fitting-set means and standard deviations are reused for
validation data.

---

# 3. PCA and Singular Value Decomposition

## 3.1 Data Matrix

Let the centered fitting data matrix be:

\[
X\in\mathbb{R}^{n\times p}
\]

For this project:

\[
X\in\mathbb{R}^{50000\times784}
\]

The standardized preprocessing arm also produces a transformed
fitting matrix with this shape.

## 3.2 Singular Value Decomposition

The fitting matrix is decomposed as:

\[
X=U\Sigma V^T
\]

Where:

- \(U\): left singular-vector matrix.
- \(\Sigma\): diagonal matrix of singular values.
- \(V^T\): transpose of the right singular-vector matrix.
- Singular values are non-negative and ordered from largest to smallest.

The implementation uses NumPy SVD with:

\[
\texttt{full\_matrices=False}
\]

For the project data, the resulting shapes are:

\[
U\in\mathbb{R}^{50000\times784}
\]

\[
\Sigma\in\mathbb{R}^{784\times784}
\]

\[
V^T\in\mathbb{R}^{784\times784}
\]

## 3.3 Covariance Matrix

For mean-centered data, the sample covariance matrix is:

\[
S=\frac{1}{n-1}X^TX
\]

The covariance matrix has dimensions:

\[
S\in\mathbb{R}^{784\times784}
\]

## 3.4 Eigenvalues and Singular Values

The covariance eigenvalues are related to singular values by:

\[
\lambda_j=\frac{s_j^2}{n-1}
\]

Where:

- \(\lambda_j\): covariance eigenvalue for component \(j\).
- \(s_j\): corresponding singular value.
- \(n\): number of fitting observations.

## 3.5 Explained Variance

The explained variance of component \(j\) is:

\[
EV_j=\lambda_j
\]

Equivalently, when calculated from singular values:

\[
EV_j=\frac{s_j^2}{n-1}
\]

## 3.6 Explained Variance Ratio

The explained variance ratio of component \(j\) is:

\[
EVR_j=
\frac{\lambda_j}
{\sum_{r=1}^{p}\lambda_r}
\]

## 3.7 Cumulative Explained Variance

The cumulative explained variance ratio through component \(k\) is:

\[
CEVR_k=\sum_{j=1}^{k}EVR_j
\]

## 3.8 PCA Projection

The first \(k\) principal-component directions are represented by:

\[
V_k\in\mathbb{R}^{p\times k}
\]

The transformed fitting observations are:

\[
Z_{\mathrm{fit}}=X_{\mathrm{fit}}V_k
\]

The transformed validation observations are:

\[
Z_{\mathrm{val}}=X_{\mathrm{val}}V_k
\]

In the saved implementation, the component directions are stored as
rows in \(V^T\). Therefore, the equivalent operation is:

\[
Z=X(V_k^T)
\]

where \(V_k^T\) is the first \(k\) rows of the saved component matrix.

## 3.9 PCA Reconstruction

The rank-\(k\) reconstruction of the centered data is:

\[
\hat{X}_k=ZV_k^T
\]

This reconstructs observations using only the first \(k\)
principal-component directions.

---

# 4. Dimension-Selection Methods

The project compares six component-selection methods under two
preprocessing approaches.

The exact operational formulas and implementation conventions for
each method will be recorded after inspecting the corresponding
source scripts.

## 4.1 Explained Variance Thresholds

The selection rule is based on cumulative explained variance:

\[
k_q=\min\{k:CEVR_k\geq q\}
\]

Where:

- \(q=0.90\) for the 90% threshold.
- \(q=0.95\) for the 95% threshold.

## 4.2 Kaiser Criterion

The standardised Kaiser criterion and the centered covariance
analogue are different procedures.

Their exact implementation rules will be recorded separately after
source-code verification.

## 4.3 Scree Elbow

The elbow-selection procedure is heuristic.

The exact curve, geometric calculation, endpoint convention, and
selected-index convention must be documented from the implementation.

## 4.4 Horn's Parallel Analysis

Parallel analysis compares observed eigenvalues against eigenvalues
obtained from randomly permuted data.

The exact permutation procedure, number of permutations, random
seed, percentile threshold, and component-count rule must be
documented from the implementation.

## 4.5 Minka's Maximum Likelihood Estimation

Minka's MLE estimates intrinsic dimensionality using a
probabilistic PCA model.

The exact scikit-learn implementation, solver settings, and
boundary-result interpretation must be recorded.

---

# 5. Logistic Regression

The verified multiclass Logistic Regression formulation, class-score
equation, softmax probability function, cross-entropy loss, L2
regularisation objective, prediction rule, and implementation
correspondence are documented in the later section:

- `Logistic Regression: Multiclass Objective, Probabilities, and Prediction`
  (Section beginning at line 705).

The formulation corresponds to the settings used in the full classifier
experiment. See that section for notation, implementation conventions,
and verification details.

---

# 6. Evaluation Metrics

## 6.1 Accuracy

Accuracy was the primary metric used for the classifier experiments.
Its mathematical definition and validation-comparison convention are
documented in the later sections:

- `Full Classifier Experiment: Validation Accuracy`
- `Validation Comparison: Accuracy Difference`

The final held-out test evaluation also reports accuracy.

## 6.2 Confusion Matrices

Confusion matrices were generated and audited for all 16 final test
configurations. They are saved as evaluation artifacts under
`outputs/test_evaluation/`.

## 6.3 Metrics Not Reported

Per-class precision, per-class recall, per-class F1-score, macro
precision, macro recall, and macro F1-score were not reported as
experimental metrics. Their formulas are therefore not presented as
methods used in this study.

---

# 7. Bootstrap Stability

Bootstrap or subsampling stability analysis was not performed in this
study. No bootstrap stability statistic, sampling procedure, sample
size, repetition count, or bootstrap random seed is claimed.

Paired bootstrap confidence intervals for validation accuracy
differences could not be calculated because per-sample validation
predictions were not saved. This limitation is documented in Section 15
of `reports/full_classifier_experiment_record.md`.

---

# 8. Experimental Data Separation

The following rules govern every mathematical transformation:

1. Preprocessing statistics are estimated using fitting observations.
2. PCA directions are fitted using fitting observations.
3. Validation observations are transformed using the fitted
   preprocessing statistics and PCA directions.
4. Validation results may be used for model comparison and
   component selection.
5. Official test observations must not influence preprocessing,
   PCA fitting, model selection, or hyperparameter selection.
6. Final test evaluation requires explicit authorisation.

---

# 9. Documentation and Verification Status

This register is maintained throughout the project.

A formula is considered verified only when:

1. Its mathematical definition is correct.
2. Its notation and symbols are defined.
3. Its implementation convention is identified.
4. Its correspondence with the actual source code is checked.
5. Any variant, approximation, or limitation is documented.

Do not mark an unverified formula as implementation-verified.


---

# Verified Implementation Addendum — Preprocessing, PCA/SVD and Dimension Selection

Verification source: direct inspection of the corresponding Python scripts.
Dataset dimensions: n = 50,000 fitting samples and p = 784 pixel features.
All preprocessing statistics and PCA components are fitted using the fitting set only.

## 1. Pixel scaling

For sample i and feature j:

x'_{ij} = x_{ij} / 255

Symbols:
- x_{ij}: original pixel intensity, between 0 and 255.
- x'_{ij}: scaled pixel intensity, between 0 and 1.

Implementation:
- Convert pixel arrays to float64.
- Divide fitting and validation pixels by 255.0.

Purpose:
- Put pixel intensities on a consistent numerical scale.

## 2. Feature-wise centering

mu_j = (1/n) * sum_{i=1}^{n} x'_{ij}

x^c_{ij} = x'_{ij} - mu_j

Symbols:
- mu_j: mean of fitting-set feature j.
- n: number of fitting samples.
- x^c_{ij}: centered value of feature j for sample i.

Implementation:
- NumPy mean with axis=0.
- Calculate means from fitting data only.
- Subtract the same fitting-set means from fitting and validation data.

Purpose:
- Center each feature around its fitting-set mean.
- Prevent validation information from influencing preprocessing.

## 3. Population standard deviation and standardisation

sigma_j = sqrt((1/n) * sum_{i=1}^{n} (x'_{ij} - mu_j)^2)

safe_sigma_j = 1 if sigma_j = 0, otherwise sigma_j.

x^s_{ij} = (x'_{ij} - mu_j) / safe_sigma_j

Symbols:
- sigma_j: population standard deviation of fitting-set feature j.
- safe_sigma_j: standard deviation after zero-value protection.
- x^s_{ij}: standardised feature value.

Implementation:
- NumPy std with axis=0 and ddof=0.
- Replace zero standard deviations with 1.0 using np.where.
- Apply the fitting-set means and safe standard deviations to both fitting and validation data.

Important convention:
- Preprocessing standard deviation uses denominator n (population standard deviation).
- This is not the n-1 sample-standard-deviation convention.
- The standardised PCA explained-variance calculation later uses n-1. These are distinct implementation conventions.

## 4. Singular Value Decomposition

X = U Sigma V^T

Symbols:
- X: preprocessed fitting matrix, shape (n, p).
- U: left singular vectors.
- Sigma: diagonal matrix of singular values.
- V^T: transpose of the right singular-vector matrix.
- sigma_j: j-th singular value (when written as a singular value rather than a feature standard deviation).

Implementation:
- np.linalg.svd(X_fit, full_matrices=False).
- The decomposition is fitted on the fitting matrix only.
- For n=50,000 and p=784, the returned shapes are:
  U: (50000, 784)
  singular_values: (784,)
  Vt: (784, 784)

Saved component convention:
- The saved array named components is Vt.
- Each principal component is stored as one row of Vt.
- Do not describe the saved components array as V rather than V^T.

## 5. PCA explained variance

lambda_j = sigma_j^2 / (n - 1)

Symbols:
- lambda_j: explained variance (eigenvalue) of principal component j.
- sigma_j: j-th singular value of the fitting matrix.
- n: number of fitting samples.

Implementation:
- singular_values**2 / (X_fit.shape[0] - 1).
- This uses the sample covariance denominator n-1.
- Applied identically in centered and standardised PCA/SVD scripts.

## 6. Explained-variance ratio

EVR_j = lambda_j / sum_{ell=1}^{p} lambda_ell

Symbols:
- EVR_j: proportion of total variance explained by component j.
- lambda_j: explained variance of component j.
- p: number of features/components (784).

Implementation:
- explained_variance / explained_variance.sum().

## 7. Cumulative explained variance

C_k = sum_{j=1}^{k} EVR_j

Symbols:
- C_k: cumulative explained-variance ratio through component k.
- EVR_j: explained-variance ratio of component j.
- k: number of retained components.

Implementation:
- np.cumsum(explained_variance_ratio).

## 8. PCA projection

Z_k = X V_k^T

Symbols:
- X: centered or standardised data matrix.
- V_k^T: first k rows of the saved Vt/components matrix.
- Z_k: representation in the first k principal-component directions.
- k: retained component count.

Implementation:
- Saved components are rows of Vt.
- Projection is performed by multiplying X by the transpose of the first k component rows.
- Fitting and validation data use the same fitted component directions.

## 9. Explained-variance threshold selection

k_q = min { k : sum_{j=1}^{k} EVR_j >= q }

Symbols:
- q: target variance threshold (0.90 or 0.95).
- EVR_j: explained-variance ratio of component j.
- k_q: smallest number of components reaching or exceeding q.

Implementation:
- Centered script: np.argmax(cumulative_variance >= threshold) + 1.
- Standardised script: np.searchsorted(cumulative_variance, threshold) + 1.
- The observed cumulative variance reaches approximately 1 in both arms, so both implementations returned valid threshold selections in this experiment.
- The centered np.argmax expression would return 1 if no threshold were reached; this is an edge-case limitation of that implementation.

## 10. Standardised Kaiser criterion

k = sum_{j=1}^{p} I(lambda_j > 1)

Symbols:
- lambda_j: descending eigenvalues of the sample covariance matrix of standardised features.
- I(condition): 1 when the condition is true, otherwise 0.
- p: number of features.

Implementation:
- np.cov(X_fit, rowvar=False).
- np.linalg.eigvalsh for symmetric covariance-matrix eigenvalues.
- Sort eigenvalues in descending order.
- Retain eigenvalues strictly greater than 1.

Interpretation:
- This is the conventional Kaiser threshold applied to the standardised covariance matrix.
- Because standardisation used ddof=0 while np.cov uses its sample-covariance convention, the resulting covariance diagonal/eigenvalue scale is not exactly the same as a covariance matrix normalised by n.

## 11. Centered covariance analogue of the Kaiser criterion

k = sum_{j=1}^{p} I(lambda_j > mean(lambda))

Symbols:
- lambda_j: descending eigenvalues of the centered-data sample covariance matrix.
- mean(lambda): arithmetic mean of all p covariance eigenvalues.
- I(condition): indicator function.

Implementation:
- Calculate covariance eigenvalues from centered fitting data.
- Set threshold to eigenvalues.mean().
- Retain eigenvalues strictly greater than this threshold.

Important methodological qualification:
- This is a protocol-defined covariance analogue used for the centered arm.
- It is not the conventional Kaiser criterion on standardised features.
- Keep this distinction explicit in tables and paper discussion.

## 12. Scree elbow: maximum-distance heuristic

For component index j from 1 to p:

x_j = (j - 1) / (p - 1)

y_j = (lambda_j - lambda_min) / (lambda_max - lambda_min)

d_j = abs(y_j - (1 - x_j)) / sqrt(2)

k = argmax_j d_j + 1

Symbols:
- lambda_j: eigenvalue for component j, ordered descending.
- lambda_max and lambda_min: largest and smallest eigenvalues.
- x_j and y_j: min-max normalised component index and eigenvalue.
- d_j: perpendicular distance from the point to the reference line y = 1 - x.
- k: selected elbow index.

Implementation:
- The scripts use the maximum perpendicular-distance heuristic.
- The selected index is converted to a one-based component count.
- The eigenvalue range is nonzero for the observed data.

Limitation:
- This is one operational definition of a scree elbow, not a uniquely established universal elbow rule.
- Report the exact heuristic rather than simply saying that an elbow was visually identified.

## 13. Permutation-based Horn parallel analysis

For permutation b and component j, let lambda_j^(b) be the j-th descending covariance eigenvalue from the independently column-permuted data.

r_j = Q_0.95(lambda_j^(1), ..., lambda_j^(B))

Retain consecutive components while:

lambda_j_observed > r_j

k = number of consecutive components satisfying the inequality, stopping at the first failure.

Symbols:
- B: number of random permutations; B=100 in this experiment.
- b: permutation index.
- j: component index.
- Q_0.95: empirical 95th percentile.
- r_j: random-data reference eigenvalue for component j.
- lambda_j_observed: observed-data eigenvalue.
- k: retained component count.

Implementation:
- Random seed: 42.
- Each feature column is independently permuted across samples.
- Covariance eigenvalues are calculated for each permuted matrix and sorted in descending order.
- The 95th percentile is calculated separately for each component.
- Retention stops at the first component that does not exceed its reference eigenvalue.

Method qualification:
- This is a permutation-based parallel-analysis implementation that preserves each feature's marginal values while disrupting cross-feature dependence.
- The number of permutations, percentile and random seed are experimental settings that can affect the estimate.
- Describe the implemented procedure precisely; do not imply that all parallel-analysis variants use this exact randomisation scheme.

## 14. Minka maximum-likelihood dimension selection

Implementation:
- sklearn.decomposition.PCA(n_components="mle", svd_solver="full").
- Fit on the fitting matrix only.
- scikit-learn version recorded for this project: 1.9.1.
- The dimension estimate is delegated to scikit-learn's implementation of Minka's probabilistic PCA model-selection method.

Observed result:
- Both centered and standardised arms returned k=783.
- This is a boundary result at the maximum dimension permitted by this MLE configuration for 784 input features.
- Do not interpret k=783 as proof that 783 dimensions are substantively meaningful.
- Do not invent or substitute a likelihood equation for the library implementation. If a detailed likelihood derivation is later included, it must be sourced and checked against the implemented method.

## 15. Verification status and remaining formula work

Verified directly against source code:
- Pixel scaling.
- Fitting-set centering.
- Population standard deviation (ddof=0).
- Zero-standard-deviation handling.
- SVD decomposition and saved component orientation.
- Explained variance using n-1.
- Explained-variance ratios and cumulative sums.
- PCA projection convention.
- Both explained-variance threshold implementations.
- Standardised Kaiser criterion.
- Centered covariance Kaiser analogue.
- Maximum-distance scree heuristic.
- Permutation-based parallel analysis settings and comparison rule.
- scikit-learn Minka MLE configuration and observed boundary result.

Still to verify and document before the formula register is complete:
- Logistic Regression objective and regularisation scaling as implemented by the installed scikit-learn version.
- Multiclass probability and prediction equations.
- Accuracy and any other final evaluation metrics actually used.
- Bootstrap confidence intervals or statistical comparisons, if included in the final protocol.
- Exact final experiment and validation-selection conventions.

This register records implementation conventions, not merely generic textbook formulas. Any later methodological change must be reflected here and in the relevant research record.

---

## Logistic Regression: Multiclass Objective, Probabilities, and Prediction

### Implementation verified

- Library: scikit-learn 1.9.1
- Estimator: `sklearn.linear_model.LogisticRegression`
- Solver: `lbfgs`
- Multiclass mode: multinomial
- Regularisation: L2 (`l1_ratio=0.0`; `penalty` is deprecated in this version)
- Inverse regularisation strength: `C=1.0`
- Intercept: enabled (`fit_intercept=True`)
- Maximum iterations: 5000 in the corrected pilot configuration
- Convergence tolerance: `1e-4`
- Sample weights: not supplied
- Class weights: `None`

### 1. Class scores

For sample i and class c:

\[
s_{ic}=\mathbf{w}_c^T\mathbf{z}_i+b_c
\]

where:

- \(\mathbf{z}_i\) is the PCA-transformed feature vector for sample i.
- \(\mathbf{w}_c\) is the coefficient vector for class c.
- \(b_c\) is the intercept for class c.
- \(s_{ic}\) is the unnormalised score for class c.

There are 10 classes in Fashion-MNIST.

### 2. Multiclass probabilities: softmax

\[
p_{ic}=
\frac{\exp(s_{ic})}
{\sum_{r=1}^{10}\exp(s_{ir})}
\]

where \(p_{ic}\) is the predicted probability that sample i belongs to class c.

The probabilities across all 10 classes for one sample sum to 1.

### 3. Multiclass cross-entropy

\[
L(W,b)=
-\frac{1}{n}
\sum_{i=1}^{n}
\log(p_{i,y_i})
\]

where:

- n is the number of fitting samples used to train the classifier.
- \(y_i\) is the true class index of sample i.
- \(p_{i,y_i}\) is the probability assigned to the true class.
- W is the matrix containing all class coefficient vectors.
- b is the vector of class intercepts.

This is the mean categorical cross-entropy used by the installed multinomial loss implementation.

### 4. L2 regularisation and complete objective

The installed scikit-learn implementation uses:

\[
\alpha=\frac{1}{Cn}
\]

and the L2 penalty:

\[
P(W)=\frac{\alpha}{2}\|W\|_F^2
\]

Therefore, the objective for this experiment is:

\[
J(W,b)=
-\frac{1}{n}
\sum_{i=1}^{n}\log(p_{i,y_i})
+
\frac{1}{2Cn}\|W\|_F^2
\]

For this project, \(C=1.0\), so:

\[
J(W,b)=
-\frac{1}{n}
\sum_{i=1}^{n}\log(p_{i,y_i})
+
\frac{1}{2n}\|W\|_F^2
\]

The Frobenius norm is:

\[
\|W\|_F^2=
\sum_{c=1}^{10}\sum_{j=1}^{k}w_{cj}^2
\]

where k is the number of input features supplied to the classifier.

**Intercept convention:** The intercept vector b is not included in the L2 penalty. This was verified from `LinearModelLoss.weight_intercept` and `LinearModelLoss.l2_penalty`.

### 5. Predicted class

\[
\hat{y}_i=\arg\max_{c\in\{1,\ldots,10\}}p_{ic}
\]

The predicted class is the class with the highest predicted probability.

### Implementation correspondence and verification

- Multinomial loss: `HalfMultinomialLoss`
- Probability calculation: `HalfMultinomialLoss.predict_proba`, using the softmax inverse link
- L2 penalty: `LinearModelLoss.l2_penalty`
- Intercept separation: `LinearModelLoss.weight_intercept`
- Regularisation scaling: `_logistic_regression_path`, with `l2_reg_strength = 1.0 / (C * sw_sum)`
- With no sample weights, `sw_sum = n`.
- The classifier is trained with `solver="lbfgs"`.

These formulas describe the installed scikit-learn implementation inspected in version 1.9.1. The objective is written as mean cross-entropy plus the corresponding L2 penalty. The intercept is unpenalised.



## Full Classifier Experiment: Validation Accuracy

The primary evaluation metric is classification accuracy.

\[
\mathrm{Accuracy}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbb{1}(\hat{y}_i=y_i)
\]

Where:
- \(N\): number of validation samples.
- \(\hat{y}_i\): predicted class for sample \(i\).
- \(y_i\): true class for sample \(i\).
- \(\mathbb{1}(\cdot)\): indicator function, equal to 1 when its
  condition is true and 0 otherwise.

Implementation: scikit-learn `accuracy_score(y_val, predictions)`.

Purpose: evaluate the fraction of correctly classified validation
samples for each Logistic Regression configuration.


## Validation Comparison: Accuracy Difference

For each dimension-selection rule, its validation accuracy was compared
with the best validation accuracy from the component grid in the same
preprocessing arm.

\[
\Delta_{\mathrm{pp}}
=
100\left(A_{\mathrm{rule}}-A_{\mathrm{grid,best}}\right)
\]

Where:
- \(A_{\mathrm{rule}}\): validation accuracy of the rule-selected model.
- \(A_{\mathrm{grid,best}}\): highest validation accuracy among grid models
  in the same preprocessing arm.
- \(\Delta_{\mathrm{pp}}\): difference expressed in percentage points.

Positive values indicate that the rule-selected configuration exceeded
the grid result; negative values indicate lower validation accuracy.
This is a descriptive comparison, not a statistical significance test.
