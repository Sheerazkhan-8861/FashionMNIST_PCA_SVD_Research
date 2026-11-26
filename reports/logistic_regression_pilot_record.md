# Logistic Regression Pilot Experiment Record

## 1. Purpose

This pilot evaluates Logistic Regression performance across PCA component
counts under two preprocessing approaches:

1. Scaled and centered.
2. Standardised.

The purpose is to test the experimental pipeline and identify a reasonable
component-count search range before conducting full-scale classifier
experiments.

## 2. Experimental Design

- Dataset: Fashion-MNIST
- Pilot training subset: 20,000 samples
- Validation set: 10,000 samples
- Pilot subset selection: Stratified
- Random seed: 42
- Number of evaluated component counts: 23
- Component grid:
  [1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 28, 37, 49, 64, 85,
   112, 148, 196, 258, 341, 450, 594, 784]
- PCA fitting: Pilot training subset only
- Validation transformation: Same fitted preprocessing statistics
  and PCA directions as the pilot training subset.

Preprocessing statistics were estimated using the pilot training subset
only, not the validation set.

## 3. Classifier Configuration

- Classifier: Logistic Regression
- Solver: lbfgs
- Maximum iterations: 5,000
- Random state: 42
- Other Logistic Regression parameters: scikit-learn defaults
- BLAS thread limit: 1
- Convergence warnings: Captured and recorded

The corrected experiment used identical classifier settings for both
preprocessing arms.

## 4. Initial Pilot Issue

The initial pilot used max_iter=1,000.

- Scaled and centered: 0 convergence warnings across 23 runs.
- Standardised: 12 convergence warnings across 23 runs.

The initial standardised results were therefore treated as preliminary
because some classifier fits had not converged.

The initial results were preserved in:

outputs/pilot/logistic_regression_pilot_initial.csv

## 5. Corrected Pilot Results

The maximum iteration limit was increased to 5,000 for both preprocessing
arms. All other experimental settings were retained.

| Measurement | Scaled and Centered | Standardised |
|---|---:|---:|
| Number of runs | 23 | 23 |
| Best component count | 258 | 258 |
| Best validation accuracy | 0.8586 | 0.8555 |
| Convergence warnings | 0 | 0 |
| Maximum iterations used | 408 | 1914 |

Total classifier runs: 46

All 46 runs completed without convergence warnings.

Corrected results file:

outputs/pilot/logistic_regression_pilot_convergence_checked.csv

## 6. Interpretation

Both preprocessing arms achieved their highest observed pilot validation
accuracy at 258 components.

The scaled-and-centered arm achieved 85.86%, while the standardised arm
achieved 85.55%. The observed difference is 0.31 percentage points.

These findings are pilot observations only. They do not establish that
258 is the final optimal component count or that one preprocessing
approach is generally superior.

## 7. Limitations

1. The pilot uses 20,000 training samples rather than the full 50,000
   fitting samples.
2. The component grid contains 23 unique values, although the
   configuration requested 25 logarithmically spaced values before
   duplicate removal.
3. The validation set was used to compare component counts.
4. The official Fashion-MNIST test set was not used.
5. Final classifier comparisons must be conducted using the full
   fitting set and the predefined experimental protocol.
6. Pilot results must not be presented as final experimental results.

## 8. Research Integrity and Data Separation

- The official test set remains untouched.
- PCA was fitted on the pilot training subset only.
- Pilot preprocessing statistics were fitted on the pilot training subset.
- The same validation set and classifier configuration were used for
  both preprocessing arms.
- Initial and corrected pilot outputs were preserved separately.

## 9. Next Stage

Proceed to the planned full classifier evaluation after confirming the
classifier protocol and computational feasibility.

The official test set remains locked until final authorised evaluation.
