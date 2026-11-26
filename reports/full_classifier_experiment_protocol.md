# Full Classifier Experiment Protocol

## 1. Research Objective

Evaluate whether PCA dimension-selection rules identify component counts
that provide strong downstream Logistic Regression classification
performance on Fashion-MNIST.

## 2. Dataset and Split

- Dataset: Fashion-MNIST
- Fitting set: 50,000 samples
- Validation set: 10,000 samples
- Official test set: 10,000 samples
- The original training set was stratified into fitting and validation sets
  using random seed 42.
- The official test set remains locked until final authorised evaluation.

## 3. Preprocessing Conditions

Two experimental arms:

1. Scaled and centered.
2. Standardised.

All preprocessing parameters and PCA directions are derived from the
full fitting set only. Validation data is transformed using the
corresponding fitting-set parameters and PCA directions.

## 4. Classifier

- Model: scikit-learn LogisticRegression
- Solver: lbfgs
- Maximum iterations: 5,000
- Random state: 42
- Other parameters: scikit-learn defaults
- BLAS thread limit: 1
- Convergence warnings: captured and recorded

## 5. Experiment Matrix

For each preprocessing arm:

- One original 784-feature baseline.
- The 23 component counts in the established pilot grid.
- Six component counts selected by the predefined dimension-selection
  methods.

The six selection methods are:

1. 90% explained variance
2. 95% explained variance
3. Kaiser criterion
4. Scree elbow
5. Horn's parallel analysis
6. Minka's MLE

The standardised Scree elbow value (k=21) overlaps with the pilot grid.
Its classifier result will be reused rather than fitted twice.

Total planned unique classifier fits: 59.

## 6. Evaluation

Primary metric: validation accuracy.

For every unique fit, record:

- Preprocessing arm
- Representation type
- Component count
- Validation accuracy
- Training time in seconds
- Number of iterations
- Convergence warning and warning details

The best grid component count will be selected separately for each
preprocessing arm using validation accuracy.

Dimension-selection methods will be compared against the corresponding
validation results.

## 7. Data Separation and Research Integrity

- Classifiers are trained using the full 50,000-sample fitting set.
- Validation data is used for model comparison and component selection.
- The official test set must not influence preprocessing, PCA fitting,
  component selection, or model selection.
- No statistical significance claim will be made without an appropriate
  statistical analysis.
- Pilot results remain separate from full-experiment results.

## 8. Protocol Status

Approved and frozen on 2026-11-26.

The full classifier experiment was executed under this frozen protocol. The final held-out test evaluation was subsequently completed and audited on 2026-11-26.
