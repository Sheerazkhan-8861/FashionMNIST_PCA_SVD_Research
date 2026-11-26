# Full Logistic Regression Classifier Experiment Record

## 1. Research Objective

Evaluate how PCA dimension-selection rules affect downstream
Fashion-MNIST classification accuracy under two preprocessing
approaches.

## 2. Experimental Dataset

- Dataset: Fashion-MNIST
- Fitting samples: 50,000
- Validation samples: 10,000
- Official test samples: 10,000
- The official test set was not used.
- The fitting and validation sets were established before this experiment.

## 3. Preprocessing and PCA

Two preprocessing arms were evaluated:

1. Scaled and centered.
2. Standardised.

Preprocessing parameters and PCA directions were estimated using the
full fitting set. Validation samples were transformed using the
corresponding fitted preprocessing parameters and PCA directions.

Previously saved PCA/SVD artifacts were reused. PCA was not refitted
during classifier evaluation.

## 4. Classifier Configuration

- Classifier: scikit-learn LogisticRegression
- Solver: lbfgs
- Maximum iterations: 5,000
- Random state: 42
- Other parameters: scikit-learn defaults
- BLAS thread limit: 1
- Execution: sequential
- Convergence warnings: captured and recorded

## 5. Experiment Matrix

| Experiment | Scaled and Centered | Standardised |
|---|---:|---:|
| Original 784-feature baseline | 1 | 1 |
| PCA grid | 23 | 23 |
| Additional rule-selected dimensions | 6 | 5 |
| Total unique fits | 30 | 29 |

Total unique classifier fits: 59.

The standardised Scree Elbow selection (k=21) overlapped with the
component grid and was evaluated once.

## 6. Evaluation Metric

The primary metric was validation accuracy:

Accuracy = number of correct validation predictions / total validation predictions.

Training time, iteration count, and convergence warnings were also
recorded.

## 7. Full Experiment Results

### 7.1 Best grid configurations

| Measurement | Scaled and Centered | Standardised |
|---|---:|---:|
| Best grid component count | 148 | 258 |
| Best grid validation accuracy | 0.8641 | 0.8629 |
| Convergence warnings | 0 | 0 |

### 7.2 Best configurations across all evaluated dimensions

| Preprocessing | Representation | Components | Validation accuracy |
|---|---|---:|---:|
| Scaled and Centered | PCA, 95% explained variance | 187 | 0.8654 |
| Standardised | PCA grid | 258 | 0.8629 |

### 7.3 Original baseline comparison

| Preprocessing | Features | Validation accuracy | Training time (seconds) |
|---|---:|---:|---:|
| Scaled and Centered | 784 | 0.8614 | 16.39 |
| Standardised | 784 | 0.8549 | 77.55 |

### 7.4 Best rule-selected configurations

| Preprocessing | Method | k | Validation accuracy |
|---|---|---:|---:|
| Scaled and Centered | 90% explained variance | 84 | 0.8544 |
| Scaled and Centered | 95% explained variance | 187 | 0.8654 |
| Scaled and Centered | Kaiser covariance analogue | 58 | 0.8480 |
| Scaled and Centered | Scree elbow | 15 | 0.7989 |
| Scaled and Centered | Horn's parallel analysis | 39 | 0.8397 |
| Scaled and Centered | Minka's MLE | 783 | 0.8627 |
| Standardised | 90% explained variance | 137 | 0.8562 |
| Standardised | 95% explained variance | 256 | 0.8621 |
| Standardised | Kaiser eigenvalue > 1 | 80 | 0.8531 |
| Standardised | Scree elbow | 21 | 0.8120 |
| Standardised | Horn's parallel analysis | 67 | 0.8502 |
| Standardised | Minka's MLE | 783 | 0.8548 |

## 8. Results Audit

The saved results file was independently audited.

- Total rows: 59
- Unique task IDs: 59
- Duplicate task IDs: 0
- Scaled-and-centered fits: 30
- Standardised fits: 29
- Convergence warnings: 0
- Missing values in experimental result fields: 0

The 47 empty Selection_Methods fields correspond to configurations
not directly selected by a dimension-selection rule.

The 59 empty Warning_Details fields correspond to the absence of
convergence warnings.

Validation accuracy range across all fits: 0.2739 to 0.8654.

## 9. Interpretation

The scaled-and-centered arm achieved the highest observed validation
accuracy of 0.8654 at 187 components, selected by the 95% explained
variance rule.

Its best grid configuration achieved 0.8641 at 148 components.

The standardised arm achieved its highest observed validation accuracy
of 0.8629 at 258 components in the component grid. Its 95% explained
variance selection achieved 0.8621 at 256 components.

The original scaled-and-centered baseline achieved 0.8614, while the
standardised baseline achieved 0.8549.

These results suggest that dimension-selection rules differ in their
downstream classification performance. The 95% explained variance
rule performed particularly well in the scaled-and-centered arm.
However, validation accuracy alone does not establish statistical
significance or generalisation to unseen test data.

## 10. Limitations

1. Validation data was used to compare configurations and select
   component counts.
2. The reported best configurations are validation-selected results,
   not independent test estimates.
3. No confidence intervals or statistical significance tests were
   computed in this experiment.
4. The experiment uses Logistic Regression as its downstream classifier;
   conclusions should not automatically be generalised to other models.
5. Minka's MLE selected 783 components, a boundary result that should
   be interpreted cautiously.
6. The centered Kaiser method is a covariance analogue rather than
   the conventional standardised Kaiser criterion.
7. Results depend on the fixed classifier settings and component grid.

## 11. Research Integrity

- The official Fashion-MNIST test set was not used.
- Preprocessing and PCA were fitted on the fitting set only.
- Validation data was transformed using fitting-set parameters.
- All 59 classifier fits completed without convergence warnings.
- The pilot experiment remains separately preserved.
- No statistical significance claim is made.

## 12. Saved Artifact

Full experiment results:

outputs/full_experiment/full_classifier_results.csv

## 13. Status

Full classifier experiment completed and audited on 2026-11-26.

The final authorised evaluation of all 16 model configurations was completed and audited on 2026-11-26. See outputs/test_evaluation/test_metrics.csv.


## 14. Validation Selection-Rule Comparison

The 12 rule-selected configurations were compared with the best
component-grid result in their respective preprocessing arms.
Differences are expressed in percentage points (pp).

### 14.1 Scaled and Centered

Best grid result: k=148, validation accuracy 0.8641.

| Selection method | k | Accuracy | Difference (pp) |
|---|---:|---:|---:|
| 95% explained variance | 187 | 0.8654 | +0.13 |
| Minka's MLE | 783 | 0.8627 | -0.14 |
| 90% explained variance | 84 | 0.8544 | -0.97 |
| Kaiser covariance analogue | 58 | 0.8480 | -1.61 |
| Horn's parallel analysis | 39 | 0.8397 | -2.44 |
| Scree elbow | 15 | 0.7989 | -6.52 |

### 14.2 Standardised

Best grid result: k=258, validation accuracy 0.8629.

| Selection method | k | Accuracy | Difference (pp) |
|---|---:|---:|---:|
| 95% explained variance | 256 | 0.8621 | -0.08 |
| 90% explained variance | 137 | 0.8562 | -0.67 |
| Minka's MLE | 783 | 0.8548 | -0.81 |
| Kaiser eigenvalue > 1 | 80 | 0.8531 | -0.98 |
| Horn's parallel analysis | 67 | 0.8502 | -1.27 |
| Scree elbow | 21 | 0.8120 | -5.09 |

### 14.3 Interpretation

The 95% explained-variance rule achieved the strongest result among
the selection rules in both preprocessing arms.

In the scaled-and-centered arm, its result at k=187 (0.8654)
exceeded the best grid result at k=148 (0.8641) by 0.13 percentage
points. In the standardised arm, its result at k=256 (0.8621)
was 0.08 percentage points below the grid result at k=258 (0.8629).

Scree elbow produced the largest accuracy reductions relative to the
grid optimum in both arms. Minka's MLE retained 783 components and
therefore provided very limited dimensionality reduction.

These comparisons are descriptive. All configurations were evaluated
on the same validation set, which was also used to identify the best
configurations. The small observed differences must not be interpreted
as statistically significant or as independent test performance.

### 14.4 Analysis Artifacts

- Comparison table:
  outputs/tables/validation_selection_rule_comparison.csv
- Accuracy across component counts:
  outputs/figures/validation_accuracy_by_components.png
- Rule-versus-grid accuracy differences:
  outputs/figures/selection_rule_accuracy_difference.png
- Reproducible analysis script:
  src/analyze_validation_results.py

No classifier was retrained during this analysis. At the time this validation analysis was conducted, the official test set remained unused and locked.


## 15. Statistical Uncertainty and Inference Limitations

The experiment saved aggregate validation accuracies but did not save
per-sample predictions or predicted probabilities.

Consequently:

1. Paired bootstrap confidence intervals cannot be calculated from
   the currently saved results.
2. McNemar's test cannot be performed because paired correctness
   outcomes for individual validation samples are unavailable.
3. Aggregate accuracy differences alone do not establish statistical
   significance.
4. The 0.13 percentage-point advantage of the centered 95% explained
   variance configuration over the best centered grid configuration
   must be treated as a descriptive observation, not evidence of
   superiority.
5. The validation set was used to compare configurations and identify
   strong-performing dimensions. This introduces selection-related
   optimism into the reported best validation result.

No additional classifier training was performed to recover predictions.
The official test set was evaluated in the final authorised evaluation. All 16 prediction files and confusion matrices were verified against the saved test metrics. No further model selection or tuning was performed using test results.

The validation comparisons are descriptive. Statistical uncertainty and significance of differences between configurations have not been established. The official held-out test evaluation provides an independent performance assessment of the 16 fixed configurations; those results are reported in Section 16. No statistical significance or universal superiority is inferred from the test results.

## 16. Final Held-Out Test Evaluation

The final authorised evaluation was completed on 2026-11-26 using the
official Fashion-MNIST test set of 10,000 samples.

The preprocessing transformations and PCA representations were fitted
using the 50,000-sample fitting set. The official test set was used only
for this final evaluation. No model selection or tuning was performed
using test results.

### 16.1 Test Results

| Preprocessing | Configuration | Components | Test Accuracy |
|---|---|---:|---:|
| Scaled and Centered | Original baseline | 784 | 0.8425 |
| Scaled and Centered | Best validation grid | 148 | 0.8453 |
| Scaled and Centered | 90% Explained Variance | 84 | 0.8381 |
| Scaled and Centered | 95% Explained Variance | 187 | 0.8441 |
| Scaled and Centered | Kaiser (Covariance Analogue) | 58 | 0.8300 |
| Scaled and Centered | Scree Elbow | 15 | 0.7832 |
| Scaled and Centered | Horn's Parallel Analysis | 39 | 0.8233 |
| Scaled and Centered | Minka's MLE | 783 | 0.8419 |
| Standardised | Original baseline | 784 | 0.8322 |
| Standardised | Best validation grid | 258 | 0.8453 |
| Standardised | 90% Explained Variance | 137 | 0.8414 |
| Standardised | 95% Explained Variance | 256 | 0.8455 |
| Standardised | Kaiser (Eigenvalue > 1) | 80 | 0.8347 |
| Standardised | Scree Elbow | 21 | 0.7964 |
| Standardised | Horn's Parallel Analysis | 67 | 0.8303 |
| Standardised | Minka's MLE | 783 | 0.8319 |

### 16.2 Evaluation Integrity Checks

All 16 model configurations were verified.

- 16 unique evaluation records were present.
- Each prediction file contained exactly 10,000 test predictions.
- Test indices covered 0 through 9,999 in every prediction file.
- Prediction-derived accuracies matched the saved test metrics.
- All confusion matrices contained 10,000 observations.
- Confusion-matrix-derived accuracies matched the saved test metrics.
- No duplicate metric records were found.
- No convergence warnings were recorded.

All consistency checks passed.

### 16.3 Main Test Observation

The highest observed test accuracy was 0.8455 (84.55%), obtained
using standardised preprocessing and PCA with 256 components selected
by the 95% explained-variance rule.

The scaled-and-centered 95% explained-variance configuration achieved
0.8441 (84.41%). The best validation-grid configurations in the two
preprocessing arms each achieved 0.8453 (84.53%) on the test set.

These are descriptive results for the evaluated configurations.
No statistical significance or universal superiority is claimed.

### 16.4 Saved Test Evaluation Artifacts

- Test metrics: outputs/test_evaluation/test_metrics.csv
- Per-model predictions: outputs/test_evaluation/predictions/
- Per-model confusion matrices: outputs/test_evaluation/confusion_matrices/
- Evaluation script: src/run_final_test_evaluation.py

The validation inference limitations in Section 15 remain applicable
to validation comparisons. The separately reported held-out test
results provide an independent evaluation on the official test set.
