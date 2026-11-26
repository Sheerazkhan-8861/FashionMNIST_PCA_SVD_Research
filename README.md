# Do Rules of Thumb Pick the Right k? Comparing PCA Dimension-Selection Rules Against Downstream Accuracy on Fashion-MNIST

## 1. Project Overview

Principal Component Analysis (PCA) is widely used to reduce the dimensionality of data before applying machine learning algorithms. Several established statistical rules are available for choosing how many principal components to retain. However, a dimension that satisfies a statistical selection criterion does not necessarily produce the best predictive performance for a downstream task.

This project investigates that relationship using the Fashion-MNIST image-classification dataset.

Six PCA dimension-selection methods are compared under two preprocessing approaches: centered-only and standardized data. Multinomial Logistic Regression is then used to evaluate the classification performance associated with the selected dimensions.

The central research question is:

**Do commonly used PCA dimension-selection rules identify a number of components that performs well for downstream classification?**

The study compares statistical dimension-selection outcomes with validation accuracy and evaluates the fixed experimental configurations on the official held-out test set.

## 2. Research Objectives

The objectives are:

1. Apply PCA using a full Singular Value Decomposition (SVD) implementation.
2. Compare centered-only and standardized preprocessing.
3. Evaluate six PCA dimension-selection methods.
4. Compare rule-selected dimensions against a grid of candidate component counts.
5. Evaluate the downstream classification performance of the selected dimensions.
6. Assess the fixed configurations using the official Fashion-MNIST test set.
7. Document the mathematical formulations, implementation decisions, experimental protocol, and results.

## 3. Dataset

The experiments use **Fashion-MNIST**, a dataset of grayscale clothing images.

| Dataset property | Value |
|---|---:|
| Original training images | 60,000 |
| Official test images | 10,000 |
| Image dimensions | 28 × 28 |
| Features per image | 784 |
| Number of classes | 10 |

Each image is represented as a vector containing 784 pixel-intensity features.

Dataset source: [Fashion-MNIST GitHub repository](https://github.com/zalandoresearch/fashion-mnist).

### 3.1 Experimental Data Split

The original 60,000-image training set was divided using stratified sampling with random seed 42.

| Subset | Samples |
|---|---:|
| Fitting set | 50,000 |
| Validation set | 10,000 |
| Official test set | 10,000 |

The fitting and validation subsets contain 5,000 and 1,000 samples per class, respectively.

The official test set was kept separate from model development and used for the final evaluation of the fixed configurations.

## 4. Methodology

### 4.1 Data Preprocessing

Two preprocessing approaches were investigated.

**Centered-only preprocessing**

1. Convert pixel intensities to floating-point values.
2. Scale pixel values by dividing by 255.
3. Calculate feature means using the fitting set.
4. Center fitting and validation data using those fitting-set means.

**Standardized preprocessing**

1. Apply pixel scaling and centering.
2. Calculate population standard deviations using the fitting set.
3. Replace zero standard deviations with 1 before division.
4. Standardize fitting and validation data using the fitting-set statistics.

All preprocessing statistics were fitted using the fitting set only. Validation data was transformed using the same fitting-set statistics.

### 4.2 PCA Using SVD

PCA was implemented through full SVD of the preprocessed fitting data:

\[
X_{\mathrm{fit}} = U\Sigma V^T
\]

The principal components were obtained from the rows of \(V^T\). Explained variance, explained-variance ratios, and cumulative explained variance were calculated from the singular values.

PCA was fitted on the fitting data only. Validation data was projected using the resulting components.

### 4.3 PCA Dimension-Selection Methods

Six dimension-selection methods were compared.

| Method | Selection principle |
|---|---|
| 90% explained variance | Retain enough components to reach 90% cumulative explained variance |
| 95% explained variance | Retain enough components to reach 95% cumulative explained variance |
| Kaiser criterion | Apply the relevant eigenvalue threshold |
| Scree elbow | Select the elbow of the scree curve |
| Horn's parallel analysis | Compare observed eigenvalues with a permutation-based reference distribution |
| Minka's MLE | Estimate dimensionality using a maximum-likelihood approach |

Horn's parallel analysis used 100 permutations, random seed 42, independent permutation of feature columns, and the 95th-percentile reference threshold.

For centered data, the Kaiser result is a covariance-based analogue using an eigenvalue threshold greater than the mean eigenvalue. For standardized data, the conventional eigenvalue-greater-than-one criterion was used.

Minka's MLE returned \(k=783\) in both preprocessing conditions. This is a boundary result and should not be interpreted as evidence that all 783 components are necessary or meaningful.

### 4.4 Classification Model

Multinomial Logistic Regression was used to evaluate downstream classification accuracy.

The full validation experiment used:

- Solver: `lbfgs`
- Maximum iterations: 5,000
- Random state: 42
- Scikit-learn default regularization settings
- Single-threaded BLAS execution

A grid of 23 candidate component counts was evaluated for each preprocessing approach. The six dimension-selection methods and the original 784-feature baseline were also evaluated.

The full experiment produced 59 unique model configurations.

### 4.5 Validation Protocol

Preprocessing and PCA fitting were restricted to the 50,000-sample fitting set. The resulting configurations were evaluated on the 10,000-sample validation set.

Validation results were used to compare configurations. The official test set was not used for model selection or tuning.

### 4.6 Final Held-Out Test Evaluation

After the experimental configurations were fixed, all 16 designated configurations were evaluated on the official 10,000-sample test set.

The final evaluation generated prediction files, confusion matrices, and test metrics.

All 16 configurations completed without convergence warnings. Saved prediction and confusion-matrix records were checked against the reported metrics.

The test set was used only for the final evaluation. Its results were not used for further model selection or tuning.

## 5. Dimension-Selection Results

The selected component counts were:

| Selection method | Centered | Standardized |
|---|---:|---:|
| 90% explained variance | 84 | 137 |
| 95% explained variance | 187 | 256 |
| Kaiser | 58 | 80 |
| Scree elbow | 15 | 21 |
| Horn's parallel analysis | 39 | 67 |
| Minka's MLE | 783 | 783 |

The selected dimensions differ considerably across methods and preprocessing approaches. Their downstream classification performance therefore needs to be evaluated rather than assuming that a particular statistical rule automatically identifies the most predictive dimension.

## 6. Validation Results

The best validation grid configurations were:

| Preprocessing | Best grid dimension | Validation accuracy |
|---|---:|---:|
| Centered | 148 | 86.41% |
| Standardized | 258 | 86.29% |

The dimension-selection rules produced the following results:

| Method | Centered k | Centered accuracy | Standardized k | Standardized accuracy |
|---|---:|---:|---:|---:|
| 90% explained variance | 84 | 85.44% | 137 | 85.62% |
| 95% explained variance | 187 | 86.54% | 256 | 86.21% |
| Kaiser | 58 | 84.80% | 80 | 85.31% |
| Scree elbow | 15 | 79.89% | 21 | 81.20% |
| Horn's parallel analysis | 39 | 83.97% | 67 | 85.02% |
| Minka's MLE | 783 | 86.27% | 783 | 85.48% |

The centered 95% explained-variance configuration achieved 86.54%, slightly above the best centered grid configuration at 86.41%.

For standardized data, the 95% explained-variance configuration achieved 86.21%, compared with 86.29% for the best grid configuration.

These comparisons are descriptive. They do not establish statistical significance or universal superiority.

## 7. Final Held-Out Test Results

The best test accuracy among the 16 evaluated configurations was achieved by standardized preprocessing with the 95% explained-variance selection rule.

| Configuration | Components | Test accuracy |
|---|---:|---:|
| Centered baseline | 784 | 84.25% |
| Centered best validation grid | 148 | 84.53% |
| Centered 90% explained variance | 84 | 83.81% |
| Centered 95% explained variance | 187 | 84.41% |
| Centered Kaiser | 58 | 83.00% |
| Centered scree elbow | 15 | 78.32% |
| Centered Horn's parallel analysis | 39 | 82.33% |
| Centered Minka's MLE | 783 | 84.19% |
| Standardized baseline | 784 | 83.22% |
| Standardized best validation grid | 258 | 84.53% |
| Standardized 90% explained variance | 137 | 84.14% |
| Standardized 95% explained variance | 256 | **84.55%** |
| Standardized Kaiser | 80 | 83.47% |
| Standardized scree elbow | 21 | 79.64% |
| Standardized Horn's parallel analysis | 67 | 83.03% |
| Standardized Minka's MLE | 783 | 83.19% |

The best test accuracy was **84.55%**, obtained using standardized preprocessing and 256 components selected by the 95% explained-variance rule.

The results show that classification performance varies with both the selection method and preprocessing approach. The scree-elbow configurations performed less well than the stronger configurations in this experiment, while the 95% explained-variance configurations performed competitively.

Small differences between configurations should not be interpreted as statistically significant. No statistical significance testing was performed.

## 8. Figures

The project includes four generated figures.

### PCA Scree Plots

- `outputs/figures/scree_plot.png` — Scree plot for centered data.
- `outputs/figures/scree_plot_standardized.png` — Scree plot for standardized data.

### Classification Performance

- `outputs/figures/validation_accuracy_by_components.png` — Validation accuracy across candidate component counts.
- `outputs/figures/selection_rule_accuracy_difference.png` — Validation accuracy differences between selection rules and the best grid configuration.

## 9. Repository Structure

The repository is organized as follows:

```text
FashionMNIST_PCA_SVD_Research/
├── src/
│   ├── preprocess_data.py
│   ├── select_scree_elbow.py
│   ├── select_scree_elbow_standardized.py
│   ├── run_full_classifier_experiment.py
│   ├── analyze_validation_results.py
│   └── run_final_test_evaluation.py
├── reports/
│   ├── pca_dimension_selection_research_record.md
│   ├── full_classifier_experiment_protocol.md
│   ├── full_classifier_experiment_record.md
│   └── mathematical_formulas_and_notation.md
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── pilot/
│   ├── full_experiment/
│   └── test_evaluation/
├── data/
│   ├── raw/
│   └── processed/
├── requirements.txt
├── .gitignore
└── README.md
```

Raw and processed datasets are stored locally and excluded from version control. The output directories contain generated experimental results and records.

## 10. Research Documentation

Detailed methodology and verification are maintained in separate research records:

- `reports/pca_dimension_selection_research_record.md` — PCA implementation and dimension-selection methods.
- `reports/full_classifier_experiment_protocol.md` — Frozen full-experiment protocol.
- `reports/full_classifier_experiment_record.md` — Validation analysis and final held-out test evaluation.
- `reports/mathematical_formulas_and_notation.md` — Mathematical definitions, equations, notation, and implementation correspondence.

The mathematical formula register preserves the detailed formulations and their correspondence with the actual implementation.

## 11. Reproducibility and Environment

The experiments were developed in a Python virtual environment.

The recorded software environment included:

- Python 3.14.4
- NumPy 2.5.5
- Pandas 3.1.1
- SciPy 1.18.1
- scikit-learn 1.9.1
- Matplotlib 3.11.2
- Seaborn 0.13.2

To create a virtual environment and install dependencies:

1. Create the environment using `python -m venv .venv`.
2. Activate it using `source .venv/bin/activate`.
3. Install dependencies using `pip install -r requirements.txt`.

The raw and processed dataset files are not included in the repository. Obtain Fashion-MNIST from its official source and follow the preprocessing and experiment scripts in `src/`.

For exact reproduction, use the recorded software versions, random seeds, and experimental protocol. The official test evaluation is the final evaluation stage, not a model-selection step.

## 12. Research Limitations

- The experiments use Fashion-MNIST and multinomial Logistic Regression. Results may differ with other datasets or classifiers.
- Accuracy differences are reported descriptively. Statistical significance testing was not performed.
- Per-sample validation predictions were not retained for paired bootstrap confidence intervals or McNemar's test.
- Minka's MLE reached the \(k=783\) boundary in both preprocessing conditions.
- The results do not establish that one dimension-selection method is universally best.

## 13. Project Status

The following experimental stages have been completed:

- [x] Dataset preparation and stratified fitting/validation split
- [x] Centered-only and standardized preprocessing
- [x] PCA using full SVD
- [x] Six dimension-selection methods
- [x] Logistic Regression pilot experiment
- [x] Full validation experiment
- [x] Validation results analysis and figures
- [x] Final held-out test evaluation
- [x] Test prediction and confusion-matrix verification
- [x] Mathematical and implementation documentation

## 14. Conclusion

This project examines whether established PCA dimension-selection rules identify component counts that work well for downstream classification on Fashion-MNIST.

The results demonstrate that dimension-selection outcomes depend on the selection method and preprocessing approach. Some rules produced competitive classification accuracy, while others retained fewer components and performed less well in this experiment.

The findings emphasize the importance of evaluating dimensionality-reduction choices in the context of the downstream task while keeping model selection and final test evaluation properly separated.
