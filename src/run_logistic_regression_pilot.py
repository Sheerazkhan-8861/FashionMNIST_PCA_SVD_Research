from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits


# --------------------------------------------------
# Configuration
# --------------------------------------------------

RANDOM_SEED = 42
PILOT_SIZE = 20_000
N_COMPONENTS = 25
MAX_ITER = 5000

DATA_PATH = Path("data/processed/fashion_mnist_split.npz")
OUTPUT_DIR = Path("outputs/pilot")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load original fitting and validation data
# --------------------------------------------------

with np.load(DATA_PATH, allow_pickle=False) as data:
    X_fit_raw = data["X_fit"].astype(np.float64) / 255.0
    X_val_raw = data["X_val"].astype(np.float64) / 255.0
    y_fit = data["y_fit"]
    y_val = data["y_val"]


# --------------------------------------------------
# Create fixed stratified pilot subset
# --------------------------------------------------

all_indices = np.arange(len(y_fit))

pilot_indices, _ = train_test_split(
    all_indices,
    train_size=PILOT_SIZE,
    random_state=RANDOM_SEED,
    stratify=y_fit,
)

X_pilot_raw = X_fit_raw[pilot_indices]
y_pilot = y_fit[pilot_indices]

print("LOGISTIC REGRESSION PILOT")
print("-" * 55)
print(f"Pilot training shape: {X_pilot_raw.shape}")
print(f"Validation shape: {X_val_raw.shape}")
print(f"Pilot class counts: {np.bincount(y_pilot, minlength=10)}")


# --------------------------------------------------
# Define the logarithmic component grid
# --------------------------------------------------

component_grid = np.unique(
    np.rint(
        np.geomspace(1, 784, N_COMPONENTS)
    ).astype(int)
)

print(f"\nNumber of component values: {len(component_grid)}")
print(f"Component grid: {component_grid}")


# --------------------------------------------------
# Run both preprocessing arms
# --------------------------------------------------

all_results = []

preprocessing_arms = [
    "Scaled and Centered",
    "Standardised",
]

for arm in preprocessing_arms:

    print(f"\n{'=' * 55}")
    print(f"PREPROCESSING: {arm}")
    print("=" * 55)

    # Fit preprocessing statistics on pilot training data only.
    pilot_mean = X_pilot_raw.mean(axis=0)

    if arm == "Scaled and Centered":
        pilot_scale = np.ones(X_pilot_raw.shape[1])

    else:
        pilot_scale = X_pilot_raw.std(axis=0)
        pilot_scale[pilot_scale == 0] = 1.0

    X_pilot = (X_pilot_raw - pilot_mean) / pilot_scale
    X_validation = (X_val_raw - pilot_mean) / pilot_scale

    # Fit PCA only on the pilot training subset.
    pca = PCA(
        n_components=None,
        svd_solver="full",
    )

    X_pilot_pca = pca.fit_transform(X_pilot)
    X_validation_pca = pca.transform(X_validation)

    print(f"PCA fitting shape: {X_pilot.shape}")
    print(f"PCA transformed training shape: {X_pilot_pca.shape}")
    print(f"PCA transformed validation shape: {X_validation_pca.shape}")

    for k in component_grid:

        X_train_k = X_pilot_pca[:, :k]
        X_val_k = X_validation_pca[:, :k]

        model = LogisticRegression(
            solver="lbfgs",
            max_iter=MAX_ITER,
            random_state=RANDOM_SEED,
        )

        start_time = time.time()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            with threadpool_limits(limits=1):
                model.fit(X_train_k, y_pilot)

        training_time = time.time() - start_time

        with threadpool_limits(limits=1):
            predictions = model.predict(X_val_k)

        accuracy = accuracy_score(y_val, predictions)

        convergence_warnings = [
            str(w.message)
            for w in caught
            if "converg" in str(w.message).lower()
        ]

        result = {
            "Preprocessing": arm,
            "Components": int(k),
            "Validation_Accuracy": float(accuracy),
            "Training_Time_Seconds": float(training_time),
            "Convergence_Warning": bool(convergence_warnings),
            "Warning_Details": " | ".join(convergence_warnings),
            "Iterations": int(np.max(model.n_iter_)),
        }

        all_results.append(result)

        print(
            f"k={k:3d} | "
            f"Accuracy={accuracy:.4f} | "
            f"Time={training_time:.2f}s | "
            f"Iterations={result['Iterations']} | "
            f"Convergence warning={result['Convergence_Warning']}"
        )


# --------------------------------------------------
# Save pilot results
# --------------------------------------------------

results_df = pd.DataFrame(all_results)

results_path = OUTPUT_DIR / "logistic_regression_pilot_convergence_checked.csv"
results_df.to_csv(results_path, index=False)

print("\n" + "=" * 55)
print("PILOT RESULTS SUMMARY")
print("=" * 55)

for arm in preprocessing_arms:
    arm_results = results_df[
        results_df["Preprocessing"] == arm
    ]

    best_row = arm_results.loc[
        arm_results["Validation_Accuracy"].idxmax()
    ]

    print(f"\n{arm}")
    print(f"Best k: {best_row['Components']}")
    print(f"Best validation accuracy: {best_row['Validation_Accuracy']:.4f}")
    print(f"Convergence warning: {best_row['Convergence_Warning']}")

print(f"\nSaved results: {results_path}")
print("\nLOGISTIC REGRESSION PILOT COMPLETED")
