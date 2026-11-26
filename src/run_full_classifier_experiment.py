from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from threadpoolctl import threadpool_limits


# --------------------------------------------------
# Configuration
# --------------------------------------------------

RANDOM_SEED = 42
MAX_ITER = 5000

PREPROCESSING_CONFIGS = {
    "Scaled and Centered": {
        "data": Path("data/processed/fashion_mnist_scaled_centered.npz"),
        "pca": Path("outputs/pca_svd_results.npz"),
        "selection": Path("outputs/tables/pca_selection_centered.csv"),
    },
    "Standardised": {
        "data": Path("data/processed/fashion_mnist_standardized.npz"),
        "pca": Path("outputs/pca_svd_standardized_results.npz"),
        "selection": Path("outputs/tables/pca_selection_standardized.csv"),
    },
}

COMPONENT_GRID = np.unique(
    np.rint(np.geomspace(1, 784, 25)).astype(int)
)

OUTPUT_DIR = Path("outputs/full_experiment")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = OUTPUT_DIR / "full_classifier_results.csv"
TEMP_RESULTS_PATH = OUTPUT_DIR / "full_classifier_results.tmp.csv"


# --------------------------------------------------
# Build the experiment matrix
# --------------------------------------------------

def build_tasks(selection_path):

    selection = pd.read_csv(selection_path)

    selected_by_k = {}

    for _, row in selection.iterrows():
        k = int(row["Selected k"])
        method = str(row["Method"])
        selected_by_k.setdefault(k, []).append(method)

    tasks = []

    # Original 784-feature baseline.
    tasks.append({
        "Task_ID": "baseline_784",
        "Representation": "Original baseline",
        "Components": 784,
        "Is_Grid_Point": False,
        "Selection_Methods": "",
    })

    # PCA component grid.
    for k in COMPONENT_GRID:
        methods = selected_by_k.get(int(k), [])

        tasks.append({
            "Task_ID": f"pca_{int(k)}",
            "Representation": "PCA",
            "Components": int(k),
            "Is_Grid_Point": True,
            "Selection_Methods": "; ".join(methods),
        })

    # Rule-selected values not already covered by the grid.
    for k, methods in selected_by_k.items():
        if k not in COMPONENT_GRID:
            tasks.append({
                "Task_ID": f"pca_{k}",
                "Representation": "PCA",
                "Components": k,
                "Is_Grid_Point": False,
                "Selection_Methods": "; ".join(methods),
            })

    return tasks


# --------------------------------------------------
# Validate required input artifacts
# --------------------------------------------------

for arm, config in PREPROCESSING_CONFIGS.items():

    for key in ("data", "pca", "selection"):
        path = config[key]

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing required artifact for {arm}: {path}"
            )

    with np.load(config["data"], allow_pickle=False) as data:
        required = {"X_fit", "X_val", "y_fit", "y_val"}

        if not required.issubset(data.files):
            raise ValueError(
                f"Unexpected data arrays in {config['data']}: "
                f"{data.files}"
            )

    with np.load(config["pca"], allow_pickle=False) as data:
        if "components" not in data.files:
            raise ValueError(
                f"Missing PCA components in {config['pca']}"
            )

        if data["components"].shape != (784, 784):
            raise ValueError(
                f"Unexpected PCA component shape in {config['pca']}"
            )


# --------------------------------------------------
# Load existing checkpoint, if available
# --------------------------------------------------

if RESULTS_PATH.exists():
    results_df = pd.read_csv(RESULTS_PATH)

    if "Task_ID" not in results_df.columns:
        raise ValueError("Existing checkpoint has no Task_ID column.")

    print("Existing checkpoint found.")
    print(f"Previously completed fits: {len(results_df)}")

else:
    results_df = pd.DataFrame()

    print("No existing checkpoint found.")
    print("A new results file will be created.")


completed_ids = (
    set(results_df["Task_ID"].astype(str))
    if not results_df.empty
    else set()
)


# --------------------------------------------------
# Run the full experiment sequentially
# --------------------------------------------------

planned_total = 0

for arm, config in PREPROCESSING_CONFIGS.items():
    planned_total += len(build_tasks(config["selection"]))

if planned_total != 59:
    raise RuntimeError(
        f"Expected 59 unique fits, but experiment matrix contains "
        f"{planned_total}."
    )

print("\nFULL CLASSIFIER EXPERIMENT")
print("=" * 65)
print(f"Planned unique classifier fits: {planned_total}")
print(f"Already completed: {len(completed_ids)}")
print(f"Remaining: {planned_total - len(completed_ids)}")
print("BLAS thread limit: 1")
print("Execution: sequential")
print("Official test set: NOT USED")


for arm, config in PREPROCESSING_CONFIGS.items():

    tasks = build_tasks(config["selection"])

    pending_tasks = [
        task for task in tasks
        if f"{arm}|{task['Task_ID']}" not in completed_ids
    ]

    if not pending_tasks:
        print(f"\n{arm}: all fits already completed.")
        continue

    print("\n" + "=" * 65)
    print(f"PREPROCESSING: {arm}")
    print("=" * 65)

    with np.load(config["data"], allow_pickle=False) as data:
        X_fit = data["X_fit"]
        X_val = data["X_val"]
        y_fit = data["y_fit"]
        y_val = data["y_val"]

    with np.load(config["pca"], allow_pickle=False) as data:
        components = data["components"]

    if X_fit.shape != (50000, 784):
        raise ValueError(
            f"Unexpected fitting shape for {arm}: {X_fit.shape}"
        )

    if X_val.shape != (10000, 784):
        raise ValueError(
            f"Unexpected validation shape for {arm}: {X_val.shape}"
        )

    if not np.isfinite(X_fit).all() or not np.isfinite(X_val).all():
        raise ValueError(f"Non-finite input values found for {arm}.")

    print(f"Fitting data: {X_fit.shape}")
    print(f"Validation data: {X_val.shape}")
    print(f"Pending fits for this arm: {len(pending_tasks)}")

    # Project using the already-fitted full-data PCA directions.
    # No PCA fitting is performed here.
    print("Projecting data using saved PCA components...")

    with threadpool_limits(limits=1):
        X_fit_pca = X_fit @ components.T
        X_val_pca = X_val @ components.T

    print("PCA projection completed.")

    for task in pending_tasks:

        k = task["Components"]

        if task["Representation"] == "Original baseline":
            X_train_k = X_fit
            X_val_k = X_val
        else:
            X_train_k = X_fit_pca[:, :k]
            X_val_k = X_val_pca[:, :k]

        model = LogisticRegression(
            solver="lbfgs",
            max_iter=MAX_ITER,
            random_state=RANDOM_SEED,
        )

        print("\n" + "-" * 65)
        print(
            f"Arm: {arm} | "
            f"Representation: {task['Representation']} | "
            f"k={k}"
        )

        start_time = time.time()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            with threadpool_limits(limits=1):
                model.fit(X_train_k, y_fit)

        training_time = time.time() - start_time

        with threadpool_limits(limits=1):
            predictions = model.predict(X_val_k)

        accuracy = accuracy_score(y_val, predictions)

        convergence_warnings = [
            str(w.message)
            for w in caught
            if "converg" in str(w.message).lower()
        ]

        task_id = f"{arm}|{task['Task_ID']}"

        result = {
            "Task_ID": task_id,
            "Preprocessing": arm,
            "Representation": task["Representation"],
            "Components": int(k),
            "Is_Grid_Point": bool(task["Is_Grid_Point"]),
            "Selection_Methods": task["Selection_Methods"],
            "Validation_Accuracy": float(accuracy),
            "Training_Time_Seconds": float(training_time),
            "Convergence_Warning": bool(convergence_warnings),
            "Warning_Details": " | ".join(convergence_warnings),
            "Iterations": int(np.max(model.n_iter_)),
        }

        results_df = pd.concat(
            [results_df, pd.DataFrame([result])],
            ignore_index=True,
        )

        # Save after every completed classifier fit.
        results_df.to_csv(TEMP_RESULTS_PATH, index=False)
        TEMP_RESULTS_PATH.replace(RESULTS_PATH)

        completed_ids.add(task_id)

        print(
            f"Accuracy={accuracy:.4f} | "
            f"Time={training_time:.2f}s | "
            f"Iterations={result['Iterations']} | "
            f"Convergence warning={result['Convergence_Warning']}"
        )

    del X_fit_pca, X_val_pca
    del X_fit, X_val, y_fit, y_val, components


# --------------------------------------------------
# Final experiment summary
# --------------------------------------------------

results_df = pd.read_csv(RESULTS_PATH)

print("\n" + "=" * 65)
print("FULL EXPERIMENT PROGRESS SUMMARY")
print("=" * 65)

print(f"Completed unique fits: {len(results_df)}")
print(f"Expected unique fits: {planned_total}")

if len(results_df) != planned_total:
    print("Experiment remains incomplete.")
else:
    print("All planned classifier fits are recorded.")

for arm in PREPROCESSING_CONFIGS:

    arm_results = results_df[
        results_df["Preprocessing"] == arm
    ]

    grid_results = arm_results[
        arm_results["Is_Grid_Point"] == True
    ]

    if not grid_results.empty:
        best_row = grid_results.loc[
            grid_results["Validation_Accuracy"].idxmax()
        ]

        print(f"\n{arm}")
        print(f"Best grid k: {best_row['Components']}")
        print(
            "Best grid validation accuracy: "
            f"{best_row['Validation_Accuracy']:.4f}"
        )

print(
    "\nTotal convergence warnings: "
    f"{results_df['Convergence_Warning'].sum()}"
)

print(f"Results saved: {RESULTS_PATH}")
print("\nFULL CLASSIFIER EXPERIMENT RUNNER FINISHED")
