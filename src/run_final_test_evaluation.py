from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from threadpoolctl import threadpool_limits


SEED = 42
MAX_ITER = 5000
ARMS = {
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

SPLIT_PATH = Path("data/processed/fashion_mnist_split.npz")
VALIDATION_RESULTS = Path(
    "outputs/full_experiment/full_classifier_results.csv"
)

OUT = Path("outputs/test_evaluation")
PRED_DIR = OUT / "predictions"
CM_DIR = OUT / "confusion_matrices"
RESULTS_PATH = OUT / "test_metrics.csv"


def build_tasks():
    validation = pd.read_csv(VALIDATION_RESULTS)

    if len(validation) != 59 or validation["Task_ID"].nunique() != 59:
        raise ValueError("Validation experiment must contain 59 unique fits.")

    tasks = []

    for arm, config in ARMS.items():
        subset = validation[validation["Preprocessing"] == arm]
        grid = subset[
            subset["Is_Grid_Point"].astype(str).str.lower() == "true"
        ]

        if grid.empty:
            raise ValueError(f"No validation grid results for {arm}.")

        best = grid.loc[grid["Validation_Accuracy"].idxmax()]

        tasks.append({
            "Task_ID": f"{arm}|baseline_784",
            "Preprocessing": arm,
            "Representation": "Original baseline",
            "Components": 784,
            "Selection_Method": "Original baseline",
        })

        tasks.append({
            "Task_ID": f"{arm}|best_validation_grid_k{int(best['Components'])}",
            "Preprocessing": arm,
            "Representation": "PCA",
            "Components": int(best["Components"]),
            "Selection_Method": "Best validation grid",
        })

        selection = pd.read_csv(config["selection"])

        for _, row in selection.iterrows():
            k = int(row["Selected k"])
            method = str(row["Method"])

            tasks.append({
                "Task_ID": f"{arm}|rule_{method.lower().replace(' ', '_').replace('/', '_')}",
                "Preprocessing": arm,
                "Representation": "PCA",
                "Components": k,
                "Selection_Method": method,
            })

    ids = [task["Task_ID"] for task in tasks]
    if len(tasks) != 16 or len(set(ids)) != 16:
        raise ValueError("Expected exactly 16 unique evaluation configurations.")

    return tasks


def atomic_csv(frame, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def check_only(tasks):
    with np.load(SPLIT_PATH, allow_pickle=False) as split:
        X_fit_raw = split["X_fit"].astype(np.float64) / 255.0
        y_fit = split["y_fit"]

    mean = X_fit_raw.mean(axis=0)
    std = X_fit_raw.std(axis=0, ddof=0)
    safe_std = np.where(std == 0, 1.0, std)

    centered = X_fit_raw - mean
    standardized = centered / safe_std

    for arm, config in ARMS.items():
        with np.load(config["data"], allow_pickle=False) as saved:
            expected = centered if arm == "Scaled and Centered" else standardized

            if not np.allclose(saved["X_fit"], expected, rtol=1e-12, atol=1e-12):
                raise ValueError(f"Fitting preprocessing mismatch: {arm}")

            if not np.array_equal(saved["y_fit"], y_fit):
                raise ValueError(f"Fitting labels mismatch: {arm}")

        with np.load(config["pca"], allow_pickle=False) as pca:
            components = pca["components"]

            if components.shape != (784, 784):
                raise ValueError(f"Invalid PCA shape: {arm}")

            if not np.isfinite(components).all():
                raise ValueError(f"Non-finite PCA components: {arm}")

        print(f"{arm}: fitting preprocessing and PCA artifacts PASS")

    for task in tasks:
        print(
            f"{task['Preprocessing']} | "
            f"{task['Representation']} | "
            f"k={task['Components']} | "
            f"{task['Selection_Method']}"
        )

    print("\nCHECK-ONLY RESULT: PASS")
    print("Configurations:", len(tasks))
    print("Official test images or labels loaded: NO")


def run_evaluation(tasks):
    from gzip import open as gzip_open
    sys.path.insert(0, str(Path("src").resolve()))
    from load_fashion_mnist import load_images, load_labels

    with np.load(SPLIT_PATH, allow_pickle=False) as split:
        X_fit_raw = split["X_fit"].astype(np.float64) / 255.0
        y_fit = split["y_fit"]

    mean = X_fit_raw.mean(axis=0)
    std = X_fit_raw.std(axis=0, ddof=0)
    safe_std = np.where(std == 0, 1.0, std)

    # Verify saved fitting transformations before accessing test data.
    centered_fit = X_fit_raw - mean
    standardized_fit = centered_fit / safe_std

    for arm, config in ARMS.items():
        expected = centered_fit if arm == "Scaled and Centered" else standardized_fit
        with np.load(config["data"], allow_pickle=False) as saved:
            if not np.allclose(saved["X_fit"], expected, rtol=1e-12, atol=1e-12):
                raise ValueError(f"Preprocessing verification failed: {arm}")

    # Official test data is first loaded here, after fitting-only checks.
    data_dir = Path("data/raw/fashion")
    X_test_images = load_images(data_dir / "t10k-images-idx3-ubyte.gz")
    y_test = load_labels(data_dir / "t10k-labels-idx1-ubyte.gz")

    if X_test_images.shape != (10000, 28, 28):
        raise ValueError(f"Unexpected test image shape: {X_test_images.shape}")

    if y_test.shape != (10000,) or not np.array_equal(np.unique(y_test), np.arange(10)):
        raise ValueError("Unexpected test labels.")

    X_test_raw = X_test_images.reshape(10000, 784).astype(np.float64) / 255.0

    X_by_arm = {
        "Scaled and Centered": centered_fit,
        "Standardised": standardized_fit,
    }
    X_test_by_arm = {
        "Scaled and Centered": X_test_raw - mean,
        "Standardised": (X_test_raw - mean) / safe_std,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    CM_DIR.mkdir(parents=True, exist_ok=True)

    if RESULTS_PATH.exists():
        results = pd.read_csv(RESULTS_PATH)
        if "Task_ID" not in results.columns:
            raise ValueError(
                "Existing test_metrics.csv has an invalid structure. "
                "Inspect it before continuing."
            )
    else:
        results = pd.DataFrame(columns=[
            "Task_ID",
            "Preprocessing",
            "Representation",
            "Components",
            "Selection_Method",
            "Test_Accuracy",
            "Training_Time_Seconds",
            "Convergence_Warning",
            "Warning_Details",
            "Iterations",
        ])

    for task in tasks:
        task_id = task["Task_ID"]
        safe_name = str(tasks.index(task)).zfill(2)
        prediction_path = PRED_DIR / f"{safe_name}_predictions.csv"
        confusion_path = CM_DIR / f"{safe_name}_confusion_matrix.csv"

        already_saved = (
            not results.empty
            and task_id in set(results["Task_ID"])
            and prediction_path.exists()
            and confusion_path.exists()
        )
        if already_saved:
            print(f"Already evaluated; skipping: {task_id}")
            continue

        results = results[results["Task_ID"] != task_id]

        arm = task["Preprocessing"]
        config = ARMS[arm]

        X_fit = X_by_arm[arm]
        X_test = X_test_by_arm[arm]

        if task["Representation"] == "Original baseline":
            X_train_model = X_fit
            X_test_model = X_test
        else:
            with np.load(config["pca"], allow_pickle=False) as pca:
                components = pca["components"]

            with threadpool_limits(limits=1):
                X_train_model = (X_fit @ components.T)[:, :task["Components"]]
                X_test_model = (X_test @ components.T)[:, :task["Components"]]

        model = LogisticRegression(
            solver="lbfgs",
            max_iter=MAX_ITER,
            random_state=SEED,
        )

        print(f"\nTraining and evaluating: {task_id}")

        start = time.time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with threadpool_limits(limits=1):
                model.fit(X_train_model, y_fit)
        training_time = time.time() - start

        with threadpool_limits(limits=1):
            predictions = model.predict(X_test_model)

        accuracy = accuracy_score(y_test, predictions)
        convergence_warnings = [
            str(w.message) for w in caught
            if "converg" in str(w.message).lower()
        ]

        prediction_frame = pd.DataFrame({
            "Test_Index": np.arange(len(y_test)),
            "True_Label": y_test,
            "Predicted_Label": predictions,
        })
        atomic_csv(prediction_frame, prediction_path)

        matrix = confusion_matrix(y_test, predictions, labels=np.arange(10))
        matrix_frame = pd.DataFrame(
            matrix,
            index=[f"True_{i}" for i in range(10)],
            columns=[f"Predicted_{i}" for i in range(10)],
        )
        matrix_frame.index.name = "Class"
        atomic_csv(matrix_frame.reset_index(), confusion_path)

        result = {
            "Task_ID": task_id,
            "Preprocessing": arm,
            "Representation": task["Representation"],
            "Components": task["Components"],
            "Selection_Method": task["Selection_Method"],
            "Test_Accuracy": float(accuracy),
            "Training_Time_Seconds": float(training_time),
            "Convergence_Warning": bool(convergence_warnings),
            "Warning_Details": " | ".join(convergence_warnings),
            "Iterations": int(np.max(model.n_iter_)),
        }

        results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)
        atomic_csv(results, RESULTS_PATH)

        print(
            f"Test accuracy={accuracy:.4f} | "
            f"Iterations={result['Iterations']} | "
            f"Convergence warning={result['Convergence_Warning']}"
        )

    print("\nFINAL TEST EVALUATION SUMMARY")
    print("Completed configurations:", len(results))
    print("Expected configurations:", len(tasks))
    print("Results:", RESULTS_PATH)
    print("Predictions:", PRED_DIR)
    print("Confusion matrices:", CM_DIR)


if __name__ == "__main__":
    tasks = build_tasks()

    if "--check-only" in sys.argv:
        check_only(tasks)
    else:
        run_evaluation(tasks)
