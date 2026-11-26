import numpy as np
import pandas as pd
from pathlib import Path

# Load selected component counts
variance = np.load("outputs/variance_selection_results.npz")
kaiser = np.load("outputs/kaiser_selection_results.npz")
scree = np.load("outputs/scree_elbow_results.npz")
parallel = np.load("outputs/parallel_analysis_results.npz")
minka = np.load("outputs/minka_mle_results.npz")

# Consolidate results
results = [
    {"Method": "90% Explained Variance", "Selected k": int(variance["k_90"])},
    {"Method": "95% Explained Variance", "Selected k": int(variance["k_95"])},
    {"Method": "Kaiser Criterion", "Selected k": int(kaiser["k_kaiser"])},
    {"Method": "Scree Elbow", "Selected k": int(scree["k_elbow"])},
    {"Method": "Horn's Parallel Analysis", "Selected k": int(parallel["k_parallel"])},
    {"Method": "Minka's MLE", "Selected k": int(minka["k_minka"])},
]

df = pd.DataFrame(results)

# Save research table
output_dir = Path("outputs/tables")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "pca_dimension_selection_comparison.csv"
df.to_csv(output_file, index=False)

# Display results
print("PCA DIMENSION-SELECTION COMPARISON")
print("-" * 50)
print(df.to_string(index=False))

print("\nTotal methods:", len(df))
print("Minimum selected k:", df["Selected k"].min())
print("Maximum selected k:", df["Selected k"].max())

print("\nSaved research table:", output_file)
print("\nRESULT CONSOLIDATION COMPLETED")
