
# prepare_ai4i_split.py

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "ai4i_engineered.csv"

df = pd.read_csv(INPUT_FILE)

if "Machine failure" not in df.columns:
    raise ValueError(
        "Machine failure column is missing from ai4i_engineered.csv"
    )

if len(df) != 10000:
    raise ValueError(
        f"Expected 10,000 AI4I rows, but found {len(df)}. "
        "Check that the correct dataset was loaded."
    )

# Remove identifiers and columns that reveal the failure type.
excluded = [
    "UDI", "Product ID", "TWF", "HDF",
    "PWF", "OSF", "RNF", "Machine failure"
]

X = df.drop(columns=excluded, errors="ignore")
y = df["Machine failure"]

# Use only numeric sensor and engineered features.
X = X.select_dtypes(include="number")

if y.nunique() != 2:
    raise ValueError("Machine failure must contain classes 0 and 1.")

# Stratification preserves the failure/non-failure proportions.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Save separate files for AI4I only.
X_train.to_csv(BASE_DIR / "ai4i_X_train.csv", index=False)
X_test.to_csv(BASE_DIR / "ai4i_X_test.csv", index=False)
y_train.to_frame().to_csv(
    BASE_DIR / "ai4i_y_train.csv", index=False
)
y_test.to_frame().to_csv(
    BASE_DIR / "ai4i_y_test.csv", index=False
)

print("AI4I split completed successfully.")
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting target distribution:")
print(y_test.value_counts().sort_index())
