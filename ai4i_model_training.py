"""
ai4i_model_training.py

Train and compare classifiers using the existing AI4I 2020 prepared split.
This version explicitly validates the target distribution before training.

If y_test contains only one class, the script stops with a clear explanation.
It does not silently switch to a random split. To fix the split, regenerate the
prepared split in the feature-engineering/preparation step using a justified
stratified split (if the rows are independent) or a chronological split that
contains failure examples in the test period (if time ordering matters).

Required files:
  X_train.csv, X_test.csv, y_train.csv, y_test.csv

Optional context file:
  ai4i_engineered.csv

Outputs:
  ai4i_model_comparison.csv
  ai4i_best_model.joblib
  ai4i_model_plots/
"""

from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
)

BASE_DIR = Path(__file__).resolve().parent

# AI4I-specific dataset files
X_TRAIN_FILE = BASE_DIR / "ai4i_X_train.csv"
X_TEST_FILE = BASE_DIR / "ai4i_X_test.csv"
Y_TRAIN_FILE = BASE_DIR / "ai4i_y_train.csv"
Y_TEST_FILE = BASE_DIR / "ai4i_y_test.csv"

ENGINEERED_FILE = BASE_DIR / "ai4i_engineered.csv"

# Model outputs
COMPARISON_FILE = BASE_DIR / "ai4i_model_comparison.csv"
BEST_MODEL_FILE = BASE_DIR / "ai4i_best_model.joblib"
PLOT_DIR = BASE_DIR / "ai4i_model_plots"

RANDOM_STATE = 42
TARGET_NAME = "Machine failure"

FORBIDDEN_FEATURES = {
    "UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF",
    TARGET_NAME, "Failure_Status", "failure", "target",
}



def stop(message):
    print(f"\nERROR: {message}")
    sys.exit(1)


def read_csv(path):
    if not path.exists():
        stop(
            f"Required file not found: {path.name}. Place the prepared CSV files "
            "in the same folder as this script."
        )
    try:
        return pd.read_csv(path)
    except Exception as exc:
        stop(f"Could not read {path.name}: {exc}")


def read_target(path):
    target_df = read_csv(path)
    target_df = target_df.loc[
        :, [c for c in target_df.columns if not str(c).lower().startswith("unnamed:")]
    ]
    if target_df.shape[1] != 1:
        stop(
            f"{path.name} must contain exactly one target column after ignoring "
            f"saved index columns. Found: {list(target_df.columns)}"
        )
    target = target_df.iloc[:, 0].copy()
    target.name = TARGET_NAME
    return target


def validate_target_values(y, label):
    if y.isna().any():
        stop(f"{label} has {int(y.isna().sum())} missing target values.")
    numeric = pd.to_numeric(y, errors="coerce")
    if numeric.isna().any():
        stop(f"{label} contains non-numeric target values.")
    values = set(numeric.unique())
    if not values.issubset({0, 1}) or not values:
        stop(f"{label} must contain only binary values 0 and 1. Found: {sorted(values)}")
    return numeric.astype(int)


def prepare_features(X, label):
    X = X.copy()
    index_cols = [c for c in X.columns if str(c).lower().startswith("unnamed:")]
    if index_cols:
        print(f"{label}: ignoring saved index columns: {index_cols}")
        X = X.drop(columns=index_cols)

    forbidden = [c for c in X.columns if c in FORBIDDEN_FEATURES]
    if forbidden:
        print(f"{label}: excluding forbidden/target-leakage columns: {forbidden}")
        X = X.drop(columns=forbidden)

    if X.empty:
        stop(f"{label} has no usable features after exclusions.")

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def show_distribution(y, label):
    counts = y.value_counts().reindex([0, 1], fill_value=0)
    print(f"{label}:")
    print(f"  Class 0 - no machine failure: {int(counts.loc[0]):,}")
    print(f"  Class 1 - machine failure:    {int(counts.loc[1]):,}")


def main():
    print("=" * 76)
    print("AI4I 2020 - MODEL TRAINING AND EVALUATION")
    print("=" * 76)

    X_train = prepare_features(read_csv(X_TRAIN_FILE), "X_train")
    X_test = prepare_features(read_csv(X_TEST_FILE), "X_test")
    y_train = validate_target_values(read_target(Y_TRAIN_FILE), "y_train")
    y_test = validate_target_values(read_target(Y_TEST_FILE), "y_test")

    print("\nLoaded shapes:")
    print(f"X_train: {X_train.shape}; y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}; y_test:  {y_test.shape}")

    if len(X_train) != len(y_train):
        stop(f"X_train has {len(X_train)} rows but y_train has {len(y_train)}.")
    if len(X_test) != len(y_test):
        stop(f"X_test has {len(X_test)} rows but y_test has {len(y_test)}.")

    train_features = list(X_train.columns)
    test_features = list(X_test.columns)
    if set(train_features) != set(test_features):
        stop(
            "X_train and X_test feature columns do not match.\n"
            f"Missing from X_test: {[c for c in train_features if c not in test_features]}\n"
            f"Extra in X_test: {[c for c in test_features if c not in train_features]}"
        )
    X_test = X_test[train_features]

    print("\nFeatures used:")
    for feature in train_features:
        print(f"  - {feature}")
    print(f"Feature count: {len(train_features)}")

    print("\nMissing values in X_train:")
    missing_train = X_train.isna().sum()
    print(missing_train[missing_train > 0].to_string() if missing_train.any() else "None")
    print("\nMissing values in X_test:")
    missing_test = X_test.isna().sum()
    print(missing_test[missing_test > 0].to_string() if missing_test.any() else "None")

    entirely_missing = [c for c in X_train.columns if X_train[c].isna().all()]
    if entirely_missing:
        stop(f"Features entirely missing in training: {entirely_missing}")

    print("\nTarget distribution:")
    show_distribution(y_train, "Training")
    show_distribution(y_test, "Testing")

    # Require both classes in each partition for this model-comparison task.
    if set(y_train.unique()) != {0, 1}:
        stop(
            "The training target does not contain both classes. Repair the "
            "prepared split before fitting classifiers."
        )
    if set(y_test.unique()) != {0, 1}:
        stop(
            "The test target contains only one class. Failure-detection "
            "precision, recall and F1 cannot be meaningfully evaluated against "
            "a test set with no positive failure examples. No model was trained. "
            "Recreate the prepared split with a defensible strategy, then rerun. "
            "Do not silently replace it with a random split."
        )

    if ENGINEERED_FILE.exists():
        try:
            preview = pd.read_csv(ENGINEERED_FILE, nrows=5)
            print(f"\nOptional engineered dataset found. Columns: {list(preview.columns)}")
        except Exception as exc:
            print(f"\nWarning: could not inspect {ENGINEERED_FILE.name}: {exc}")
    else:
        print(f"\nOptional file {ENGINEERED_FILE.name} not found; using prepared splits only.")

    still_forbidden = [c for c in train_features if c in FORBIDDEN_FEATURES]
    if still_forbidden:
        stop(f"Forbidden columns remain in model inputs: {still_forbidden}")

    # Preprocessing is fitted on training data only, inside each Pipeline.
    numeric_features = train_features
    tree_preprocessor = ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median"), numeric_features)],
        remainder="drop",
    )
    scaled_preprocessor = ColumnTransformer(
        [("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_features)],
        remainder="drop",
    )

    models = {
        "Random Forest": Pipeline([
            ("preprocessing", tree_preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=300, random_state=RANDOM_STATE,
                class_weight="balanced", n_jobs=-1,
            )),
        ]),
        "Decision Tree": Pipeline([
            ("preprocessing", tree_preprocessor),
            ("classifier", DecisionTreeClassifier(
                random_state=RANDOM_STATE, class_weight="balanced",
            )),
        ]),
        "Logistic Regression": Pipeline([
            ("preprocessing", scaled_preprocessor),
            ("classifier", LogisticRegression(
                max_iter=2000, random_state=RANDOM_STATE,
                class_weight="balanced",
            )),
        ]),
    }

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    rows, fitted, reports = [], {}, {}

    print("\n" + "=" * 76)
    print("TRAINING AND EVALUATION")
    print("=" * 76)

    for name, pipeline in models.items():
        print(f"\n--- {name} ---")
        try:
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
        except Exception as exc:
            stop(f"{name} failed during training or prediction: {exc}")

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, pos_label=1, zero_division=0)
        recall = recall_score(y_test, predictions, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, predictions, pos_label=1, zero_division=0)
        cm = confusion_matrix(y_test, predictions, labels=[0, 1])
        report_dict = classification_report(
            y_test, predictions, labels=[0, 1],
            target_names=["No machine failure (0)", "Machine failure (1)"],
            output_dict=True, zero_division=0,
        )
        report_text = classification_report(
            y_test, predictions, labels=[0, 1],
            target_names=["No machine failure (0)", "Machine failure (1)"],
            zero_division=0,
        )

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Failure precision: {precision:.4f}")
        print(f"Failure recall:    {recall:.4f}")
        print(f"Failure F1-score:  {f1:.4f}")
        print("Confusion matrix (actual rows, predicted columns; order [0, 1]):")
        print(cm)
        print("Classification report:")
        print(report_text)

        rows.append({
            "Model": name, "Accuracy": accuracy,
            "Failure_Precision": precision, "Failure_Recall": recall,
            "Failure_F1": f1, "Test_Rows": len(y_test),
            "Test_Failure_Count": int((y_test == 1).sum()),
            "Test_NonFailure_Count": int((y_test == 0).sum()),
            "True_Negatives": int(cm[0, 0]), "False_Positives": int(cm[0, 1]),
            "False_Negatives": int(cm[1, 0]), "True_Positives": int(cm[1, 1]),
        })
        fitted[name] = pipeline
        reports[name] = (cm, report_text)

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No failure (0)", "Failure (1)"],
        )
        display.plot(values_format="d")
        plt.title(f"{name} - Confusion Matrix")
        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_")
        plt.savefig(PLOT_DIR / f"{safe_name}_confusion_matrix.png", dpi=150)
        plt.close()

        pd.DataFrame(report_dict).transpose().to_csv(
            PLOT_DIR / f"{safe_name}_classification_report.csv"
        )

    comparison = pd.DataFrame(rows).sort_values(
        ["Failure_F1", "Failure_Recall", "Failure_Precision"],
        ascending=False, kind="stable",
    ).reset_index(drop=True)
    try:
        comparison.to_csv(COMPARISON_FILE, index=False)
    except Exception as exc:
        stop(f"Could not save comparison CSV: {exc}")

    best_name = str(comparison.iloc[0]["Model"])
    best_model = fitted[best_name]

    metrics = [
        ("Accuracy", "Accuracy"),
        ("Failure_Precision", "Failure precision"),
        ("Failure_Recall", "Failure recall"),
        ("Failure_F1", "Failure F1-score"),
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(comparison))
    width = 0.18
    for i, (column, label) in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, comparison[column], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison["Model"], rotation=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("AI4I Model Metrics Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "model_metrics_comparison.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 6))
    for _, row in comparison.iterrows():
        ax.scatter(row["Failure_Recall"], row["Failure_Precision"], s=90)
        ax.annotate(row["Model"], (row["Failure_Recall"], row["Failure_Precision"]),
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Failure recall")
    ax.set_ylabel("Failure precision")
    ax.set_title("Failure Detection: Precision vs Recall")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "failure_precision_recall_comparison.png", dpi=150)
    plt.close()

    try:
        joblib.dump(best_model, BEST_MODEL_FILE)
    except Exception as exc:
        stop(f"Could not save best model: {exc}")

    best_cm, best_report_text = reports[best_name]
    (PLOT_DIR / "best_model_classification_report.txt").write_text(
        f"Best model: {best_name}\n"
        "Selection: highest failure F1-score, then recall, then precision.\n\n"
        f"Confusion matrix (actual rows, predicted columns; order [0, 1]):\n"
        f"{best_cm}\n\n{best_report_text}\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 76)
    print("MODEL COMPARISON")
    print("=" * 76)
    print(comparison[[
        "Model", "Accuracy", "Failure_Precision", "Failure_Recall",
        "Failure_F1", "False_Negatives", "False_Positives",
    ]].to_string(index=False, formatters={
        "Accuracy": "{:.4f}".format,
        "Failure_Precision": "{:.4f}".format,
        "Failure_Recall": "{:.4f}".format,
        "Failure_F1": "{:.4f}".format,
    }))
    print(f"\nBest model by failure-class F1: {best_name}")
    print(
        "\nAccuracy alone can mislead because machine failures are rare. Failure "
        "precision, recall, F1-score and false negatives are essential to see "
        "how well failures are detected."
    )
    print(
        "\nLIMITATION: This evaluates classification of the supplied Machine "
        "failure labels. It does not by itself prove prediction before a failure. "
        "The evaluation also depends on the validity of the prepared split."
    )
    print("\nSaved outputs:")
    print(f"  Comparison CSV: {COMPARISON_FILE}")
    print(f"  Best model pipeline: {BEST_MODEL_FILE}")
    print(f"  Plots and reports: {PLOT_DIR}")
    print("Original and engineered datasets were not modified.")
    print("Done.")


if __name__ == "__main__":
    main()
