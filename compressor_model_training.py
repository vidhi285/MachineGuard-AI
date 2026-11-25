"""
MetroPT-3 Air Compressor: Event-Based Model Training and Evaluation

Input: compressor_engineered.csv
Outputs:
  compressor_model_comparison.csv
  compressor_model_plots/
  compressor_best_model.joblib (only when a valid event-based split is available)

Failure_Status=1 means inside a documented failure interval.
Failure_Status=0 means outside documented intervals, not necessarily healthy.
This is a limited case study, not proof of early-warning prediction.
"""

from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "compressor_engineered.csv"
COMPARISON_FILE = BASE_DIR / "compressor_model_comparison.csv"
BEST_MODEL_FILE = BASE_DIR / "compressor_best_model.joblib"
PLOT_DIR = BASE_DIR / "compressor_model_plots"

TIMESTAMP = "timestamp"
TARGET = "Failure_Status"
RANDOM_STATE = 42

# Hold out the latest event and 24 hours on either side.
HOURS_BEFORE_EVENT = 24
HOURS_AFTER_EVENT = 24

# Keep a one-hour embargo before the test window.
EMBARGO_HOURS = 1

# To keep training practical on a 1.5-million-row dataset, retain every
# positive training row and at most 10 negative rows per positive.
MAX_NEGATIVE_TO_POSITIVE_RATIO = 10


def stop(message):
    raise SystemExit(f"\nEVALUATION STOPPED: {message}")


def print_distribution(label, y):
    counts = y.value_counts().reindex([0, 1], fill_value=0)
    print(f"{label}:")
    print(f"  Class 0 (outside documented intervals): {counts.loc[0]:,}")
    print(f"  Class 1 (inside documented intervals):  {counts.loc[1]:,}")


def find_events(frame):
    """Find contiguous positive-label runs in chronological order."""
    positive = frame[TARGET].eq(1).to_numpy()
    times = frame[TIMESTAMP].to_numpy()
    events = []
    start = None

    for i, flag in enumerate(positive):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            end = i - 1
            events.append({
                "start": pd.Timestamp(times[start]),
                "end": pd.Timestamp(times[end]),
                "rows": end - start + 1,
            })
            start = None

    if start is not None:
        end = len(frame) - 1
        events.append({
            "start": pd.Timestamp(times[start]),
            "end": pd.Timestamp(times[end]),
            "rows": end - start + 1,
        })
    return events


def save_confusion_matrix(cm, model_name, output_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Outside interval (0)", "Inside interval (1)"],
    )
    display.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"{model_name}: held-out event confusion matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    print("=" * 78)
    print("METROPT-3 AIR COMPRESSOR - EVENT-BASED MODEL TRAINING")
    print("=" * 78)

    if not DATA_FILE.exists():
        stop(f"Cannot find {DATA_FILE.name} in {BASE_DIR}. Put it beside this script.")

    try:
        data = pd.read_csv(DATA_FILE)
    except Exception as exc:
        stop(f"Could not read the CSV: {exc}")

    missing_required = {TIMESTAMP, TARGET} - set(data.columns)
    if missing_required:
        stop(f"Required column(s) missing: {sorted(missing_required)}")
    if data.empty:
        stop("The input CSV has no rows.")

    data[TIMESTAMP] = pd.to_datetime(data[TIMESTAMP], errors="coerce")
    invalid_times = int(data[TIMESTAMP].isna().sum())
    if invalid_times:
        stop(f"{invalid_times:,} timestamps are missing or invalid. Resolve them first.")
    if data[TARGET].isna().any():
        stop(f"{TARGET} contains missing values.")

    target_numeric = pd.to_numeric(data[TARGET], errors="coerce")
    if target_numeric.isna().any() or not set(target_numeric.unique()).issubset({0, 1}):
        stop(f"{TARGET} must contain only binary values 0 and 1.")
    if set(target_numeric.unique()) != {0, 1}:
        stop(f"{TARGET} must contain both classes 0 and 1.")

    data[TARGET] = target_numeric.astype(int)
    data = data.sort_values(TIMESTAMP, kind="mergesort").reset_index(drop=True)

    if data[TIMESTAMP].duplicated().any():
        count = int(data[TIMESTAMP].duplicated().sum())
        stop(f"Found {count:,} duplicate timestamps; resolve them before evaluation.")

    print(f"\nRows loaded: {len(data):,}")
    print(f"Columns loaded: {len(data.columns)}")
    print(f"Date range: {data[TIMESTAMP].min()} to {data[TIMESTAMP].max()}")
    print(f"Missing cells across all columns: {int(data.isna().sum().sum()):,}")
    missing = data.isna().sum()
    missing = missing[missing > 0]
    print("\nMissing values by column:")
    print(missing.to_string() if not missing.empty else "  None")
    print("\nTarget distribution:")
    print_distribution("All records", data[TARGET])

    events = find_events(data)
    print(f"\nDetected contiguous failure events: {len(events)}")
    for i, event in enumerate(events, start=1):
        print(f"  Event {i}: {event['start']} to {event['end']} ({event['rows']:,} rows)")

    if len(events) < 2:
        stop("At least two events are needed for an event-based holdout.")
    if len(events) != 4:
        warnings.warn(
            f"Expected four events but detected {len(events)}. Check the label intervals."
        )

    # Hold out the latest event and surrounding time. Only records strictly
    # earlier than the embargo cutoff can enter training.
    held_out = events[-1]
    test_start = held_out["start"] - pd.Timedelta(hours=HOURS_BEFORE_EVENT)
    test_end = held_out["end"] + pd.Timedelta(hours=HOURS_AFTER_EVENT)
    training_cutoff = test_start - pd.Timedelta(hours=EMBARGO_HOURS)

    train = data.loc[data[TIMESTAMP] < training_cutoff].copy()
    test = data.loc[data[TIMESTAMP].between(test_start, test_end, inclusive="both")].copy()

    print("\n" + "=" * 78)
    print("EVENT-BASED CHRONOLOGICAL SPLIT")
    print("=" * 78)
    print(f"Held-out event: Event {len(events)}")
    print(f"Event interval: {held_out['start']} to {held_out['end']}")
    print(f"Rows in held-out event: {held_out['rows']:,}")
    print(f"Test window: {test_start} to {test_end}")
    print(f"Training cutoff: records earlier than {training_cutoff}")
    print(f"Embargo before test window: {EMBARGO_HOURS} hour(s)")
    print(f"Training rows: {len(train):,}")
    print(f"Testing rows: {len(test):,}")
    print()
    print_distribution("Training target", train[TARGET])
    print_distribution("Testing target", test[TARGET])

    if train.empty or test.empty:
        stop("Training or test set is empty.")
    if train[TARGET].nunique() != 2:
        stop("Earlier training data does not contain both classes; no random split used.")
    if test[TARGET].nunique() != 2:
        stop(
            "Held-out event window lacks one of the classes. Do not report failure metrics "
            "or save a model; review event labels/window definition."
        )
    if train[TIMESTAMP].max() >= test[TIMESTAMP].min():
        stop("Chronological separation check failed.")

    print(f"\nActual training dates: {train[TIMESTAMP].min()} to {train[TIMESTAMP].max()}")
    print(f"Actual testing dates:  {test[TIMESTAMP].min()} to {test[TIMESTAMP].max()}")

    # Exclude timestamp, label, index/ID columns, and other label-like columns.
    exact_exclusions = {
        TIMESTAMP, TARGET, "Unnamed: 0", "UDI", "Product ID", "id", "ID",
        "failure", "target", "label",
    }
    feature_columns, excluded = [], []
    for col in data.columns:
        lower = str(col).strip().lower()
        if col in exact_exclusions or "timestamp" in lower or "target" in lower or "label" in lower or lower.endswith("_id"):
            excluded.append(col)
        elif pd.api.types.is_numeric_dtype(data[col]):
            feature_columns.append(col)
        else:
            excluded.append(col)

    if not feature_columns:
        stop("No numeric sensor/engineered features found.")

    X_train = train[feature_columns].replace([np.inf, -np.inf], np.nan)
    X_test = test[feature_columns].replace([np.inf, -np.inf], np.nan)
    y_train = train[TARGET].copy()
    y_test = test[TARGET].copy()

    # Drop features that are entirely missing in training: a training-only
    # imputation value cannot be learned for those columns.
    all_missing = [col for col in X_train if X_train[col].isna().all()]
    if all_missing:
        print(f"\nDropping all-missing training feature(s): {all_missing}")
        X_train = X_train.drop(columns=all_missing)
        X_test = X_test.drop(columns=all_missing)
        feature_columns = list(X_train.columns)
    if X_train.shape[1] == 0:
        stop("No usable features remain.")

    print("\nFeatures used:")
    for col in feature_columns:
        print(f"  - {col}")
    print(f"Feature count: {len(feature_columns)}")
    print(f"Excluded columns: {sorted(set(excluded))}")

    # Training-only sampling: retain all positives and sample negatives to
    # at most 10:1. Test data is not sampled. Imputation is fitted in pipelines.
    positive = X_train.loc[y_train.eq(1)].copy()
    negative = X_train.loc[y_train.eq(0)].copy()
    if positive.empty or negative.empty:
        stop("Training must contain both classes before sampling.")

    negative_n = min(len(negative), MAX_NEGATIVE_TO_POSITIVE_RATIO * len(positive))
    negative_sample = negative.sample(n=negative_n, random_state=RANDOM_STATE)
    X_fit = pd.concat([positive, negative_sample], axis=0)
    y_fit = pd.concat([
        pd.Series(1, index=positive.index, name=TARGET),
        pd.Series(0, index=negative_sample.index, name=TARGET),
    ])
    order = np.random.RandomState(RANDOM_STATE).permutation(len(X_fit))
    X_fit = X_fit.iloc[order]
    y_fit = y_fit.iloc[order]

    print("\nTraining-only negative sampling:")
    print(f"  Training rows before sampling: {len(X_train):,}")
    print(f"  Rows used for fitting: {len(X_fit):,}")
    print(f"  Failure rows retained: {int(y_fit.sum()):,}")
    print(f"  Non-failure rows retained: {int((y_fit == 0).sum()):,}")
    print("  Test data remains unchanged.")

    models = {
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=150, random_state=RANDOM_STATE,
                class_weight="balanced", n_jobs=-1, min_samples_leaf=2,
            )),
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(
                random_state=RANDOM_STATE, class_weight="balanced",
                min_samples_leaf=10,
            )),
        ]),
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                random_state=RANDOM_STATE, class_weight="balanced", max_iter=1000,
            )),
        ]),
    }

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    rows, fitted = [], {}
    print("\n" + "=" * 78)
    print("TRAINING AND EVALUATION")
    print("=" * 78)

    for name, model in models.items():
        print(f"\n--- {name} ---")
        try:
            model.fit(X_fit, y_fit)
            pred = model.predict(X_test)
        except Exception as exc:
            warnings.warn(f"{name} failed: {exc}")
            continue

        acc = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, pos_label=1, zero_division=0)
        recall = recall_score(y_test, pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, pred, pos_label=1, zero_division=0)
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        print(f"Accuracy:          {acc:.4f}")
        print(f"Failure precision: {precision:.4f}")
        print(f"Failure recall:    {recall:.4f}")
        print(f"Failure F1-score:  {f1:.4f}")
        print("Confusion matrix (actual rows, predicted columns; order [0, 1]):")
        print(cm)
        print("Classification report:")
        print(classification_report(
            y_test, pred, labels=[0, 1],
            target_names=["Outside documented interval (0)", "Inside documented interval (1)"],
            zero_division=0,
        ))

        safe_name = name.lower().replace(" ", "_")
        save_confusion_matrix(cm, name, PLOT_DIR / f"{safe_name}_confusion_matrix.png")
        rows.append({
            "Model": name, "Accuracy": acc, "Failure_Precision": precision,
            "Failure_Recall": recall, "Failure_F1": f1,
            "True_Negatives": int(tn), "False_Positives": int(fp),
            "False_Negatives": int(fn), "True_Positives": int(tp),
            "Held_Out_Event": len(events), "Training_Rows_Used": len(X_fit),
            "Test_Rows": len(X_test), "Test_Start": str(test[TIMESTAMP].min()),
            "Test_End": str(test[TIMESTAMP].max()),
        })
        fitted[name] = model

    if not rows:
        stop("All model fits failed. No comparison or model bundle was saved.")

    comparison = pd.DataFrame(rows).sort_values("Failure_F1", ascending=False).reset_index(drop=True)
    comparison.to_csv(COMPARISON_FILE, index=False)

    metric_columns = ["Accuracy", "Failure_Precision", "Failure_Recall", "Failure_F1"]
    ax = comparison.set_index("Model")[metric_columns].plot(
        kind="bar", figsize=(11, 6), ylim=(0, 1)
    )
    ax.set_title("MetroPT-3 held-out event model comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "model_metric_comparison.png", dpi=150)
    plt.close()

    print("\n" + "=" * 78)
    print("MODEL COMPARISON")
    print("=" * 78)
    print(comparison[[
        "Model", "Accuracy", "Failure_Precision", "Failure_Recall", "Failure_F1",
        "False_Negatives", "False_Positives",
    ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    best = comparison.iloc[0]
    best_name = best["Model"]
    bundle = {
        "pipeline": fitted[best_name],
        "feature_columns": feature_columns,
        "target_name": TARGET,
        "best_model_name": best_name,
        "selected_by": "Failure-class F1-score on one held-out event window",
        "random_state": RANDOM_STATE,
        "held_out_event": held_out,
        "test_window_start": test_start,
        "test_window_end": test_end,
        "training_cutoff": training_cutoff,
        "warning": (
            "Limited event-based case study. Class 0 means outside documented intervals, "
            "not confirmed healthy. This is not proof of early-warning prediction."
        ),
    }
    joblib.dump(bundle, BEST_MODEL_FILE)

    print(f"\nBest model by failure-class F1-score: {best_name}")
    print(f"Selection reason: failure-class F1 = {best['Failure_F1']:.4f}")
    print("\nMaintenance interpretation:")
    print("  False negative: a row inside a documented failure interval was missed.")
    print("  False positive: a row outside documented intervals was flagged.")
    print("\nLIMITATIONS:")
    print("  - Only four documented events are available.")
    print("  - One held-out event is a small case study; performance may not generalize.")
    print("  - Labels mark timestamps inside failure intervals, not future failure horizons.")
    print("  - This does not demonstrate early-warning prediction.")
    print("\nSaved outputs:")
    print(f"  Comparison CSV: {COMPARISON_FILE}")
    print(f"  Best model bundle: {BEST_MODEL_FILE}")
    print(f"  Plots/confusion matrices: {PLOT_DIR}")
    print("Original and engineered CSV files were not modified.")
    print("Done.")


if __name__ == "__main__":
    main()
