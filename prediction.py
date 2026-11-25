"""
prediction.py - Model loading, validation, feature engineering and prediction.

The feature logic here is a faithful copy of your existing scripts:

AI4I 2020  (ai4i_feature_engineering.py / ai4i_model_training.py)
    * 4 engineered features are added to the 6 raw inputs.
    * The saved model was trained on the already preprocessed training CSV
      (median imputation + StandardScaler for numbers, most-frequent imputation +
      OneHotEncoder for Type and Tool_Wear_Category). That preprocessor was not
      saved, so it is rebuilt from ai4i_engineered.csv using the same split
      (test_size=0.20, random_state=42, stratified) and the same steps. At start-up
      it is compared with ai4i_X_train.csv when that file exists.

MetroPT-3  (compressor_feature_engineering.py / compressor_model_training.py)
    * The model file is a dict bundle that stores 'feature_columns'; the exact input
      list and order are read from it.
    * Motor_Current_Change and Oil_Temperature_Change are differences from the
      previous chronological reading, so a previous reading is required.

Manual input and file upload both go through predict_frame().
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as cfg


class PredictionError(Exception):
    """Raised for problems with model files or with running the model."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ModelBundle:
    key: str
    pipeline: Any
    algorithm: str
    model_input_columns: list          # exact columns the fitted model receives, in order
    input_columns: list                # raw columns the user must supply
    engineered_columns: list           # engineered features shown to the user
    stat_columns: list                 # numeric columns used for reference ranges
    preprocessor: Any = None           # AI4I only
    preprocessor_feature_names: list = field(default_factory=list)
    verification: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def spec(self) -> dict:
        return cfg.MODELS[self.key]


@dataclass
class ValidationReport:
    n_rows: int = 0
    fatal: str | None = None
    missing_columns: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    issues: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(
        columns=["File row", "Column", "Severity", "Problem", "Value"]))
    invalid: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    clean: pd.DataFrame = field(default_factory=pd.DataFrame)
    extras: pd.DataFrame = field(default_factory=pd.DataFrame)
    order: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    @property
    def n_invalid(self) -> int:
        return int(self.invalid.sum())

    @property
    def n_errors(self) -> int:
        return int((self.issues["Severity"] == "Error").sum()) if len(self.issues) else 0

    @property
    def n_warnings(self) -> int:
        return int((self.issues["Severity"] == "Warning").sum()) if len(self.issues) else 0


@dataclass
class PredictionResult:
    machine_key: str
    model_name: str
    algorithm: str
    created_at: datetime
    input_method: str
    n_input_rows: int = 0
    n_records: int = 0
    n_failure: int = 0
    n_non_failure: int = 0
    n_invalid: int = 0
    has_probability: bool = False
    results: pd.DataFrame = field(default_factory=pd.DataFrame)
    features: pd.DataFrame = field(default_factory=pd.DataFrame)
    issues: pd.DataFrame = field(default_factory=pd.DataFrame)
    input_columns: list = field(default_factory=list)
    engineered_columns: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    source_name: str | None = None
    error: str | None = None

    def detail_records(self, limit: int | None = None) -> list[dict]:
        """Row-level records for MongoDB. Predicted failures are stored first."""
        if self.results.empty:
            return []
        limit = limit or cfg.MAX_STORED_DETAIL_ROWS
        order = np.argsort(-self.results["Predicted class"].to_numpy(), kind="stable")[:limit]
        order = np.sort(order) if self.n_failure == 0 else order
        records = []
        for pos in order:
            row = self.results.iloc[pos]
            inputs = {c: _py(row[c]) for c in self.input_columns}
            engineered = {c: _py(row[c]) for c in self.engineered_columns}
            records.append({
                "row_number": int(row["File row"]),
                "predicted_class": int(row["Predicted class"]),
                "failure_probability": _py(row["Failure probability"]) if self.has_probability else None,
                "inputs": inputs,
                "engineered": engineered,
            })
        return records


def _py(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
_ALGORITHM_NAMES = {
    "RandomForestClassifier": "Random Forest",
    "DecisionTreeClassifier": "Decision Tree",
    "LogisticRegression": "Logistic Regression",
}


def _final_estimator(pipeline):
    return pipeline.steps[-1][1] if hasattr(pipeline, "steps") else pipeline


def _fit_ai4i_preprocessor():
    """Rebuild the AI4I preprocessor exactly as ai4i_feature_engineering.py fitted it."""
    path = cfg.MODELS["ai4i"]["reference_csv"]
    if not path.exists():
        raise PredictionError(
            f"{path.name} was not found in the project folder. It is needed to rebuild the AI4I "
            "scaler and encoder (they were not saved by the feature-engineering script)."
        )
    data = pd.read_csv(path)
    if cfg.AI4I_TARGET not in data.columns:
        raise PredictionError(f"{path.name} has no '{cfg.AI4I_TARGET}' column.")
    X = data.drop(columns=[cfg.AI4I_TARGET])
    y = data[cfg.AI4I_TARGET].astype(int)
    X_train, _, _, _ = train_test_split(
        X, y, test_size=cfg.AI4I_TEST_SIZE, random_state=cfg.AI4I_RANDOM_STATE, stratify=y
    )
    numeric = [c for c in X_train.columns if is_numeric_dtype(X_train[c]) and not is_bool_dtype(X_train[c])]
    categorical = [c for c in X_train.columns if c not in numeric]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older scikit-learn
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                  ("scaler", StandardScaler())]), numeric),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                      ("onehot", encoder)]), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor.fit(X_train)
    return preprocessor, X_train


def _verify_ai4i_preprocessor(preprocessor, X_train, names) -> dict:
    """Compare the rebuilt preprocessor with your saved training matrix, if present."""
    for filename in ("ai4i_X_train.csv", "X_train.csv"):
        path = cfg.BASE_DIR / filename
        if not path.exists():
            continue
        try:
            saved = pd.read_csv(path).reindex(columns=list(names))
            ours = pd.DataFrame(preprocessor.transform(X_train), columns=list(names))
            same = saved.shape == ours.shape and np.allclose(
                saved.to_numpy(dtype=float), ours.to_numpy(dtype=float), atol=1e-6, equal_nan=True)
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "message": f"Could not compare with {filename}: {exc}"}
        if same:
            return {"status": "verified",
                    "message": f"Rebuilt preprocessor matches {filename} ({len(saved):,} rows)."}
        return {"status": "mismatch",
                "message": f"Rebuilt preprocessor does NOT match {filename}. Predictions may be unreliable."}
    return {"status": "not_checked",
            "message": "ai4i_X_train.csv not found, so the rebuilt preprocessor was not cross-checked."}


def load_model(key: str) -> ModelBundle:
    """Load a saved model and work out its exact input schema."""
    spec = cfg.MODELS[key]
    path = spec["model_path"]
    if not path.exists():
        raise PredictionError(
            f"Model file not found: {path}. Copy {spec['model_file']} into the models/ folder."
        )
    try:
        obj = joblib.load(path)
    except Exception as exc:  # noqa: BLE001
        raise PredictionError(
            f"Could not load {path.name}: {exc}. This is usually a scikit-learn version mismatch; "
            "install the same scikit-learn version that trained the model."
        ) from exc

    metadata, saved_features = {}, None
    if isinstance(obj, dict):
        pipeline = obj.get("pipeline")
        saved_features = obj.get("feature_columns")
        metadata = {k: v for k, v in obj.items() if k not in ("pipeline",)}
    else:
        pipeline = obj
    if pipeline is None or not hasattr(pipeline, "predict"):
        raise PredictionError(f"{path.name} does not contain a usable scikit-learn model.")

    estimator = _final_estimator(pipeline)
    algorithm = _ALGORITHM_NAMES.get(type(estimator).__name__, type(estimator).__name__)
    fitted_names = getattr(pipeline, "feature_names_in_", None)

    if key == "compressor":
        columns = list(saved_features) if saved_features else (list(fitted_names) if fitted_names is not None else None)
        if not columns:
            raise PredictionError("The compressor model file does not list its input columns.")
        if fitted_names is not None and set(fitted_names) != set(columns):
            raise PredictionError("The compressor model's saved feature list and fitted columns disagree.")
        if fitted_names is not None:
            columns = list(fitted_names)
        engineered = [c for c in columns if c in cfg.COMPRESSOR_ENGINEERED]
        raw = [c for c in columns if c not in cfg.COMPRESSOR_ENGINEERED]
        for feature in engineered:
            for source in cfg.COMPRESSOR_SOURCES[feature]:
                if source not in raw:
                    raw.append(source)
        return ModelBundle(
            key=key, pipeline=pipeline, algorithm=metadata.get("best_model_name", algorithm),
            model_input_columns=columns, input_columns=raw, engineered_columns=engineered,
            stat_columns=raw + engineered, metadata=metadata,
        )

    # AI4I
    if fitted_names is None:
        raise PredictionError("The AI4I model does not record its input column names.")
    preprocessor, X_train = _fit_ai4i_preprocessor()
    names = [str(n) for n in preprocessor.get_feature_names_out()]
    unknown = [c for c in fitted_names if c not in names]
    if unknown:
        raise PredictionError(
            "The AI4I model expects columns that the feature-engineering script does not produce: "
            f"{unknown}. Retrain the model on the 14 preprocessed features only."
        )
    numeric_stats = [c for c in cfg.AI4I_RAW_COLUMNS if c != "Type"] + [
        c for c in cfg.AI4I_ENGINEERED if c != "Tool_Wear_Category"]
    return ModelBundle(
        key=key, pipeline=pipeline, algorithm=algorithm, model_input_columns=list(fitted_names),
        input_columns=list(cfg.AI4I_RAW_COLUMNS), engineered_columns=list(cfg.AI4I_ENGINEERED),
        stat_columns=numeric_stats, preprocessor=preprocessor, preprocessor_feature_names=names,
        verification=_verify_ai4i_preprocessor(preprocessor, X_train, names), metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Reference statistics (ranges seen in your engineered CSV files)
# ---------------------------------------------------------------------------
def _describe(frame: pd.DataFrame) -> dict:
    out = {}
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        values = values[np.isfinite(values)]
        if values.empty:
            continue
        out[column] = {
            "min": float(values.min()), "max": float(values.max()),
            "median": float(values.median()), "mean": float(values.mean()),
        }
    return out


@functools.lru_cache(maxsize=8)
def reference_stats(key: str, columns: tuple) -> dict:
    """min / max / median of numeric columns in the engineered CSV (empty if unavailable)."""
    path = cfg.MODELS[key]["reference_csv"]
    if not path.exists():
        return {}
    try:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        wanted = [c for c in columns if c in header]
        if not wanted:
            return {}
        if key == "ai4i":
            return _describe(pd.read_csv(path, usecols=wanted))
        chunks = pd.read_csv(path, usecols=wanted, dtype="float32", chunksize=250_000)
        return _describe(pd.concat(chunks, ignore_index=True))
    except Exception:  # noqa: BLE001
        return {}


def stats_for(bundle: ModelBundle) -> dict:
    return reference_stats(bundle.key, tuple(bundle.stat_columns))


def is_digital(bundle: ModelBundle, column: str) -> bool:
    """True for 0/1 compressor signals."""
    if bundle.key != "compressor":
        return False
    return cfg.COMPRESSOR_SENSORS.get(column, {}).get("unit") == "0/1"


# ---------------------------------------------------------------------------
# Feature engineering (same formulas as your scripts)
# ---------------------------------------------------------------------------
def engineer_ai4i(frame: pd.DataFrame) -> pd.DataFrame:
    X = frame[cfg.AI4I_RAW_COLUMNS].copy()
    X["Temperature_Difference"] = X["Process temperature [K]"] - X["Air temperature [K]"]
    X["Power_Proxy"] = X["Rotational speed [rpm]"] * X["Torque [Nm]"]
    X["Tool_Wear_Category"] = pd.cut(
        X["Tool wear [min]"],
        bins=[-np.inf, *cfg.AI4I_WEAR_BINS, np.inf],
        labels=cfg.AI4I_WEAR_LABELS,
        include_lowest=True,
    ).astype("object")
    safe_rpm = X["Rotational speed [rpm]"].replace(0, np.nan)
    X["Torque_Per_RPM"] = X["Torque [Nm]"] / safe_rpm
    return X


def engineer_compressor(frame: pd.DataFrame) -> pd.DataFrame:
    """frame must already be in chronological order."""
    X = frame.copy()
    if {"TP3", "TP2"} <= set(X.columns):
        X["Pressure_Difference_TP3_TP2"] = X["TP3"] - X["TP2"]
    if {"Reservoirs", "TP2"} <= set(X.columns):
        X["Pressure_Difference_Reservoir_TP2"] = X["Reservoirs"] - X["TP2"]
    if "Motor_current" in X.columns:
        X["Motor_Current_Change"] = X["Motor_current"].diff()
    if "Oil_temperature" in X.columns:
        X["Oil_Temperature_Change"] = X["Oil_temperature"].diff()
    return X


def _engineer(bundle: ModelBundle, clean: pd.DataFrame) -> pd.DataFrame:
    if bundle.key == "ai4i":
        return engineer_ai4i(clean)
    return engineer_compressor(clean)


def _model_matrix(bundle: ModelBundle, engineered: pd.DataFrame) -> pd.DataFrame:
    """Build exactly the matrix the fitted model was trained on."""
    if bundle.key == "ai4i":
        pre = bundle.preprocessor
        X = engineered[list(pre.feature_names_in_)]
        matrix = pd.DataFrame(pre.transform(X), columns=bundle.preprocessor_feature_names, index=X.index)
        return matrix[bundle.model_input_columns]
    X = engineered[bundle.model_input_columns].replace([np.inf, -np.inf], np.nan)
    return X


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _parse_timestamps(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, format="mixed", errors="coerce")
    except (TypeError, ValueError):
        return pd.to_datetime(series, errors="coerce")


def _shown(series: pd.Series) -> pd.Series:
    text = series.astype(object).where(series.notna(), "(blank)").astype(str)
    return text.where(text.str.strip() != "", "(blank)")


def validate_input(bundle: ModelBundle, raw_df: pd.DataFrame, first_row_number: int = 2) -> ValidationReport:
    """Check columns, types, missing values and ranges. Never drops rows silently."""
    df = raw_df.copy().reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    n = len(df)
    report = ValidationReport(n_rows=n)

    required = list(bundle.input_columns)
    if df.columns.duplicated().any():
        report.fatal = "The file has duplicate column names: " + ", ".join(
            sorted(set(df.columns[df.columns.duplicated()])))
        return report

    lower = {c.lower(): c for c in df.columns}
    renames = {lower[c.lower()]: c for c in required
               if c not in df.columns and c.lower() in lower and lower[c.lower()] not in required}
    if renames:
        df = df.rename(columns=renames)
        report.notes.append("Column names matched ignoring letter case: "
                            + ", ".join(f"'{a}' -> '{b}'" for a, b in renames.items()))

    missing = [c for c in required if c not in df.columns]
    report.missing_columns = missing
    if n == 0:
        report.fatal = "The file contains no data rows."
        return report
    if missing:
        report.fatal = ("Required column(s) missing for this machine's model: " + ", ".join(missing)
                        + ". Each machine only accepts its own model's columns.")
        return report

    file_row = np.arange(n) + first_row_number
    invalid = np.zeros(n, dtype=bool)
    parts: list[pd.DataFrame] = []

    def flag(mask, column, severity, problem, shown):
        idx = np.flatnonzero(np.asarray(mask, dtype=bool))
        if idx.size == 0:
            return
        if severity == "Error":
            invalid[idx] = True
        parts.append(pd.DataFrame({
            "File row": file_row[idx], "Column": column, "Severity": severity,
            "Problem": problem, "Value": shown.iloc[idx].to_numpy(),
        }))

    clean = pd.DataFrame({"File row": file_row})
    stats = stats_for(bundle)

    # Categorical column (AI4I Type)
    if bundle.key == "ai4i":
        original = df["Type"]
        shown = _shown(original)
        text = original.astype(str).str.strip().str.upper()
        blank = original.isna() | (original.astype(str).str.strip() == "")
        flag(blank, "Type", "Error", "Missing value", shown)
        flag(~blank & ~text.isin(cfg.AI4I_TYPES), "Type", "Error", "Type must be L, M or H", shown)
        clean["Type"] = text.where(~blank, np.nan).to_numpy()

    numeric_columns = [c for c in required if not (bundle.key == "ai4i" and c == "Type")]
    for column in numeric_columns:
        original = df[column]
        shown = _shown(original)
        coerced = pd.to_numeric(original, errors="coerce").astype(float)
        blank = original.isna() | (original.astype(str).str.strip() == "")
        flag(blank, column, "Error", "Missing value", shown)
        flag(coerced.isna() & ~blank, column, "Error", "Not a number", shown)
        flag(np.isinf(coerced), column, "Error", "Infinite value", shown)
        coerced = coerced.replace([np.inf, -np.inf], np.nan)

        if bundle.key == "ai4i" and column in cfg.AI4I_RULES:
            rule, message = cfg.AI4I_RULES[column]
            bad = (coerced <= 0) if rule == "gt0" else (coerced < 0)
            flag(bad.to_numpy(), column, "Error", message, shown)
        if is_digital(bundle, column):
            flag((~coerced.isin([0, 1]) & coerced.notna()).to_numpy(), column, "Error",
                 "Digital signal must be 0 or 1", shown)
        elif column in stats:
            s = stats[column]
            margin = 1e-4 * (s["max"] - s["min"]) + 1e-6
            flag((coerced < s["min"] - margin).to_numpy(), column, "Warning",
                 f"Below the range in your reference data (min {s['min']:.4g})", shown)
            flag((coerced > s["max"] + margin).to_numpy(), column, "Warning",
                 f"Above the range in your reference data (max {s['max']:.4g})", shown)
        clean[column] = coerced.to_numpy()

    # Chronological order (compressor)
    order = np.arange(n)
    ts_name = next((c for c in df.columns if c.lower() == cfg.COMPRESSOR_TIMESTAMP), None)
    if bundle.key == "compressor":
        if ts_name:
            ts = _parse_timestamps(df[ts_name])
            shown = _shown(df[ts_name])
            flag(ts.isna().to_numpy(), ts_name, "Error", "Timestamp missing or not recognised", shown)
            flag((ts.duplicated(keep=False) & ts.notna()).to_numpy(), ts_name, "Warning",
                 "Duplicate timestamp", shown)
            order = ts.sort_values(kind="stable", na_position="last").index.to_numpy()
            if not np.array_equal(order, np.arange(n)):
                report.notes.append("Rows were re-ordered by timestamp before the change features were "
                                    "calculated (as in training).")
        else:
            report.notes.append("No 'timestamp' column: rows are assumed to be in chronological order, and the "
                                "first row has no previous reading.")

    report.issues = (pd.concat(parts, ignore_index=True).sort_values(["File row", "Column"], kind="stable")
                     .reset_index(drop=True) if parts else report.issues)
    report.invalid = invalid
    report.clean = clean
    report.order = np.asarray(order)
    report.extras = df[[c for c in df.columns if c not in required]].reset_index(drop=True)
    return report


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_frame(
    bundle: ModelBundle,
    raw_df: pd.DataFrame,
    input_method: str,
    *,
    first_row_number: int = 2,
    context_rows: int = 0,
    report: ValidationReport | None = None,
    source_name: str | None = None,
) -> PredictionResult:
    """The single prediction pipeline used by both manual input and uploaded files.

    context_rows: leading rows that are only used to calculate change features
    (the 'previous reading' of a manual compressor entry); they are not predicted.
    """
    result = PredictionResult(
        machine_key=bundle.key, model_name=bundle.spec["dataset"], algorithm=bundle.algorithm,
        created_at=datetime.now(timezone.utc), input_method=input_method, source_name=source_name,
        input_columns=list(bundle.input_columns), engineered_columns=list(bundle.engineered_columns),
    )
    report = report or validate_input(bundle, raw_df, first_row_number)
    result.issues = report.issues
    result.notes = list(report.notes)
    result.n_input_rows = max(report.n_rows - context_rows, 0)
    if report.fatal:
        result.error = report.fatal
        return result

    order = report.order
    clean = report.clean.iloc[order].reset_index(drop=True)
    invalid = report.invalid[order]
    extras = report.extras.iloc[order].reset_index(drop=True)
    is_context = np.arange(len(clean)) < context_rows
    result.n_invalid = int((invalid & ~is_context).sum())

    try:
        engineered = _engineer(bundle, clean)
    except Exception as exc:  # noqa: BLE001
        raise PredictionError(f"Feature engineering failed: {exc}") from exc

    extra_issues = []
    if bundle.key == "compressor":
        change_cols = [c for c in ("Motor_Current_Change", "Oil_Temperature_Change") if c in bundle.engineered_columns]
        if change_cols:
            gap = engineered[change_cols].isna().any(axis=1).to_numpy() & ~invalid & ~is_context
            for pos in np.flatnonzero(gap):
                extra_issues.append({
                    "File row": int(clean.loc[pos, "File row"]), "Column": ", ".join(change_cols),
                    "Severity": "Warning",
                    "Problem": "No valid previous reading, so the change features are missing; the model's "
                               "median imputer was applied",
                    "Value": "(not available)",
                })
    if extra_issues:
        result.issues = pd.concat([result.issues, pd.DataFrame(extra_issues)], ignore_index=True)

    keep = ~invalid & ~is_context
    if not keep.any():
        result.error = "No valid records are available for prediction. See the validation issues."
        return result

    matrix = _model_matrix(bundle, engineered.loc[keep])
    try:
        predicted = np.asarray(bundle.pipeline.predict(matrix)).astype(int)
    except Exception as exc:  # noqa: BLE001
        raise PredictionError(f"The model could not predict: {exc}") from exc

    probability = None
    if hasattr(bundle.pipeline, "predict_proba"):
        try:
            classes = list(bundle.pipeline.classes_)
            if 1 in classes:
                probability = bundle.pipeline.predict_proba(matrix)[:, classes.index(1)]
        except Exception:  # noqa: BLE001
            probability = None

    out = pd.DataFrame({"File row": clean.loc[keep, "File row"].to_numpy()})
    out["Prediction"] = np.where(predicted == 1, "Predicted failure", "Predicted non-failure")
    out["Predicted class"] = predicted
    if probability is not None:
        out["Failure probability"] = probability
    for column in bundle.input_columns:
        out[column] = engineered.loc[keep, column].to_numpy()
    for column in bundle.engineered_columns:
        out[column] = engineered.loc[keep, column].to_numpy()
    for column in extras.columns:
        out[column] = extras.loc[keep, column].to_numpy()

    features = engineered.loc[keep].reset_index(drop=True)
    sort_order = np.argsort(out["File row"].to_numpy(), kind="stable")
    result.results = out.iloc[sort_order].reset_index(drop=True)
    result.features = features.iloc[sort_order].reset_index(drop=True)
    result.n_records = int(len(out))
    result.n_failure = int((predicted == 1).sum())
    result.n_non_failure = int((predicted == 0).sum())
    result.has_probability = probability is not None
    return result


def manual_frame(bundle: ModelBundle, values: dict, previous: dict | None = None):
    """Build the DataFrame for a manual entry. Returns (frame, context_rows, first_row_number)."""
    row = {c: values[c] for c in bundle.input_columns}
    if bundle.key == "compressor" and previous:
        earlier = dict(row)
        for column in cfg.COMPRESSOR_PREVIOUS_COLUMNS:
            if column in previous and column in earlier:
                earlier[column] = previous[column]
        return pd.DataFrame([earlier, row]), 1, 0
    return pd.DataFrame([row]), 0, 1


def predict_manual(bundle: ModelBundle, values: dict, previous: dict | None = None) -> PredictionResult:
    frame, context_rows, first_row = manual_frame(bundle, values, previous)
    return predict_frame(bundle, frame, "Manual input", first_row_number=first_row, context_rows=context_rows)


def preview_features(bundle: ModelBundle, values: dict, previous: dict | None = None):
    """Engineered values for the entered readings (same engineering code, no model call)."""
    frame, _, first_row = manual_frame(bundle, values, previous)
    report = validate_input(bundle, frame, first_row)
    if report.fatal:
        return None
    clean = report.clean.iloc[report.order].reset_index(drop=True)
    engineered = _engineer(bundle, clean)
    return engineered.iloc[-1][bundle.engineered_columns]


def blank_template(bundle: ModelBundle) -> pd.DataFrame:
    columns = list(bundle.input_columns)
    if bundle.key == "compressor":
        columns = [cfg.COMPRESSOR_TIMESTAMP, *columns]
    return pd.DataFrame(columns=columns)


# ---------------------------------------------------------------------------
# Explanations (only what the fitted model supports)
# ---------------------------------------------------------------------------
def feature_importance(bundle: ModelBundle) -> pd.DataFrame | None:
    estimator = _final_estimator(bundle.pipeline)
    importance = getattr(estimator, "feature_importances_", None)
    if importance is None and hasattr(estimator, "coef_"):
        importance = np.abs(np.asarray(estimator.coef_)).ravel()
    names = bundle.model_input_columns
    if importance is None or len(importance) != len(names):
        return None
    table = pd.DataFrame({"Feature": names, "Importance": np.asarray(importance, dtype=float)})
    return table.sort_values("Importance", ascending=False).reset_index(drop=True)


def explain_row(bundle: ModelBundle, features_row: pd.Series, top_n: int = 8) -> pd.DataFrame | None:
    """Most important model features with this record's value next to your reference data."""
    table = feature_importance(bundle)
    if table is None:
        return None
    stats = stats_for(bundle)
    rows = []
    for _, item in table.head(top_n).iterrows():
        feature = item["Feature"]
        value, source = None, feature
        if feature in features_row.index:
            value = features_row[feature]
        else:  # one-hot column such as Type_L or Tool_Wear_Category_High
            for categorical in ("Type", "Tool_Wear_Category"):
                if feature.startswith(categorical + "_") and categorical in features_row.index:
                    value = f"{features_row[categorical]}"
                    source = categorical
        ref = stats.get(feature)
        rows.append({
            "Model feature": feature,
            "Importance": float(item["Importance"]),
            "This record": value,
            "Reference median": ref["median"] if ref else None,
            "Reference min": ref["min"] if ref else None,
            "Reference max": ref["max"] if ref else None,
        })
    return pd.DataFrame(rows)
