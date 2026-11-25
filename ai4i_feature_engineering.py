
"""
AI4I 2020 Predictive Maintenance Dataset
Feature Engineering and Data Preparation

This program:
1. Loads and examines the dataset.
2. Cleans missing values and duplicate rows.
3. Creates four engineered features.
4. Visualizes the data.
5. Prevents target leakage.
6. Splits data into training and testing sets.
7. Preprocesses numerical and categorical features.
8. Saves the engineered dataset and train/test CSV files.

No machine-learning model is trained in this program.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# ============================================================
# 1. CONFIGURATION
# ============================================================

TARGET = "Machine failure"

IDENTIFIER_COLUMNS = ["UDI", "Product ID"]

# These columns directly indicate specific failure modes.
# Excluding them prevents target leakage.
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

REQUIRED_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    TARGET,
]


# ============================================================
# 2. FIND AND LOAD THE DATASET
# ============================================================

def find_dataset():
    """
    Accept a CSV path from the command line.
    Otherwise, look for common AI4I filenames beside this script.
    """

    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser().resolve()

    folder = Path(__file__).resolve().parent

    possible_files = [
        folder / "ai4i2020.csv",
        folder / "ai4i2020(1).csv",
    ]

    for file_path in possible_files:
        if file_path.exists():
            return file_path

    return possible_files[0]


def main():

    input_path = find_dataset()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}\n"
            "Place ai4i2020.csv or ai4i2020(1).csv beside this script, "
            "or provide the CSV path when running the program."
        )

    output_folder = input_path.parent

    print("=" * 65)
    print("AI4I 2020 FEATURE ENGINEERING")
    print("=" * 65)

    print("\nDataset path:", input_path)

    # Read the source file. All modifications are made to copies.
    df = pd.read_csv(input_path)

    original_shape = df.shape

    print("\nOriginal dataset shape:", original_shape)

    # Check that the required columns exist.
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns are missing: "
            + ", ".join(missing_columns)
        )

    # ========================================================
    # 3. UNDERSTAND THE DATASET
    # ========================================================

    print("\n--- Column Names ---")
    print(df.columns.tolist())

    print("\n--- First Five Rows ---")
    print(df.head())

    print("\n--- Data Types ---")
    print(df.dtypes)

    print("\n--- Missing Values Before Cleaning ---")
    print(df.isnull().sum())

    print("\n--- Duplicate Rows ---")
    print("Number of duplicate rows:", df.duplicated().sum())

    print("\n--- Descriptive Statistics ---")
    print(df.describe(include="all").T)

    print("\n--- Numerical Columns ---")
    print(df.select_dtypes(include=np.number).columns.tolist())

    print("\n--- Categorical Columns ---")
    print(
        df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
    )

    # ========================================================
    # 4. TARGET CLASS DISTRIBUTION
    # ========================================================

    if df[TARGET].isnull().any():
        raise ValueError(
            "The target column contains missing values. "
            "Target labels must be investigated rather than imputed."
        )

    target_values = set(df[TARGET].unique())

    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"Expected binary 0/1 target labels, found {target_values}"
        )

    df[TARGET] = df[TARGET].astype(int)

    print("\n--- Target Class Distribution ---")
    print(
        df[TARGET]
        .value_counts()
        .sort_index()
        .rename(index={0: "No Failure", 1: "Failure"})
    )

    print("\nTarget percentages:")
    print(
        (df[TARGET].value_counts(normalize=True) * 100)
        .round(2)
    )

    # ========================================================
    # 5. DATA CLEANING
    # ========================================================

    # Create a copy so that the original CSV is never modified.
    cleaned = df.copy()

    # Remove exact duplicate records to avoid counting the same
    # observation more than once.
    duplicate_count = cleaned.duplicated().sum()

    if duplicate_count > 0:
        cleaned = cleaned.drop_duplicates().copy()

    print("\nDuplicate rows removed:", duplicate_count)

    # Keep target labels separate from all feature transformations.
    y = cleaned[TARGET].copy()

    # Remove target, identifiers and failure-mode flags from X.
    # The failure-mode flags are not general operating measurements
    # and can reveal information about the target.
    excluded_columns = (
        IDENTIFIER_COLUMNS + LEAKAGE_COLUMNS
    )

    excluded_columns_present = [
        column
        for column in excluded_columns
        if column in cleaned.columns
    ]

    X = cleaned.drop(
        columns=[TARGET] + excluded_columns_present,
        errors="ignore"
    ).copy()

    # The target is kept separately and is never used to create features.
    print("\nFeatures before engineering:")
    print(X.columns.tolist())

    # ========================================================
    # 6. FEATURE ENGINEERING
    # ========================================================

    # Feature 1:
    # Temperature difference between the process and surrounding air.
    X["Temperature_Difference"] = (
        X["Process temperature [K]"]
        - X["Air temperature [K]"]
    )

    # Feature 2:
    # Rotational speed multiplied by torque.
    # This is a relative mechanical power proxy, NOT a direct
    # measurement of electrical power.
    X["Power_Proxy"] = (
        X["Rotational speed [rpm]"]
        * X["Torque [Nm]"]
    )

    # Feature 3:
    # Tool wear grouped into interpretable categories.
    # These are practical thresholds, not official failure limits.
    X["Tool_Wear_Category"] = pd.cut(
        X["Tool wear [min]"],
        bins=[-np.inf, 100, 200, np.inf],
        labels=["Low", "Medium", "High"],
        include_lowest=True
    ).astype("object")

    # Feature 4:
    # Torque divided by RPM.
    # Zero RPM is replaced with NaN to avoid division by zero.
    safe_rpm = X["Rotational speed [rpm]"].replace(0, np.nan)

    X["Torque_Per_RPM"] = (
        X["Torque [Nm]"] / safe_rpm
    )

    new_features = [
        "Temperature_Difference",
        "Power_Proxy",
        "Tool_Wear_Category",
        "Torque_Per_RPM"
    ]

    print("\n--- New Features Created ---")

    for feature in new_features:
        print(" -", feature)

    # Rejoin the target only for saving and visualization.
    engineered = X.copy()
    engineered[TARGET] = y.loc[X.index]

    print("\nOriginal dataset shape:", original_shape)
    print("Cleaned dataset shape:", cleaned.shape)
    print("Engineered dataset shape:", engineered.shape)

    # ========================================================
    # 7. VISUALIZATIONS
    # ========================================================

    sns.set_theme(style="whitegrid")

    plot_folder = output_folder / "ai4i_plots"
    plot_folder.mkdir(parents=True, exist_ok=True)

    # Plot 1: Target class distribution.
    # Helps show the imbalance between failure and non-failure.
    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=engineered,
        x=TARGET
    )

    plt.title("Machine Failure Class Distribution")
    plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
    plt.ylabel("Number of Records")
    plt.tight_layout()

    plt.savefig(
        plot_folder / "target_class_distribution.png",
        dpi=150
    )
    plt.close()

    # Plot 2: Correlation heatmap of numerical features.
    # Identifiers and failure-mode flags are excluded.
    numeric_data = engineered.select_dtypes(
        include=np.number
    ).drop(
        columns=[
            c for c in IDENTIFIER_COLUMNS + LEAKAGE_COLUMNS
            if c in engineered.columns
        ],
        errors="ignore"
    )

    plt.figure(figsize=(13, 9))

    sns.heatmap(
        numeric_data.corr(),
        cmap="coolwarm",
        center=0,
        annot=False
    )

    plt.title("Correlation Heatmap of Numerical Features")
    plt.tight_layout()

    plt.savefig(
        plot_folder / "correlation_heatmap.png",
        dpi=150
    )
    plt.close()

    # Plot 3: Compare selected features for failure and non-failure.
    # These plots help reveal differences in operating conditions.
    selected_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Temperature_Difference",
        "Power_Proxy"
    ]

    for feature in selected_features:

        plt.figure(figsize=(7, 5))

        sns.boxplot(
            data=engineered,
            x=TARGET,
            y=feature
        )

        plt.title(f"{feature}: Failure vs Non-Failure")
        plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
        plt.ylabel(feature)
        plt.tight_layout()

        safe_name = (
            feature.lower()
            .replace(" ", "_")
            .replace("[", "")
            .replace("]", "")
            .replace("/", "_")
        )

        plt.savefig(
            plot_folder / f"comparison_{safe_name}.png",
            dpi=150
        )

        plt.close()

    # Plot 4: Compare an original feature with an engineered feature.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(
        data=engineered,
        x="Process temperature [K]",
        kde=True,
        ax=axes[0]
    )

    axes[0].set_title("Original: Process Temperature")

    sns.histplot(
        data=engineered,
        x="Temperature_Difference",
        kde=True,
        ax=axes[1]
    )

    axes[1].set_title("Engineered: Temperature Difference")

    plt.tight_layout()

    plt.savefig(
        plot_folder / "feature_distributions.png",
        dpi=150
    )

    plt.close()

    print("\nPlots saved in:", plot_folder)

    # ========================================================
    # 8. TRAIN-TEST SPLIT
    # ========================================================

    # Split BEFORE learning imputation, encoding or scaling parameters.
    # Stratification preserves approximately the same target proportions.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\n--- Train/Test Split ---")
    print("X_train:", X_train.shape)
    print("X_test: ", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test: ", y_test.shape)

    # ========================================================
    # 9. PREPROCESSING PIPELINE
    # ========================================================

    numerical_features = X_train.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print("\nNumerical features for modeling:")
    print(numerical_features)

    print("\nCategorical features for modeling:")
    print(categorical_features)

    # Numerical features:
    # - Missing values are filled with training-set medians.
    # - StandardScaler standardizes measurements.
    # Scaling is not essential for Random Forest, but is included
    # here as a reusable preprocessing step.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    # Categorical features:
    # - Missing values are filled using the most common category.
    # - OneHotEncoder converts categories into numeric columns.
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    # Compatibility with older and newer Scikit-learn versions.
    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            ("onehot", encoder)
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    # Fit all preprocessing steps on training data ONLY.
    X_train_array = preprocessor.fit_transform(X_train)

    # Apply the learned transformations to test data without refitting.
    X_test_array = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    # Convert processed arrays back to DataFrames for convenient CSV output.
    X_train_processed = pd.DataFrame(
        X_train_array,
        columns=feature_names,
        index=X_train.index
    )

    X_test_processed = pd.DataFrame(
        X_test_array,
        columns=feature_names,
        index=X_test.index
    )

    # ========================================================
    # 10. SAVE ALL OUTPUT FILES
    # ========================================================

    # Save the engineered data with the target.
    # IDs and failure-mode flags are excluded.
    # This version retains readable categorical values.
    engineered_path = output_folder / "ai4i_engineered.csv"

    engineered.to_csv(
        engineered_path,
        index=False
    )

    # Save preprocessed train/test features and target labels separately.
    # The original row index is retained to make matching X and y easier.
    X_train_path = output_folder / "X_train.csv"
    X_test_path = output_folder / "X_test.csv"
    y_train_path = output_folder / "y_train.csv"
    y_test_path = output_folder / "y_test.csv"

    X_train_processed.to_csv(
        X_train_path,
        index_label="original_row_index"
    )

    X_test_processed.to_csv(
        X_test_path,
        index_label="original_row_index"
    )

    y_train.to_frame(name=TARGET).to_csv(
        y_train_path,
        index_label="original_row_index"
    )

    y_test.to_frame(name=TARGET).to_csv(
        y_test_path,
        index_label="original_row_index"
    )

    # ========================================================
    # 11. FINAL REPORT
    # ========================================================

    print("\n" + "=" * 65)
    print("FINAL REPORT")
    print("=" * 65)

    print("\nDataset shapes:")
    print("Original:", original_shape)
    print("Cleaned:", cleaned.shape)
    print("Engineered:", engineered.shape)

    print("\nFeatures created:")
    for feature in new_features:
        print(" -", feature)

    print("\nFeatures excluded:")

    for column in excluded_columns_present:

        if column in IDENTIFIER_COLUMNS:
            reason = "Identifier; not used as a predictive feature."
        else:
            reason = (
                "Failure-mode indicator; excluded to reduce "
                "target leakage."
            )

        print(f" - {column}: {reason}")

    print("\nMissing values after cleaning:")
    print(cleaned.isnull().sum())

    print("\nMissing values in engineered data:")
    print(engineered.isnull().sum())

    print("\nTarget distribution in the complete dataset:")
    print(y.value_counts().sort_index())

    print("\nTraining target distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTesting target distribution:")
    print(y_test.value_counts().sort_index())

    print("\nFinal processed shapes:")
    print("X_train:", X_train_processed.shape)
    print("X_test:", X_test_processed.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    # Validate row counts and target separation before finishing.
    assert len(X_train_processed) == len(y_train)
    assert len(X_test_processed) == len(y_test)
    assert TARGET not in X_train_processed.columns
    assert TARGET not in X_test_processed.columns

    print("\nOutput files saved:")

    for path in [
        engineered_path,
        X_train_path,
        X_test_path,
        y_train_path,
        y_test_path
    ]:
        print(" -", path)

    print("\nVisualizations saved in:", plot_folder)

    print("\nFeature explanations:")
    print(
        "1. Temperature_Difference: measures the temperature gap."
    )
    print(
        "2. Power_Proxy: combines rotational speed and torque; "
        "it is not direct electrical power."
    )
    print(
        "3. Tool_Wear_Category: groups tool wear into Low, Medium "
        "and High bands."
    )
    print(
        "4. Torque_Per_RPM: measures torque relative to rotational speed."
    )

    print("\nNo machine-learning model was trained.")
    print("Feature engineering and data preparation completed successfully.")


# ============================================================
# 12. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except (FileNotFoundError, ValueError, KeyError) as error:
        print("\nERROR:", error)
        sys.exit(1)
