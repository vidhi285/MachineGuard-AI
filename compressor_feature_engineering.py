"""
compressor_feature_engineering.py

Final feature engineering for the MetroPT-3 Air Compressor dataset.

This program:
- Reads the original MetroPT-3 CSV without modifying it.
- Validates timestamps, missing values, duplicates, and failure intervals.
- Sorts records chronologically before creating difference features.
- Preserves all records in compressor_engineered.csv.
- Removes the non-sensor index column Unnamed: 0, if present.
- Creates plots in compressor_feature_plots.
- Does NOT train or evaluate any machine-learning model.
"""

from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use("Agg")  # Save plots without requiring an interactive display.
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "MetroPT3(AirCompressor).csv"
OUTPUT_FILE = BASE_DIR / "compressor_engineered.csv"
PLOT_DIR = BASE_DIR / "compressor_feature_plots"

EXPECTED_ROW_COUNT = 1_516_948

# Documented MetroPT-3 air-leak failure intervals (inclusive endpoints).
FAILURE_INTERVALS = [
    {
        "name": "Failure 1 - Air leak",
        "start": "2020-04-18 00:00:00",
        "end": "2020-04-18 23:59:00",
    },
    {
        "name": "Failure 2 - Air leak",
        "start": "2020-05-29 23:30:00",
        "end": "2020-05-30 06:00:00",
    },
    {
        "name": "Failure 3 - Air leak",
        "start": "2020-06-05 10:00:00",
        "end": "2020-06-07 14:30:00",
    },
    {
        "name": "Failure 4 - Air leak",
        "start": "2020-07-15 14:30:00",
        "end": "2020-07-15 19:00:00",
    },
]

ENGINEERED_FEATURES = [
    "Pressure_Difference_TP3_TP2",
    "Pressure_Difference_Reservoir_TP2",
    "Motor_Current_Change",
    "Oil_Temperature_Change",
]

SENSOR_COLUMNS_FOR_PLOTS = [
    "TP2",
    "TP3",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
]


def stop(message):
    """Show a helpful error and exit without silently continuing."""
    print(f"\nERROR: {message}")
    sys.exit(1)


def parse_timestamps(series):
    """
    Parse timestamps while preserving every input row.

    The dataset's observed format is ISO-like, for example
    '2020-02-01 00:00:00'. Mixed parsing is used as a fallback for
    pandas versions that support it.
    """
    try:
        parsed = pd.to_datetime(series, format="mixed", errors="coerce")
    except (TypeError, ValueError):
        # Compatibility fallback for older pandas releases.
        parsed = pd.to_datetime(series, errors="coerce")

    return parsed


def create_feature_plots(df):
    """Create requested time-series, target, event-window, and distribution plots."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # A large dataset can make very dense plots unreadable. Plot the median
    # within each minute for overview charts; the CSV still keeps every row.
    plot_data = df.set_index("timestamp")
    numeric_sensors = [c for c in SENSOR_COLUMNS_FOR_PLOTS if c in df.columns]
    if numeric_sensors:
        overview = plot_data[numeric_sensors].resample("1min").median()

        # Pressure sensors.
        pressure_columns = [c for c in ["TP2", "TP3", "Reservoirs"] if c in overview.columns]
        if pressure_columns:
            ax = overview[pressure_columns].plot(figsize=(14, 6), linewidth=0.8)
            ax.set_title("MetroPT-3 Pressure Sensor Readings Over Time (1-minute median)")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Sensor reading")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(PLOT_DIR / "pressure_sensors_over_time.png", dpi=150)
            plt.close()

        # Oil temperature.
        if "Oil_temperature" in overview.columns:
            ax = overview[["Oil_temperature"]].plot(figsize=(14, 5), linewidth=0.8)
            ax.set_title("Oil Temperature Over Time (1-minute median)")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Oil temperature")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(PLOT_DIR / "oil_temperature_over_time.png", dpi=150)
            plt.close()

        # Motor current.
        if "Motor_current" in overview.columns:
            ax = overview[["Motor_current"]].plot(figsize=(14, 5), linewidth=0.8)
            ax.set_title("Motor Current Over Time (1-minute median)")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Motor current")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(PLOT_DIR / "motor_current_over_time.png", dpi=150)
            plt.close()

    # Failure_Status distribution.
    if "Failure_Status" in df.columns:
        counts = df["Failure_Status"].value_counts().reindex([0, 1], fill_value=0)
        ax = counts.plot(kind="bar", figsize=(7, 5))
        ax.set_title("Failure_Status Distribution")
        ax.set_xlabel("Failure_Status (0 = outside reported intervals, 1 = inside)")
        ax.set_ylabel("Number of records")
        ax.set_xticklabels(["0 - Outside intervals", "1 - Inside intervals"], rotation=0)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "failure_status_distribution.png", dpi=150)
        plt.close()

    # Sensor readings around each documented failure. Use 3 hours of context
    # before and after each interval and plot minute medians for readability.
    for event in FAILURE_INTERVALS:
        start = pd.Timestamp(event["start"])
        end = pd.Timestamp(event["end"])
        window_start = start - pd.Timedelta(hours=3)
        window_end = end + pd.Timedelta(hours=3)
        window = df.loc[
            df["timestamp"].between(window_start, window_end, inclusive="both")
        ].copy()

        if window.empty:
            warnings.warn(f"No records available for plot: {event['name']}")
            continue

        window = window.set_index("timestamp")
        available = [c for c in SENSOR_COLUMNS_FOR_PLOTS if c in window.columns]
        if not available:
            continue
        window = window[available].resample("1min").median()

        # Separate pressure and temperature/current into different scales.
        pressure_columns = [c for c in ["TP2", "TP3", "Reservoirs"] if c in window.columns]
        other_columns = [c for c in ["Oil_temperature", "Motor_current"] if c in window.columns]

        if pressure_columns:
            ax = window[pressure_columns].plot(figsize=(12, 5), linewidth=1)
            ax.axvspan(start, end, alpha=0.2, label="Reported failure interval")
            ax.set_title(f"{event['name']} - Pressure Sensors (3-hour context)")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Sensor reading")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            safe_name = event["name"].lower().replace(" ", "_").replace("-", "")
            plt.savefig(PLOT_DIR / f"{safe_name}_pressure.png", dpi=150)
            plt.close()

        if other_columns:
            ax = window[other_columns].plot(figsize=(12, 5), linewidth=1)
            ax.axvspan(start, end, alpha=0.2, label="Reported failure interval")
            ax.set_title(f"{event['name']} - Oil Temperature and Motor Current")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Sensor reading")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            safe_name = event["name"].lower().replace(" ", "_").replace("-", "")
            plt.savefig(PLOT_DIR / f"{safe_name}_temperature_current.png", dpi=150)
            plt.close()

    # Distributions of the four engineered features.
    for feature in ENGINEERED_FEATURES:
        if feature not in df.columns:
            continue
        values = pd.to_numeric(df[feature], errors="coerce").dropna()
        if values.empty:
            warnings.warn(f"No valid values to plot for {feature}")
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values, bins=80)
        ax.set_title(f"Distribution of {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"{feature.lower()}_distribution.png", dpi=150)
        plt.close()


def main():
    print("=" * 76)
    print("METROPT-3 AIR COMPRESSOR - FINAL FEATURE ENGINEERING")
    print("=" * 76)
    print(f"Input file:  {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")

    if not INPUT_FILE.exists():
        stop(
            f"Input file was not found: {INPUT_FILE}\n"
            "Place MetroPT3(AirCompressor).csv in the same folder as this script."
        )

    # -----------------------------------------------------------------
    # 2. Load the original CSV and inspect its structure
    # -----------------------------------------------------------------
    try:
        original = pd.read_csv(INPUT_FILE)
    except Exception as exc:
        stop(f"Could not read the original CSV: {exc}")

    original_row_count = len(original)
    original_column_count = len(original.columns)

    print(f"\nOriginal row count: {original_row_count:,}")
    print(f"Expected row count: {EXPECTED_ROW_COUNT:,}")
    print(f"Original column count: {original_column_count}")
    print(f"Original columns: {list(original.columns)}")

    if original_row_count != EXPECTED_ROW_COUNT:
        stop(
            f"Expected {EXPECTED_ROW_COUNT:,} original rows but found "
            f"{original_row_count:,}. Check that the correct source CSV was used."
        )

    if "timestamp" not in original.columns:
        stop("The original CSV does not contain a 'timestamp' column.")

    # Parse timestamp values. If any are invalid, report them and stop rather
    # than silently dropping or altering rows.
    original["timestamp"] = parse_timestamps(original["timestamp"])
    invalid_timestamp_count = int(original["timestamp"].isna().sum())
    print(f"Invalid timestamps: {invalid_timestamp_count:,}")
    if invalid_timestamp_count:
        bad_examples = original.loc[original["timestamp"].isna()].head(10)
        print("\nExamples of rows with invalid timestamps:")
        print(bad_examples.to_string(index=False))
        stop(
            "Timestamp parsing failed for one or more rows. No output dataset "
            "was written; fix the timestamp values first."
        )

    # Inspect missing values before cleaning/removing any columns.
    print("\nMissing values in original columns:")
    missing_original = original.isna().sum()
    print(missing_original[missing_original > 0].to_string() if missing_original.any()
          else "No missing values found.")

    # Duplicate checks are reported, not automatically removed.
    duplicate_full_rows = int(original.duplicated().sum())
    duplicate_timestamps = int(original["timestamp"].duplicated().sum())
    print(f"\nDuplicate full rows: {duplicate_full_rows:,}")
    print(f"Duplicate timestamps: {duplicate_timestamps:,}")

    # -----------------------------------------------------------------
    # 3. Sort chronologically, remove the index column, and preserve rows
    # -----------------------------------------------------------------
    df = original.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        print("\nRemoved 'Unnamed: 0' because it appears to be a stored index column.")

    # Check required sensor columns for feature formulas.
    required_sensors = [
        "TP2",
        "TP3",
        "Reservoirs",
        "Motor_current",
        "Oil_temperature",
    ]
    missing_sensors = [c for c in required_sensors if c not in df.columns]
    if missing_sensors:
        stop(f"Cannot calculate the four engineered features; missing columns: {missing_sensors}")

    # Convert sensor columns to numeric explicitly.
    for column in required_sensors:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # -----------------------------------------------------------------
    # 4. Verify documented failure intervals and create Failure_Status
    # -----------------------------------------------------------------
    print("\nFailure interval validation (inclusive endpoints):")
    df["Failure_Status"] = 0
    interval_counts = []

    for event in FAILURE_INTERVALS:
        start = pd.Timestamp(event["start"])
        end = pd.Timestamp(event["end"])
        in_interval = df["timestamp"].between(start, end, inclusive="both")
        count = int(in_interval.sum())

        print(f"\n{event['name']}")
        print(f"  Start: {start}")
        print(f"  End:   {end}")
        print(f"  Rows in interval: {count:,}")

        if df["timestamp"].min() <= end and df["timestamp"].max() >= start and count == 0:
            stop(
                f"{event['name']} overlaps the dataset's date range but matched "
                "zero records. Check timestamp parsing and interval definitions."
            )

        df.loc[in_interval, "Failure_Status"] = 1
        interval_counts.append((event["name"], count))

    # -----------------------------------------------------------------
    # 5. Create the four engineered features
    # -----------------------------------------------------------------
    # Pressure_Difference_TP3_TP2:
    # Difference between TP3 and TP2 pressure measurements at the same time.
    df["Pressure_Difference_TP3_TP2"] = df["TP3"] - df["TP2"]

    # Pressure_Difference_Reservoir_TP2:
    # Reservoir pressure minus TP2 pressure at the same timestamp.
    df["Pressure_Difference_Reservoir_TP2"] = df["Reservoirs"] - df["TP2"]

    # Motor_Current_Change:
    # Current reading minus the previous chronological reading. diff() uses
    # only the current and previous rows; it does not look ahead to future rows.
    df["Motor_Current_Change"] = df["Motor_current"].diff()

    # Oil_Temperature_Change:
    # Current oil temperature minus the previous chronological reading.
    df["Oil_Temperature_Change"] = df["Oil_temperature"].diff()

    # -----------------------------------------------------------------
    # 6. Report missing values after feature engineering; do not drop rows
    # -----------------------------------------------------------------
    print("\nMissing values by column in final dataset:")
    missing_final = df.isna().sum()
    print(missing_final[missing_final > 0].to_string() if missing_final.any()
          else "No missing values found.")

    print(
        "\nNote: the first chronological row naturally has a missing value "
        "for Motor_Current_Change and Oil_Temperature_Change because no prior "
        "reading exists. These values are intentionally left missing; no future "
        "sensor readings are used to fill them, and no rows are deleted."
    )

    # -----------------------------------------------------------------
    # 7. Save final engineered CSV without modifying the original
    # -----------------------------------------------------------------
    try:
        df.to_csv(OUTPUT_FILE, index=False)
    except Exception as exc:
        stop(f"Could not save the engineered dataset: {exc}")

    # Verify output was written and can be inspected.
    if not OUTPUT_FILE.exists():
        stop("The output CSV was not created.")
    final_row_count = len(df)
    final_column_count = len(df.columns)

    # -----------------------------------------------------------------
    # 8. Generate requested visualizations
    # -----------------------------------------------------------------
    try:
        create_feature_plots(df)
    except Exception as exc:
        warnings.warn(f"Dataset was saved, but one or more plots failed: {exc}")

    # -----------------------------------------------------------------
    # 9. Final summary
    # -----------------------------------------------------------------
    print("\n" + "=" * 76)
    print("FINAL SUMMARY")
    print("=" * 76)
    print(f"Original row count: {original_row_count:,}")
    print(f"Final row count:    {final_row_count:,}")
    print(f"Original column count: {original_column_count}")
    print(f"Final column count:    {final_column_count}")
    print(f"Rows removed: {original_row_count - final_row_count:,}")
    print(f"Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Invalid timestamps: {invalid_timestamp_count:,}")
    print(f"Duplicate full rows (retained): {duplicate_full_rows:,}")
    print(f"Duplicate timestamps (retained): {duplicate_timestamps:,}")

    print("\nFailure counts by documented interval:")
    for name, count in interval_counts:
        print(f"  - {name}: {count:,} rows")

    print("\nOverall Failure_Status distribution:")
    print(df["Failure_Status"].value_counts().sort_index().rename(
        index={0: "0 - Outside reported intervals", 1: "1 - Inside reported intervals"}
    ).to_string())

    print("\nEngineered features and formulas:")
    print("  1. Pressure_Difference_TP3_TP2 = TP3 - TP2")
    print("  2. Pressure_Difference_Reservoir_TP2 = Reservoirs - TP2")
    print("  3. Motor_Current_Change = current Motor_current - previous Motor_current")
    print("  4. Oil_Temperature_Change = current Oil_temperature - previous Oil_temperature")

    print("\nTarget interpretation:")
    print("  Failure_Status=1: timestamp is inside a documented failure interval.")
    print("  Failure_Status=0: timestamp is outside the documented intervals; this")
    print("                    does not guarantee the machine was healthy.")
    print("  These labels do not automatically represent early-warning predictions.")

    print("\nMissing-value summary (final dataset):")
    if missing_final.any():
        for column, count in missing_final[missing_final > 0].items():
            print(f"  - {column}: {int(count):,}")
    else:
        print("  No missing values found.")

    print(f"\nFinal dataset saved to: {OUTPUT_FILE}")
    print(f"Plots saved in: {PLOT_DIR}")
    print("No training/testing datasets were created.")
    print("No machine-learning model was trained or evaluated.")
    print("The original source CSV was not modified.")


if __name__ == "__main__":
    main()
