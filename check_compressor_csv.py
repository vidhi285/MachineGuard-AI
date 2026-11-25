"""
Small diagnostic for compressor_engineered.csv.
Run this before rerunning compressor_model_training.py.
It does not modify the CSV.
"""
from pathlib import Path
import pandas as pd

path = Path(__file__).resolve().parent / "compressor_engineered.csv"

print("CSV path:", path)
if not path.exists():
    raise SystemExit("ERROR: compressor_engineered.csv was not found beside this script.")

print(f"File size: {path.stat().st_size:,} bytes")

for engine in ("c", "python"):
    try:
        sample = pd.read_csv(path, nrows=5, engine=engine)
        print(f"\nReader engine={engine!r}: first 5 rows read successfully")
        print("Columns:", sample.columns.tolist())
        print(sample.head(2).to_string(index=False))
    except Exception as exc:
        print(f"\nReader engine={engine!r} FAILED: {type(exc).__name__}: {exc}")

try:
    with path.open("rb") as f:
        first_line = f.readline(1000)
    print("\nFirst bytes:", repr(first_line))
except Exception as exc:
    print("\nCould not inspect file bytes:", exc)
