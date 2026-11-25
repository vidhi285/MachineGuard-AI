"""
Read compressor_engineered.csv in chunks to locate full-file CSV read issues.
This script does not modify the source file.
"""
from pathlib import Path
import pandas as pd

path = Path(__file__).resolve().parent / "compressor_engineered.csv"
print("Testing:", path)
print("File size:", f"{path.stat().st_size:,}", "bytes")

for engine in ("c", "python"):
    print(f"\nTesting engine={engine!r} in chunks...")
    total_rows = 0
    try:
        for chunk_number, chunk in enumerate(
            pd.read_csv(path, engine=engine, chunksize=100_000), start=1
        ):
            total_rows += len(chunk)
            print(f"  Chunk {chunk_number}: {len(chunk):,} rows; total {total_rows:,}")
        print(f"SUCCESS: {engine!r} engine read {total_rows:,} rows.")
    except Exception as exc:
        print(f"FAILED using {engine!r} engine after {total_rows:,} rows.")
        print(f"{type(exc).__name__}: {exc}")
