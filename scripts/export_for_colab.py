"""
Export hourly_rides from PostgreSQL to a parquet file for Colab training.

Usage (from project root, .venv activated):
    python scripts/export_for_colab.py

Outputs: data/hourly_rides.parquet  (~30 MB)
Upload that file to Google Drive, then open the Colab notebook.
"""

from __future__ import annotations
import os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import pandas as pd
from sqlalchemy import create_engine, text

POSTGRES_URL = "postgresql://taxi:Yaseen#321@localhost:5433/taxi_demand"
OUT_PATH     = PROJECT_ROOT / "data" / "hourly_rides.parquet"

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(POSTGRES_URL)

    print("Exporting hourly_rides from PostgreSQL...")
    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT pickup_location_id, pickup_hour, ride_count FROM hourly_rides ORDER BY pickup_location_id, pickup_hour"),
            conn,
        )

    print(f"  {len(df):,} rows | {df['pickup_location_id'].nunique()} zones | "
          f"{df['pickup_hour'].min().date()} to {df['pickup_hour'].max().date()}")

    df.to_parquet(OUT_PATH, index=False)
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"  Saved to: {OUT_PATH}  ({size_mb:.1f} MB)")
    print()
    print("Next steps:")
    print("  1. Upload data/hourly_rides.parquet to Google Drive")
    print("  2. Open notebooks/train_colab.ipynb in Google Colab")
    print("  3. Run all cells")

if __name__ == "__main__":
    main()
