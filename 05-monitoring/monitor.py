"""
Generate an Evidently HTML report comparing past vs recent predictions.

Usage:
    python monitor.py
This will:
- Load the logged predictions from data/predictions.csv
- Split them into reference (older) vs current (newer) data
- Generate a data + performance drift report
"""

import pandas as pd
from pathlib import Path
# ColumnMapping moved into legacy subpackage in newer Evidently releases
from evidently.legacy.pipeline.column_mapping import ColumnMapping
# Report is exposed directly from the top-level package (not a submodule)
from evidently import Report
# presets moved to `evidently.presets`
from evidently.presets import DataDriftPreset, RegressionPreset


LOG_PATH = Path("data/predictions.csv")
REPORT_PATH = Path("monitoring_report.html")


def main():
    print("\n📊 Starting monitoring report...\n")

    if not LOG_PATH.exists():
        raise FileNotFoundError("❌ No logged predictions found. Run simulate.py first!")

    df = pd.read_csv(LOG_PATH, parse_dates=["ts"])
    df = df.dropna(subset=["prediction", "duration"])
    print(f"✓ Loaded {len(df)} logged predictions")

    # Sort by timestamp and split into reference (older) vs current (recent)
    df = df.sort_values("ts")
    midpoint = len(df) // 2
    reference = df.iloc[:midpoint].copy()
    current = df.iloc[midpoint:].copy()

    print(f"Reference: {len(reference)}  |  Current: {len(current)}")

    # build a DataDefinition instead of the old ColumnMapping
    from evidently.core.datasets import DataDefinition, Dataset, Regression

    definition = DataDefinition(
        numerical_columns=["trip_distance"],
        categorical_columns=["PU_DO"],
        regression=[Regression(target="duration", prediction="prediction")],
    )

    # wrap the pandas dataframes so Evidently can interpret them correctly
    current_ds = Dataset.from_pandas(current, data_definition=definition)
    reference_ds = Dataset.from_pandas(reference, data_definition=definition)

    # Build report
    print("\n🧮 Generating Evidently drift report...")
    report = Report(metrics=[DataDriftPreset(), RegressionPreset()])
    snapshot = report.run(
        current_ds,
        reference_ds,
    )

    # Convert Path to str for Windows compatibility
    snapshot.save_html(str(REPORT_PATH))
    print(f"✅ Report saved: {REPORT_PATH.resolve()}")
    print("Open it in your browser to explore drift metrics.\n")


if __name__ == "__main__":
    main()
