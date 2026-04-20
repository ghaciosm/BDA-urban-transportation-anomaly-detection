from pathlib import Path
import pandas as pd

INPUT_DIR = Path("data/cloud_results/spark_zscore_results")
OUTPUT_FILE = Path("data/cloud_results/merged_spark_zscore_results.csv")


def main():
    csv_files = sorted(INPUT_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}")

    dfs = []
    for file_path in csv_files:
        print(f"Reading: {file_path.name}")
        df = pd.read_csv(file_path)
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved merged file to: {OUTPUT_FILE}")
    print("Shape:", merged_df.shape)
    print("Columns:", merged_df.columns.tolist())


if __name__ == "__main__":
    main()