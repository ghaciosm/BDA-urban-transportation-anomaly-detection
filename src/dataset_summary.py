import pandas as pd

INPUT_FILE = "data/cloud_results/merged_spark_zscore_results.csv"
OUTPUT_FILE = "outputs/dataset_summary_table.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    summary = {
        "Property": [
            "Data source",
            "Period used",
            "Total rows",
            "Total columns",
            "Processing framework",
            "Cloud platform",
            "Anomaly count",
            "Anomaly ratio"
        ],
        "Value": [
            "IMM Open Data Portal",
            "2020",
            f"{len(df):,}",
            len(df.columns),
            "Apache Spark",
            "Google Cloud Dataproc",
            f"{int(df['is_anomaly'].sum()):,}",
            f"{df['is_anomaly'].mean():.6f}"
        ]
    }

    summary_df = pd.DataFrame(summary)

    print("\n=== DATASET SUMMARY ===\n")
    print(summary_df.to_string(index=False))

    summary_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()