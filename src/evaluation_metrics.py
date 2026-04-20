from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = Path("data/cloud_results/merged_spark_zscore_results.csv")
OUTPUT_DIR = Path("outputs/evaluation")


def load_data():
    df = pd.read_csv(INPUT_FILE)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


def top_line_anomaly_rate(df: pd.DataFrame):
    line_stats = (
        df.groupby("line")
        .agg(
            total_obs=("is_anomaly", "count"),
            anomaly_count=("is_anomaly", "sum")
        )
        .reset_index()
    )

    line_stats["anomaly_rate"] = line_stats["anomaly_count"] / line_stats["total_obs"]
    top5 = line_stats.sort_values("anomaly_rate", ascending=False).head(5)

    top5.to_csv(OUTPUT_DIR / "top5_line_anomaly_rate.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(top5["line"], top5["anomaly_rate"] * 100)
    plt.title("Top 5 Lines by Anomaly Rate (%)")
    plt.xlabel("Line")
    plt.ylabel("Anomaly Rate (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top5_line_anomaly_rate.png", dpi=200)
    plt.close()

    print(top5)


def anomaly_distribution_by_month(df: pd.DataFrame):
    monthly = (
        df.groupby("month")
        .agg(
            total_obs=("is_anomaly", "count"),
            anomaly_count=("is_anomaly", "sum")
        )
        .reset_index()
    )

    monthly["anomaly_rate"] = monthly["anomaly_count"] / monthly["total_obs"]
    monthly.to_csv(OUTPUT_DIR / "anomaly_distribution_by_month.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(monthly["month"], monthly["anomaly_count"], marker="o")
    plt.title("Anomaly Distribution by Month")
    plt.xlabel("Month")
    plt.ylabel("Number of Anomalies")
    plt.xticks(monthly["month"])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "anomaly_distribution_by_month.png", dpi=200)
    plt.close()

    print(monthly)


def threshold_sensitivity(df: pd.DataFrame):
    thresholds = [2.0, 2.5, 3.0]
    rows = []

    z = pd.to_numeric(df["z_score"], errors="coerce")

    for t in thresholds:
        anomaly_count = (z.abs() > t).sum()
        anomaly_rate = anomaly_count / len(df)
        rows.append({
            "threshold": t,
            "anomaly_count": int(anomaly_count),
            "anomaly_rate": anomaly_rate
        })

    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(OUTPUT_DIR / "threshold_sensitivity.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(sensitivity["threshold"], sensitivity["anomaly_count"], marker="o")
    plt.title("Z-score Threshold Sensitivity")
    plt.xlabel("Threshold")
    plt.ylabel("Number of Anomalies")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "threshold_sensitivity.png", dpi=200)
    plt.close()

    print(sensitivity)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    top_line_anomaly_rate(df)
    anomaly_distribution_by_month(df)
    threshold_sensitivity(df)


if __name__ == "__main__":
    main()