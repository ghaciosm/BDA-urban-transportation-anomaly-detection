from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = Path("data/cloud_results/merged_spark_zscore_results.csv")
OUTPUT_DIR = Path("outputs/figures_cloud")
TARGET_LINE = "YENIKAPI - HACIOSMAN"


def load_data():
    df = pd.read_csv(INPUT_FILE)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


def plot_hourly_average(df: pd.DataFrame):
    hourly_avg = df.groupby("hour")["total_passengers"].mean().sort_index()

    plt.figure(figsize=(10, 5))
    plt.plot(hourly_avg.index, hourly_avg.values)
    plt.title("Average Passenger Count by Hour (Cloud Results)")
    plt.xlabel("Hour")
    plt.ylabel("Average Passengers")
    plt.xticks(range(0, 24))
    plt.tight_layout()

    out_path = OUTPUT_DIR / "cloud_hourly_average.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_line_anomalies(df: pd.DataFrame):
    line_df = df[df["line"] == TARGET_LINE].sort_values("datetime")

    if line_df.empty:
        print(f"No rows found for line: {TARGET_LINE}")
        return

    anomalies = line_df[line_df["is_anomaly"] == True]

    plt.figure(figsize=(15, 6))
    plt.plot(line_df["datetime"], line_df["total_passengers"])
    plt.scatter(anomalies["datetime"], anomalies["total_passengers"])
    plt.title(f"Passenger Flow with Z-score Anomalies ({TARGET_LINE}) - Cloud")
    plt.xlabel("Time")
    plt.ylabel("Total Passengers")
    plt.tight_layout()

    out_path = OUTPUT_DIR / "cloud_line_anomalies.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_top_anomaly_lines(df: pd.DataFrame):
    top_lines = (
        df[df["is_anomaly"] == True]
        .groupby("line")
        .size()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(12, 6))
    top_lines.plot(kind="bar")
    plt.title("Top 10 Lines with Most Z-score Anomalies (Cloud)")
    plt.xlabel("Line")
    plt.ylabel("Anomaly Count")
    plt.tight_layout()

    out_path = OUTPUT_DIR / "cloud_top_anomaly_lines.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def save_summary(df: pd.DataFrame):
    total_rows = len(df)
    total_anomalies = int(df["is_anomaly"].sum())
    anomaly_ratio = total_anomalies / total_rows if total_rows else 0

    out_path = OUTPUT_DIR / "cloud_summary.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Total rows: {total_rows}\n")
        f.write(f"Total anomalies: {total_anomalies}\n")
        f.write(f"Anomaly ratio: {anomaly_ratio:.6f}\n")

    print(f"Saved: {out_path}")
    print(f"Total rows: {total_rows}")
    print(f"Total anomalies: {total_anomalies}")
    print(f"Anomaly ratio: {anomaly_ratio:.6f}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    print("Loaded shape:", df.shape)

    plot_hourly_average(df)
    plot_line_anomalies(df)
    plot_top_anomaly_lines(df)
    save_summary(df)


if __name__ == "__main__":
    main()