from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, stddev, abs, expr

INPUT_GLOB = "gs://gulcin-bigdata-midterm-2026/processed/spark_hourly_aggregated/*.csv"
OUTPUT_PATH = "gs://gulcin-bigdata-midterm-2026/results/spark_zscore_results"


def create_spark_session():
    spark = (
        SparkSession.builder
        .appName("SparkZScore")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def main():
    spark = create_spark_session()

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(INPUT_GLOB)
    )

    print("Loaded rows:", df.count())
    print("Columns:", df.columns)

    group_cols = ["line", "hour", "spark_dayofweek"]

    stats_df = (
        df.groupBy(group_cols)
        .agg(
            mean("total_passengers").alias("group_mean"),
            stddev("total_passengers").alias("group_std")
        )
    )

    df = df.join(stats_df, on=group_cols, how="left")

    df = df.withColumn(
        "z_score",
        expr("try_divide(total_passengers - group_mean, group_std)")
    )

    df = df.withColumn(
        "is_anomaly",
        (abs(col("z_score")) > 3) & col("z_score").isNotNull()
    )

    anomaly_df = df.filter(col("is_anomaly") == True)

    print("\nTop anomalies:")
    anomaly_df.orderBy(abs(col("z_score")).desc()).show(20, truncate=False)

    (
        df.write
        .mode("overwrite")
        .option("header", True)
        .csv(OUTPUT_PATH)
    )

    print(f"Saved to: {OUTPUT_PATH}")
    print("Total anomalies:", anomaly_df.count())

    spark.stop()


if __name__ == "__main__":
    main()