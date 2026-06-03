from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_date, concat_ws, lpad, to_timestamp,
    dayofweek, month, when, sum as spark_sum, count
)

INPUT_GLOB = "gs://gulcin-bigdata-midterm-2026/raw/*.csv"
OUTPUT_PATH = "gs://gulcin-bigdata-midterm-2026/processed/spark_hourly_aggregated"


def create_spark_session():
    spark = (
        SparkSession.builder
        .appName("TransportationAnomalyDetection")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    return spark


def main():
    spark = create_spark_session()

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(INPUT_GLOB)
    )

    print("Raw row count:", df.count())
    df.printSchema()

    df = (
        df.withColumn("transition_date", to_date(col("transition_date")))
          .withColumn("transition_hour", col("transition_hour").cast("int"))
          .withColumn("number_of_passenger", col("number_of_passenger").cast("double"))
          .withColumn("number_of_passage", col("number_of_passage").cast("double"))
    )

    df = df.filter(
        col("transition_date").isNotNull() &
        col("transition_hour").isNotNull() &
        col("number_of_passenger").isNotNull() &
        (col("transition_hour") >= 0) &
        (col("transition_hour") <= 23) &
        (col("number_of_passenger") >= 0)
    )

    df = df.withColumn(
        "datetime",
        to_timestamp(
            concat_ws(
                " ",
                col("transition_date").cast("string"),
                concat_ws(":", lpad(col("transition_hour").cast("string"), 2, "0"))
            ),
            "yyyy-MM-dd HH"
        )
    )

    agg_df = (
        df.groupBy("datetime", "line", "line_name", "town")
          .agg(
              spark_sum("number_of_passenger").alias("total_passengers"),
              spark_sum("number_of_passage").alias("total_passages"),
              count("*").alias("record_count")
          )
          .withColumn("hour", col("datetime").substr(12, 2).cast("int"))
          .withColumn("month", month(col("datetime")))
          .withColumn("spark_dayofweek", dayofweek(col("datetime")))
          .withColumn(
              "is_weekend",
              when(col("spark_dayofweek").isin([1, 7]), 1).otherwise(0)
          )
    )

    print("Aggregated row count:", agg_df.count())
    agg_df.show(10, truncate=False)

    (
        agg_df.write
        .mode("overwrite")
        .option("header", True)
        .csv(OUTPUT_PATH)
    )

    print(f"Saved Spark aggregated output to: {OUTPUT_PATH}")
    spark.stop()


if __name__ == "__main__":
    main()