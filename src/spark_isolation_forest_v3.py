import math
import random
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType


BUCKET = "gulcin-bigdata-midterm-2026"

INPUT_GLOB = f"gs://{BUCKET}/results/spark_zscore_results/*.csv"

OUTPUT_PATH = f"gs://{BUCKET}/results/spark_isolation_forest_v3_results"
SUMMARY_PATH = f"gs://{BUCKET}/results/spark_isolation_forest_v3_summary"
MONTHLY_PATH = f"gs://{BUCKET}/results/spark_method_monthly_comparison_v3"
LINE_PATH = f"gs://{BUCKET}/results/spark_method_line_comparison_v3"

# Same anomaly ratio as Z-score baseline.
CONTAMINATION = 0.011448

NUM_TREES = 50
MAX_SAMPLES = 256
TRAIN_POOL_SIZE = 50000
RANDOM_SEED = 42

EULER_GAMMA = 0.5772156649


def create_spark_session():
    spark = (
        SparkSession.builder
        .appName("ContextualIsolationForestV3UrbanTransportation")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def c_factor(n):
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (math.log(n - 1) + EULER_GAMMA) - (2.0 * (n - 1) / n)


def build_tree(rows, depth, max_depth, num_features, rng):
    n = len(rows)

    if depth >= max_depth or n <= 1:
        return {"type": "leaf", "size": n}

    feature_idx = rng.randrange(num_features)
    values = [r[feature_idx] for r in rows if r[feature_idx] is not None]

    if not values:
        return {"type": "leaf", "size": n}

    min_val = min(values)
    max_val = max(values)

    if min_val == max_val:
        return {"type": "leaf", "size": n}

    split = rng.uniform(min_val, max_val)

    left = [r for r in rows if r[feature_idx] < split]
    right = [r for r in rows if r[feature_idx] >= split]

    if len(left) == 0 or len(right) == 0:
        return {"type": "leaf", "size": n}

    return {
        "type": "node",
        "feature": feature_idx,
        "split": split,
        "left": build_tree(left, depth + 1, max_depth, num_features, rng),
        "right": build_tree(right, depth + 1, max_depth, num_features, rng),
    }


def path_length(row, tree, depth=0):
    if tree["type"] == "leaf":
        return depth + c_factor(tree["size"])

    feature_idx = tree["feature"]
    split = tree["split"]

    if row[feature_idx] < split:
        return path_length(row, tree["left"], depth + 1)
    return path_length(row, tree["right"], depth + 1)


def train_isolation_forest(train_rows, num_features):
    rng = random.Random(RANDOM_SEED)
    max_depth = int(math.ceil(math.log(MAX_SAMPLES, 2)))
    trees = []

    for _ in range(NUM_TREES):
        if len(train_rows) > MAX_SAMPLES:
            sample = rng.sample(train_rows, MAX_SAMPLES)
        else:
            sample = train_rows

        trees.append(build_tree(sample, 0, max_depth, num_features, rng))

    return trees


def normalize_zscore_label(df):
    return df.withColumn(
        "zscore_anomaly",
        F.when(
            (F.lower(F.col("is_anomaly").cast("string")) == "true")
            | (F.col("is_anomaly").cast("int") == 1),
            F.lit(1),
        )
        .otherwise(F.lit(0))
        .cast(IntegerType()),
    )


def main():
    spark = create_spark_session()

    print("Reading input:", INPUT_GLOB)

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(INPUT_GLOB)
    )

    print("Loaded rows:", df.count())
    print("Columns:", df.columns)

    numeric_cols = [
        "total_passengers",
        "total_passages",
        "record_count",
        "hour",
        "month",
        "spark_dayofweek",
        "is_weekend",
        "group_mean",
        "group_std",
        "z_score",
    ]

    for c in numeric_cols:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast(DoubleType()))

    df = normalize_zscore_label(df)

    df = df.filter(
        F.col("total_passengers").isNotNull()
        & F.col("total_passages").isNotNull()
        & F.col("group_mean").isNotNull()
        & F.col("group_std").isNotNull()
        & F.col("z_score").isNotNull()
    )

    # V3: Context-normalized deviation features only.
    # No line_index, no town_index, no raw route identity.
    # The model is now aligned with the same problem as Z-score:
    # "How unusual is this passenger count relative to its expected context?"
    df = (
        df
        .withColumn("abs_z_score", F.abs(F.col("z_score")))
        .withColumn(
            "relative_to_group_mean",
            (F.col("total_passengers") - F.col("group_mean")) / (F.col("group_mean") + F.lit(1.0)),
        )
        .withColumn(
            "abs_relative_to_group_mean",
            F.abs(F.col("relative_to_group_mean")),
        )
        .withColumn(
            "log_observed_expected_ratio",
            F.log1p(F.col("total_passengers")) - F.log1p(F.col("group_mean")),
        )
        .withColumn(
            "abs_log_observed_expected_ratio",
            F.abs(F.col("log_observed_expected_ratio")),
        )
        .withColumn(
            "passage_per_passenger",
            F.col("total_passages") / (F.col("total_passengers") + F.lit(1.0)),
        )
    )

    feature_cols = [
        "z_score",
        "abs_z_score",
        "relative_to_group_mean",
        "abs_relative_to_group_mean",
        "log_observed_expected_ratio",
        "abs_log_observed_expected_ratio",
        "passage_per_passenger",
    ]

    df = df.fillna(0, subset=feature_cols)
    df = df.cache()

    total_rows = df.count()
    print("Rows after filtering:", total_rows)

    print("Feature columns used in V3:")
    for c in feature_cols:
        print("-", c)

    sample_fraction = min(1.0, max(TRAIN_POOL_SIZE / float(total_rows), 0.001))

    train_rows_raw = (
        df.select(*feature_cols)
        .sample(False, sample_fraction, seed=RANDOM_SEED)
        .limit(TRAIN_POOL_SIZE)
        .collect()
    )

    train_rows = [
        tuple(float(x) if x is not None else 0.0 for x in row)
        for row in train_rows_raw
    ]

    print("Training pool size:", len(train_rows))

    if len(train_rows) < 100:
        raise RuntimeError("Training sample is too small for Isolation Forest V3.")

    trees = train_isolation_forest(train_rows, len(feature_cols))
    bc_trees = spark.sparkContext.broadcast(trees)
    normalizer = c_factor(MAX_SAMPLES)

    def score_row(*vals):
        row = tuple(float(v) if v is not None else 0.0 for v in vals)
        trees_local = bc_trees.value

        total_path = 0.0
        for tree in trees_local:
            total_path += path_length(row, tree, 0)

        avg_path = total_path / float(len(trees_local))

        if normalizer == 0:
            return 0.0

        # Higher score means more anomalous.
        return float(2.0 ** (-avg_path / normalizer))

    score_udf = F.udf(score_row, DoubleType())

    scored = df.withColumn(
        "iforest_v3_score",
        score_udf(*[F.col(c) for c in feature_cols])
    )

    threshold = scored.stat.approxQuantile(
        "iforest_v3_score",
        [1.0 - CONTAMINATION],
        0.001
    )[0]

    print("Isolation Forest V3 score threshold:", threshold)

    scored = (
        scored
        .withColumn(
            "iforest_v3_anomaly",
            F.when(F.col("iforest_v3_score") >= F.lit(threshold), F.lit(1))
            .otherwise(F.lit(0))
            .cast(IntegerType())
        )
        .withColumn("iforest_v3_prediction", F.col("iforest_v3_anomaly"))
    )

    selected_cols = [
        "datetime",
        "line",
        "line_name",
        "town",
        "total_passengers",
        "total_passages",
        "record_count",
        "hour",
        "month",
        "spark_dayofweek",
        "is_weekend",
        "group_mean",
        "group_std",
        "z_score",
        "zscore_anomaly",
        "abs_z_score",
        "relative_to_group_mean",
        "abs_relative_to_group_mean",
        "log_observed_expected_ratio",
        "abs_log_observed_expected_ratio",
        "passage_per_passenger",
        "iforest_v3_score",
        "iforest_v3_prediction",
        "iforest_v3_anomaly",
    ]

    scored_out = scored.select(*[c for c in selected_cols if c in scored.columns])

    print("Top Isolation Forest V3 anomalies:")
    scored_out.orderBy(F.col("iforest_v3_score").desc()).show(20, truncate=False)

    print("Saving full V3 results:", OUTPUT_PATH)

    (
        scored_out.write
        .mode("overwrite")
        .option("header", True)
        .csv(OUTPUT_PATH)
    )

    summary = scored_out.agg(
        F.count("*").alias("total_rows"),
        F.sum("zscore_anomaly").alias("zscore_anomaly_count"),
        F.sum("iforest_v3_anomaly").alias("iforest_v3_anomaly_count"),
        F.sum(
            F.when(
                (F.col("zscore_anomaly") == 1) & (F.col("iforest_v3_anomaly") == 1),
                1
            ).otherwise(0)
        ).alias("both_methods_anomaly_count"),
        F.sum(
            F.when(
                (F.col("zscore_anomaly") == 1) & (F.col("iforest_v3_anomaly") == 0),
                1
            ).otherwise(0)
        ).alias("only_zscore_anomaly_count"),
        F.sum(
            F.when(
                (F.col("zscore_anomaly") == 0) & (F.col("iforest_v3_anomaly") == 1),
                1
            ).otherwise(0)
        ).alias("only_iforest_v3_anomaly_count"),
        F.avg("iforest_v3_score").alias("avg_iforest_v3_score"),
        F.max("iforest_v3_score").alias("max_iforest_v3_score"),
        F.min("iforest_v3_score").alias("min_iforest_v3_score"),
    )

    summary = (
        summary
        .withColumn("zscore_anomaly_rate", F.col("zscore_anomaly_count") / F.col("total_rows"))
        .withColumn("iforest_v3_anomaly_rate", F.col("iforest_v3_anomaly_count") / F.col("total_rows"))
        .withColumn("method_overlap_rate", F.col("both_methods_anomaly_count") / F.col("total_rows"))
    )

    summary.show(truncate=False)

    (
        summary.coalesce(1)
        .write
        .mode("overwrite")
        .option("header", True)
        .csv(SUMMARY_PATH)
    )

    monthly = (
        scored_out.groupBy("month")
        .agg(
            F.count("*").alias("total_obs"),
            F.sum("zscore_anomaly").alias("zscore_anomaly_count"),
            F.sum("iforest_v3_anomaly").alias("iforest_v3_anomaly_count"),
            F.avg("iforest_v3_score").alias("avg_iforest_v3_score"),
        )
        .withColumn("zscore_anomaly_rate", F.col("zscore_anomaly_count") / F.col("total_obs"))
        .withColumn("iforest_v3_anomaly_rate", F.col("iforest_v3_anomaly_count") / F.col("total_obs"))
        .orderBy("month")
    )

    monthly.show(20, truncate=False)

    (
        monthly.coalesce(1)
        .write
        .mode("overwrite")
        .option("header", True)
        .csv(MONTHLY_PATH)
    )

    line_comparison = (
        scored_out.groupBy("line")
        .agg(
            F.count("*").alias("total_obs"),
            F.sum("zscore_anomaly").alias("zscore_anomaly_count"),
            F.sum("iforest_v3_anomaly").alias("iforest_v3_anomaly_count"),
            F.avg("iforest_v3_score").alias("avg_iforest_v3_score"),
        )
        .withColumn("zscore_anomaly_rate", F.col("zscore_anomaly_count") / F.col("total_obs"))
        .withColumn("iforest_v3_anomaly_rate", F.col("iforest_v3_anomaly_count") / F.col("total_obs"))
        .orderBy(F.col("iforest_v3_anomaly_rate").desc())
    )

    line_comparison.show(20, truncate=False)

    (
        line_comparison.coalesce(1)
        .write
        .mode("overwrite")
        .option("header", True)
        .csv(LINE_PATH)
    )

    print("Done.")
    spark.stop()


if __name__ == "__main__":
    main()