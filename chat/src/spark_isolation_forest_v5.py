import math
import random
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType


BUCKET = "gulcin-bigdata-midterm-2026"

INPUT_GLOB = f"gs://{BUCKET}/results/spark_zscore_results/*.csv"

OUTPUT_PATH = f"gs://{BUCKET}/results/spark_isolation_forest_v5_results"
SUMMARY_PATH = f"gs://{BUCKET}/results/spark_isolation_forest_v5_summary"
MONTHLY_PATH = f"gs://{BUCKET}/results/spark_method_monthly_comparison_v5"
LINE_PATH = f"gs://{BUCKET}/results/spark_method_line_comparison_v5"

# Same target anomaly ratio as the Z-score baseline.
# Used only for threshold selection, not as a feature.
CONTAMINATION = 0.011448

NUM_TREES = 50
MAX_SAMPLES = 256
TRAIN_POOL_SIZE = 50000
RANDOM_SEED = 42

MIN_CONTEXT_OBS = 5
EPS = 1e-6
EULER_GAMMA = 0.5772156649


def create_spark_session():
    spark = (
        SparkSession.builder
        .appName("IndependentContextualIsolationForestV5UrbanTransportation")
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
    """
    zscore_anomaly is used only for evaluation/comparison.
    It is never used as a feature in V5.
    """
    if "zscore_anomaly" in df.columns:
        return df.withColumn(
            "zscore_anomaly",
            F.col("zscore_anomaly").cast(IntegerType())
        )

    if "is_anomaly" in df.columns:
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

    return df.withColumn("zscore_anomaly", F.lit(0).cast(IntegerType()))


def ensure_temporal_columns(df):
    if "datetime" in df.columns:
        df = df.withColumn("datetime_ts", F.to_timestamp("datetime"))

    if "hour" not in df.columns:
        df = df.withColumn("hour", F.hour("datetime_ts"))

    if "month" not in df.columns:
        df = df.withColumn("month", F.month("datetime_ts"))

    if "spark_dayofweek" not in df.columns:
        df = df.withColumn("spark_dayofweek", F.dayofweek("datetime_ts"))

    if "is_weekend" not in df.columns:
        df = df.withColumn(
            "is_weekend",
            F.when(F.col("spark_dayofweek").isin(1, 7), F.lit(1)).otherwise(F.lit(0))
        )

    return df


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

    df = normalize_zscore_label(df)
    df = ensure_temporal_columns(df)

    numeric_cols = [
        "total_passengers",
        "total_passages",
        "record_count",
        "hour",
        "month",
        "spark_dayofweek",
        "is_weekend",
    ]

    for c in numeric_cols:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast(DoubleType()))

    df = df.filter(
        F.col("total_passengers").isNotNull()
        & F.col("total_passages").isNotNull()
        & F.col("record_count").isNotNull()
        & F.col("hour").isNotNull()
        & F.col("month").isNotNull()
        & F.col("spark_dayofweek").isNotNull()
        & F.col("is_weekend").isNotNull()
    )

    df = df.filter(
        (F.col("total_passengers") >= 0)
        & (F.col("total_passages") >= 0)
        & (F.col("record_count") > 0)
        & (F.col("hour") >= 0)
        & (F.col("hour") <= 23)
    )

    # ------------------------------------------------------------------
    # V5 FEATURE ENGINEERING
    #
    # Goal:
    # Independent but problem-aligned Isolation Forest.
    #
    # Not used as features:
    # - z_score
    # - abs_z_score
    # - is_anomaly
    # - zscore_anomaly
    # - line_index / town_index
    #
    # Used:
    # - independently recomputed leave-one-out context statistics
    # - passenger deviation from expected context
    # - ratio-based demand consistency features
    #
    # Context:
    # line + hour + spark_dayofweek
    # ------------------------------------------------------------------

    df = (
        df
        .withColumn(
            "passage_per_passenger",
            F.col("total_passages") / (F.col("total_passengers") + F.lit(1.0))
        )
        .withColumn(
            "passenger_per_record",
            F.col("total_passengers") / (F.col("record_count") + F.lit(1.0))
        )
        .withColumn(
            "passage_per_record",
            F.col("total_passages") / (F.col("record_count") + F.lit(1.0))
        )
    )

    context_keys = ["line", "hour", "spark_dayofweek"]

    context_stats = (
        df.groupBy(*context_keys)
        .agg(
            F.count("*").alias("context_obs_count"),
            F.sum("total_passengers").alias("context_sum_passengers"),
            F.sum(F.col("total_passengers") * F.col("total_passengers")).alias("context_sumsq_passengers"),
            F.avg("passage_per_passenger").alias("context_avg_passage_per_passenger"),
            F.avg("passenger_per_record").alias("context_avg_passenger_per_record"),
            F.avg("passage_per_record").alias("context_avg_passage_per_record"),
        )
    )

    df = df.join(context_stats, on=context_keys, how="left")
    df = df.filter(F.col("context_obs_count") >= F.lit(MIN_CONTEXT_OBS))

    # Leave-one-out context mean/std for passenger count.
    # This prevents the current observation from fully defining its own expected context.
    df = (
        df
        .withColumn("loo_count", F.col("context_obs_count") - F.lit(1.0))
        .withColumn("loo_sum_passengers", F.col("context_sum_passengers") - F.col("total_passengers"))
        .withColumn(
            "loo_sumsq_passengers",
            F.col("context_sumsq_passengers") - (F.col("total_passengers") * F.col("total_passengers"))
        )
        .withColumn(
            "loo_mean_passengers",
            F.col("loo_sum_passengers") / F.col("loo_count")
        )
        .withColumn(
            "loo_var_passengers_raw",
            (
                F.col("loo_sumsq_passengers")
                - (F.col("loo_sum_passengers") * F.col("loo_sum_passengers") / F.col("loo_count"))
            ) / F.greatest(F.col("loo_count") - F.lit(1.0), F.lit(1.0))
        )
        .withColumn(
            "loo_var_passengers",
            F.when(F.col("loo_var_passengers_raw") < 0, F.lit(0.0))
            .otherwise(F.col("loo_var_passengers_raw"))
        )
        .withColumn(
            "loo_std_passengers",
            F.sqrt(F.col("loo_var_passengers") + F.lit(EPS))
        )
    )

    # Independent context-normalized deviation features.
    df = (
        df
        .withColumn(
            "independent_context_residual",
            (F.col("total_passengers") - F.col("loo_mean_passengers")) / (F.col("loo_std_passengers") + F.lit(EPS))
        )
        .withColumn(
            "abs_independent_context_residual",
            F.abs(F.col("independent_context_residual"))
        )
        .withColumn(
            "relative_to_context_mean",
            (F.col("total_passengers") - F.col("loo_mean_passengers")) / (F.col("loo_mean_passengers") + F.lit(1.0))
        )
        .withColumn(
            "abs_relative_to_context_mean",
            F.abs(F.col("relative_to_context_mean"))
        )
        .withColumn(
            "log_observed_expected_ratio",
            F.log1p(F.col("total_passengers")) - F.log1p(F.col("loo_mean_passengers"))
        )
        .withColumn(
            "abs_log_observed_expected_ratio",
            F.abs(F.col("log_observed_expected_ratio"))
        )
        .withColumn(
            "passage_per_passenger_diff",
            F.col("passage_per_passenger") - F.col("context_avg_passage_per_passenger")
        )
        .withColumn(
            "abs_passage_per_passenger_diff",
            F.abs(F.col("passage_per_passenger_diff"))
        )
        .withColumn(
            "passenger_per_record_diff",
            F.col("passenger_per_record") - F.col("context_avg_passenger_per_record")
        )
        .withColumn(
            "abs_passenger_per_record_diff",
            F.abs(F.col("passenger_per_record_diff"))
        )
    )

    feature_cols = [
        "independent_context_residual",
        "abs_independent_context_residual",
        "relative_to_context_mean",
        "abs_relative_to_context_mean",
        "log_observed_expected_ratio",
        "abs_log_observed_expected_ratio",
        "passage_per_passenger_diff",
        "abs_passage_per_passenger_diff",
        "passenger_per_record_diff",
        "abs_passenger_per_record_diff",
    ]

    df = df.fillna(0, subset=feature_cols)

    # Standardize features before custom Isolation Forest.
    stats_exprs = []
    for c in feature_cols:
        stats_exprs.append(F.avg(c).alias(f"{c}_mean"))
        stats_exprs.append(F.stddev(c).alias(f"{c}_std"))

    stats_row = df.agg(*stats_exprs).collect()[0].asDict()

    scaled_feature_cols = []

    for c in feature_cols:
        mean_val = stats_row.get(f"{c}_mean")
        std_val = stats_row.get(f"{c}_std")

        if mean_val is None:
            mean_val = 0.0

        if std_val is None or std_val == 0:
            std_val = 1.0

        scaled_col = f"{c}_scaled"
        df = df.withColumn(
            scaled_col,
            (F.col(c) - F.lit(float(mean_val))) / F.lit(float(std_val))
        )
        scaled_feature_cols.append(scaled_col)

    df = df.fillna(0, subset=scaled_feature_cols)
    df = df.cache()

    total_rows = df.count()
    print("Rows after V5 filtering:", total_rows)

    print("Feature columns used in V5:")
    for c in feature_cols:
        print("-", c)

    print("Scaled feature columns used in model:")
    for c in scaled_feature_cols:
        print("-", c)

    sample_fraction = min(1.0, max(TRAIN_POOL_SIZE / float(total_rows), 0.001))

    train_rows_raw = (
        df.select(*scaled_feature_cols)
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
        raise RuntimeError("Training sample is too small for Isolation Forest V5.")

    trees = train_isolation_forest(train_rows, len(scaled_feature_cols))
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
        "iforest_v5_score",
        score_udf(*[F.col(c) for c in scaled_feature_cols])
    )

    threshold = scored.stat.approxQuantile(
        "iforest_v5_score",
        [1.0 - CONTAMINATION],
        0.001
    )[0]

    print("Isolation Forest V5 score threshold:", threshold)

    scored = (
        scored
        .withColumn(
            "iforest_v5_anomaly",
            F.when(F.col("iforest_v5_score") >= F.lit(threshold), F.lit(1))
            .otherwise(F.lit(0))
            .cast(IntegerType())
        )
        .withColumn("iforest_v5_prediction", F.col("iforest_v5_anomaly"))
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
        "zscore_anomaly",

        "context_obs_count",
        "loo_mean_passengers",
        "loo_std_passengers",
        "independent_context_residual",
        "abs_independent_context_residual",
        "relative_to_context_mean",
        "abs_relative_to_context_mean",
        "log_observed_expected_ratio",
        "abs_log_observed_expected_ratio",
        "passage_per_passenger_diff",
        "abs_passage_per_passenger_diff",
        "passenger_per_record_diff",
        "abs_passenger_per_record_diff",

        "iforest_v5_score",
        "iforest_v5_prediction",
        "iforest_v5_anomaly",
    ]

    scored_out = scored.select(*[c for c in selected_cols if c in scored.columns])

    print("Top Isolation Forest V5 anomalies:")
    scored_out.orderBy(F.col("iforest_v5_score").desc()).show(20, truncate=False)

    print("Saving full V5 results:", OUTPUT_PATH)

    (
        scored_out.write
        .mode("overwrite")
        .option("header", True)
        .csv(OUTPUT_PATH)
    )

    summary = scored_out.agg(
        F.count("*").alias("total_rows"),
        F.sum("zscore_anomaly").alias("zscore_anomaly_count"),
        F.sum("iforest_v5_anomaly").alias("iforest_v5_anomaly_count"),
        F.sum(
            F.when(
                (F.col("zscore_anomaly") == 1) & (F.col("iforest_v5_anomaly") == 1),
                1
            ).otherwise(0)
        ).alias("both_methods_anomaly_count"),
        F.sum(
            F.when(
                (F.col("zscore_anomaly") == 1) & (F.col("iforest_v5_anomaly") == 0),
                1
            ).otherwise(0)
        ).alias("only_zscore_anomaly_count"),
        F.sum(
            F.when(
                (F.col("zscore_anomaly") == 0) & (F.col("iforest_v5_anomaly") == 1),
                1
            ).otherwise(0)
        ).alias("only_iforest_v5_anomaly_count"),
        F.avg("iforest_v5_score").alias("avg_iforest_v5_score"),
        F.max("iforest_v5_score").alias("max_iforest_v5_score"),
        F.min("iforest_v5_score").alias("min_iforest_v5_score"),
    )

    summary = (
        summary
        .withColumn("zscore_anomaly_rate", F.col("zscore_anomaly_count") / F.col("total_rows"))
        .withColumn("iforest_v5_anomaly_rate", F.col("iforest_v5_anomaly_count") / F.col("total_rows"))
        .withColumn("method_overlap_rate", F.col("both_methods_anomaly_count") / F.col("total_rows"))
        .withColumn(
            "overlap_among_zscore_anomalies",
            F.col("both_methods_anomaly_count") / F.col("zscore_anomaly_count")
        )
        .withColumn(
            "overlap_among_iforest_v5_anomalies",
            F.col("both_methods_anomaly_count") / F.col("iforest_v5_anomaly_count")
        )
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
            F.sum("iforest_v5_anomaly").alias("iforest_v5_anomaly_count"),
            F.avg("iforest_v5_score").alias("avg_iforest_v5_score"),
        )
        .withColumn("zscore_anomaly_rate", F.col("zscore_anomaly_count") / F.col("total_obs"))
        .withColumn("iforest_v5_anomaly_rate", F.col("iforest_v5_anomaly_count") / F.col("total_obs"))
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
            F.sum("iforest_v5_anomaly").alias("iforest_v5_anomaly_count"),
            F.avg("iforest_v5_score").alias("avg_iforest_v5_score"),
        )
        .withColumn("zscore_anomaly_rate", F.col("zscore_anomaly_count") / F.col("total_obs"))
        .withColumn("iforest_v5_anomaly_rate", F.col("iforest_v5_anomaly_count") / F.col("total_obs"))
        .orderBy(F.col("iforest_v5_anomaly_rate").desc())
    )

    line_comparison.show(30, truncate=False)

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