# BDA Urban Transportation Anomaly Detection

## Overview
This project investigates anomalous passenger demand patterns in large-scale urban transportation data using Apache Spark.

The dataset consists of hourly public transportation records from the Istanbul Metropolitan Municipality (IMM) Open Data Portal for the year 2020. The main goal is to build a scalable data processing pipeline and detect unusual passenger behavior through statistical anomaly detection.

At the current stage of the project, the baseline anomaly detection method is **Z-score**. A machine learning-based extension using **Isolation Forest** is planned for the final phase.

---

## Repository Structure

```text
.
├── src/
│   ├── dataset_summary.py
│   ├── evaluation_metrics.py
│   ├── merge_cloud_results.py
│   ├── spark_preprocessing.py
│   ├── spark_zscore.py
│   └── visualize_cloud_zscore.py
├── requirements.txt
├── .gitignore
└── README.md
```

Project Pipeline

Raw Data → Spark Preprocessing → Hourly Aggregation → Z-score Detection → Merged Results → Evaluation & Visualization

Methodology

The project follows the steps below:

1. Data Ingestion
Monthly CSV files are loaded from cloud storage.
2. Preprocessing
Data cleaning
Type conversion
Construction of a unified datetime field
Removal of invalid or incomplete records
3. Hourly Aggregation

Data is aggregated based on:

datetime
line
line_name
town
4. Feature Engineering

Temporal features:

hour
month
day of week
weekend indicator
5. Baseline Anomaly Detection
Group-wise Z-scores are computed on total_passengers
Grouping context:
line
hour
day of week
Observations with |Z| > 3 are labeled as anomalies
6. Post-processing and Analysis
Cloud result files are merged
Dataset summary tables are generated
Evaluation metrics and visualizations are produced
Source Files
spark_preprocessing.py

Runs the distributed preprocessing and aggregation pipeline on Apache Spark.

spark_zscore.py

Runs the baseline Z-score anomaly detection pipeline in Spark.

merge_cloud_results.py

Merges distributed Spark CSV outputs into a single dataset.

dataset_summary.py

Generates dataset summary statistics used in the report.

evaluation_metrics.py

Computes:

monthly anomaly distribution
threshold sensitivity
top lines by anomaly rate
visualize_cloud_zscore.py

Generates visualizations such as:

hourly averages
anomaly plots
top anomaly lines
Technologies Used
Python
PySpark
Apache Spark
Google Cloud Dataproc
Google Cloud Storage
Pandas
Matplotlib
Current Results (Midterm)
Total processed observations: ~6 million
Total detected anomalies: ~69,000
Anomaly ratio: ~1.14%
Strong temporal patterns observed
Anomalies concentrated in specific months and lines
Example Anomaly
Line: YENIKAPI - HACIOSMAN
Datetime: 2020-01-01 00:00:00
Observed passengers: 6215
Expected mean: ~822.5
Std: ~1789.3
Z-score: 3.01

This shows how unusually high demand is detected relative to normal patterns.