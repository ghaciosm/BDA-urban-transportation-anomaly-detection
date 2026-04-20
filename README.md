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

Project Pipeline

Raw Data → Spark Preprocessing → Hourly Aggregation → Z-score Detection → Merged Results → Evaluation & Visualization

Methodology

The project follows the steps below:

Data Ingestion
Monthly CSV files are loaded from cloud storage.
Preprocessing
Data cleaning
Type conversion
Construction of a unified datetime field
Removal of invalid or incomplete records
Hourly Aggregation
Data is aggregated based on:
datetime
line
line_name
town
Feature Engineering
Temporal features are extracted, including:
hour
month
day of week
weekend indicator
Baseline Anomaly Detection
Group-wise Z-scores are computed on total_passengers
Grouping context:
line
hour
day of week
Observations with |Z| > 3 are labeled as anomalies
Post-processing and Analysis
Cloud result files are merged
Dataset summary tables are generated
Evaluation metrics and visualizations are produced
Source Files
spark_preprocessing.py

Runs the distributed preprocessing and aggregation pipeline on Apache Spark. It reads raw CSV files, cleans the data, constructs the datetime field, aggregates hourly observations, and extracts temporal features.

spark_zscore.py

Runs the baseline Z-score anomaly detection pipeline in Spark using grouped statistics.

merge_cloud_results.py

Merges distributed Spark CSV result files into a single consolidated result file for downstream analysis.

dataset_summary.py

Generates the dataset summary table used in the report, including total rows, total columns, anomaly count, and anomaly ratio.

evaluation_metrics.py

Computes quantitative evaluation outputs such as:

monthly anomaly distribution
threshold sensitivity
top transportation lines by anomaly rate
visualize_cloud_zscore.py

Generates visualizations from merged cloud results, including:

average passenger count by hour
passenger flow with anomalies for a selected line
top anomaly-prone lines
Technologies Used
Python
PySpark
Apache Spark
Google Cloud Dataproc
Google Cloud Storage
Pandas
Matplotlib
Current Results

At the midterm stage:

Total processed observations: approximately 6 million
Total detected anomalies: approximately 69 thousand
Overall anomaly ratio: approximately 1.14%
Strong temporal demand patterns are visible across hours
Anomalies are concentrated in specific months and transportation lines
Example Anomaly

An example anomaly identified by the baseline Z-score method:

Line: YENIKAPI - HACIOSMAN
Datetime: 2020-01-01 00:00:00
Observed passenger count: 6215
Expected group mean: approximately 822.5
Standard deviation: approximately 1789.3
Z-score: 3.01

This example shows how the method captures unusually high passenger demand relative to comparable temporal groups.

Current Status
Completed Spark-based preprocessing
Completed hourly aggregation
Completed Z-score anomaly detection
Generated preliminary quantitative results
Generated visual summaries and anomaly analysis
Prepared midterm report findings
Future Work

The next stage of the project will focus on:

implementing Isolation Forest
comparing statistical and machine learning-based methods
improving anomaly interpretation
expanding evaluation and characterization of results
Notes
Large raw datasets are not included in this repository.
Generated outputs and temporary cloud result files are excluded to keep the repository lightweight.
This repository contains the code base used for the course project and will continue to be extended in the final phase.