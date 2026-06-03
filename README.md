# Scalable Anomaly Detection in Urban Transportation Data Using Apache Spark

## Overview

This project investigates anomalous passenger demand patterns in large-scale urban transportation data using **Apache Spark** and **Google Cloud Dataproc**.

The dataset consists of hourly public transportation records from the Istanbul Metropolitan Municipality (IMM) Open Data Portal for the year **2020**. The main goal is to build a scalable data processing pipeline and detect unusual passenger demand behavior in different transportation lines and time periods.

The project compares two unsupervised anomaly detection methods:

1. **Contextual Z-score baseline**
2. **Contextual Isolation Forest**

Since the dataset does not contain ground-truth anomaly labels, the evaluation focuses on anomaly counts, anomaly rates, method agreement, monthly distribution, line-based anomaly concentration, threshold sensitivity, and interpretability rather than supervised accuracy metrics.

---

## Research Question

Which transportation lines and time periods show passenger demand that deviates significantly from expected contextual behavior?

In this project, an observation is evaluated within a comparable context:

```text
line + hour + day_of_week
```

This prevents naturally different situations, such as weekday morning traffic and late-night weekend traffic, from being compared directly.

---

## Dataset

- **Data source:** Istanbul Metropolitan Municipality Open Data Portal
- **Period:** 2020
- **Granularity:** Hourly public transportation records
- **Main fields:**
  - transition_date
  - transition_hour
  - line
  - line_name
  - town
  - number_of_passenger
  - number_of_passage

After Spark preprocessing and aggregation:

```text
Z-score output observations: 6,039,872
Final comparison observations: 6,008,418
```

The slight difference is caused by context filtering before the Isolation Forest comparison.

---

## Repository Structure

```text
.
├── dashboard/
│   └── app.py
│
├── outputs/
│   └── evaluation/
│       ├── anomaly_distribution_by_month.csv
│       ├── anomaly_distribution_by_month.png
│       ├── threshold_sensitivity.csv
│       ├── threshold_sensitivity.png
│       ├── top5_line_anomaly_rate.csv
│       ├── top5_line_anomaly_rate.png
│       ├── isolation_forest_summary.csv
│       ├── monthly_method_comparison.csv
│       └── line_method_comparison.csv
│
├── src/
│   ├── dataset_summary.py
│   ├── evaluation_metrics.py
│   ├── merge_cloud_results.py
│   ├── spark_isolation_forest.py
│   ├── spark_preprocessing.py
│   ├── spark_zscore.py
│   └── visualize_cloud_zscore.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

Raw data files and full Spark output folders are not included in the repository because of file size. The repository includes source code, selected final evaluation outputs, and the dashboard application.

---

## System Architecture

The project follows a cloud-based big data processing architecture:

```text
Raw IMM CSV files
        ↓
Google Cloud Storage
        ↓
Google Cloud Dataproc
        ↓
Apache Spark preprocessing and aggregation
        ↓
Feature engineering
        ↓
Contextual Z-score detection
        ↓
Contextual Isolation Forest detection
        ↓
Evaluation outputs
        ↓
Streamlit dashboard
```

The cloud processing environment uses Google Cloud Dataproc with one master node and two worker nodes.

---

## Methodology

### 1. Spark Preprocessing

The preprocessing stage performs:

- loading monthly CSV files from Google Cloud Storage
- type casting
- invalid/null record filtering
- construction of a unified datetime field
- filtering invalid hours and negative passenger counts
- hourly aggregation by:

```text
datetime + line + line_name + town
```

For each group, the following aggregated values are computed:

```text
total_passengers
total_passages
record_count
```

Additional temporal features are also extracted:

```text
hour
month
day_of_week
is_weekend
```

---

### 2. Contextual Z-score Baseline

The Z-score method detects strong passenger-count deviations within each context group:

```text
line + hour + day_of_week
```

Formula:

```text
Z = (x - mean) / standard deviation
```

where:

```text
x = observed total passenger count
mean = average passenger count within the same context
standard deviation = passenger-count variation within the same context
```

An observation is labeled as anomalous if:

```text
|Z| > 3
```

The threshold of 3 was selected as a conservative baseline. Threshold sensitivity analysis was also performed:

```text
|Z| > 2.0  → 281,601 anomalies
|Z| > 2.5  → 140,063 anomalies
|Z| > 3.0  → 69,143 anomalies
```

---

### 3. Contextual Isolation Forest

Isolation Forest is used as a machine learning-based extension to the Z-score baseline.

The model does not rely on ground-truth anomaly labels. Instead, it detects observations that are easier to isolate based on context-derived demand features.

The final feature set focuses on residual and ratio-based passenger demand behavior, including:

```text
context-normalized residual
relative deviation from context mean
log observed / expected ratio
passage-per-passenger difference
passenger-per-record difference
absolute-value variants of these features
```

The model configuration is:

```text
Trees: 50
Max samples per tree: 256
Training pool: 50,000 observations
Context filter: at least 5 observations per context
Contamination: 1.1448%
```

The contamination value is used to obtain a comparable anomaly volume with the Z-score baseline.

---

## Final Results

| Metric | Value |
|---|---:|
| Final comparison observations | 6,008,418 |
| Z-score anomalies | 69,143 |
| Isolation Forest anomalies | 74,542 |
| Detected by both methods | 38,742 |
| Only Z-score | 30,401 |
| Only Isolation Forest | 35,800 |
| Z-score anomaly rate | 1.15% |
| Isolation Forest anomaly rate | 1.24% |

The two methods detect a substantial common set of anomalies while also producing complementary method-specific results.

```text
56.03% of Z-score anomalies are also detected by Isolation Forest.
51.97% of Isolation Forest anomalies are also detected by Z-score.
```

---

## Monthly Analysis

The monthly anomaly distribution shows that anomalies are not evenly distributed throughout the year.

Z-score anomalies are concentrated mostly in the first quarter of 2020. In April and May, Z-score anomaly counts drop sharply, while Isolation Forest still detects additional anomalies through residual and ratio-based demand features.

This suggests that different anomaly definitions highlight different aspects of mobility behavior, especially during the COVID-19 period.

---

## Line-Based Analysis

Line-level anomaly concentration is evaluated using:

```text
anomaly rate = anomaly count / total observations
```

To avoid unstable rates from very small routes, only lines with at least **10,000 observations** are included in the final ranking.

Top lines by Isolation Forest anomaly rate:

| Line | Observations | Anomalies | Rate |
|---|---:|---:|---:|
| IC HATLAR | 18,419 | 1,628 | 8.84% |
| KADIKOY-PENDIK | 31,518 | 2,413 | 7.66% |
| HALKALI-GEBZE | 88,342 | 5,407 | 6.12% |
| ALTUNIZADE-SULTANBEYLI | 18,723 | 963 | 5.14% |
| GUNESLI-BEYAZIT | 13,032 | 593 | 4.55% |

---

## Dashboard

An interactive dashboard was developed using **Streamlit** and **Plotly**.

The dashboard includes:

- project overview
- method comparison
- anomaly overlap analysis
- monthly anomaly analysis
- line-based anomaly analysis
- interpretation of key findings

To run the dashboard locally:

```bash
python3 -m streamlit run dashboard/app.py
```

The dashboard reads final evaluation CSV files from:

```text
outputs/evaluation/
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run preprocessing on Spark

```bash
spark-submit src/spark_preprocessing.py
```

### 3. Run contextual Z-score detection

```bash
spark-submit src/spark_zscore.py
```

### 4. Run Contextual Isolation Forest

```bash
spark-submit src/spark_isolation_forest.py
```

### 5. Run evaluation scripts

```bash
python src/evaluation_metrics.py
```

### 6. Launch dashboard

```bash
python3 -m streamlit run dashboard/app.py
```

For cloud execution, the Spark scripts are submitted to Google Cloud Dataproc and read/write data through Google Cloud Storage.

---

## Technologies Used

- Python
- PySpark
- Apache Spark
- Google Cloud Dataproc
- Google Cloud Storage
- Pandas
- Matplotlib
- Streamlit
- Plotly

---

## Limitations

This project is based on unsupervised anomaly detection. Therefore:

- no ground-truth anomaly labels are available
- precision, recall, accuracy, and F1-score cannot be computed honestly
- detected anomalies should be interpreted as candidate signals for further inspection
- 2020 is an unusual year because of COVID-19-related mobility changes
- Isolation Forest is implemented as a Spark-compatible custom approach rather than a standard library model

---

## Conclusion

This project presents a scalable anomaly detection pipeline for Istanbul public transportation data using Apache Spark and Google Cloud Dataproc.

The contextual Z-score baseline provides an interpretable statistical method for detecting strong passenger-demand deviations. Contextual Isolation Forest complements it by detecting additional irregular patterns based on residual and ratio-based demand features.

The final results show that the two methods agree on 38,742 anomalies while also identifying complementary cases. Since no ground-truth labels are available, the project focuses on method agreement, anomaly distribution, interpretability, and scalable processing rather than supervised accuracy claims.
