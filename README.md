# Scalable Anomaly Detection in Urban Transportation Data Using Apache Spark

## Overview
This project focuses on detecting anomalous passenger demand patterns in large-scale urban transportation data using Apache Spark.  
The dataset consists of hourly public transportation records from Istanbul Metropolitan Municipality (IMM) for the year 2020.

The main objective is to build a scalable data processing pipeline and apply statistical anomaly detection methods to identify unusual patterns in passenger behavior.

---

## Project Structure
.
├── src/
│ ├── spark_preprocessing.py
│ └── spark_zscore.py
├── requirements.txt
├── .gitignore
└── README.md


---

## Methodology

The project follows a structured data pipeline:

1. **Data Ingestion**
   - Monthly CSV files are loaded into Apache Spark

2. **Preprocessing**
   - Data cleaning
   - Type conversion
   - Construction of unified datetime field

3. **Aggregation**
   - Hourly aggregation based on:
     - datetime
     - line
     - line_name
     - town

4. **Feature Engineering**
   - hour
   - month
   - day of week
   - weekend indicator

5. **Anomaly Detection (Baseline)**
   - Z-score is computed on `total_passengers`
   - Grouped by (line, hour, day of week)
   - Observations with |Z| > 3 are labeled as anomalies

---

## Technologies Used

- Python
- PySpark
- Apache Spark
- Google Cloud Dataproc
- Google Cloud Storage

---

## Results (Midterm Stage)

- Total processed observations: ~6 million
- Detected anomalies: ~69,000
- Anomaly ratio: ~1.14%
- Strong temporal patterns observed (peak hours)
- Anomalies concentrated in specific months and transportation lines

---

## Example Anomaly

An example anomaly detected in the dataset:

- Line: YENIKAPI - HACIOSMAN  
- Time: 2020-01-01 00:00  
- Observed passengers: 6215  
- Expected (mean): ~822  
- Z-score: 3.01 → labeled as anomaly  

This demonstrates how the model detects unusually high passenger demand relative to expected behavior.

---

## Current Status

✔ Data preprocessing pipeline completed  
✔ Hourly aggregation implemented  
✔ Z-score anomaly detection working  
✔ Preliminary analysis and visualizations completed  

---

## Future Work

- Implement Isolation Forest for anomaly detection
- Compare statistical vs machine learning methods
- Improve anomaly interpretation
- Optimize pipeline performance

---

## Notes

- Large raw datasets are not included in this repository.
- The project is designed to run on a distributed cloud environment.
- This repository represents the midterm stage of the project.

---