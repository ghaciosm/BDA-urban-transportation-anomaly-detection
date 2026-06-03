from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Urban Transportation Anomaly Detection",
    page_icon="🚇",
    layout="wide",
)


# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "outputs" / "evaluation"

SUMMARY_PATH = EVAL_DIR / "isolation_forest_v3_summary.csv"
MONTHLY_PATH = EVAL_DIR / "monthly_method_comparison_v3.csv"
LINE_PATH = EVAL_DIR / "line_method_comparison_v3.csv"


# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def check_files():
    missing = []
    for path in [SUMMARY_PATH, MONTHLY_PATH, LINE_PATH]:
        if not path.exists():
            missing.append(str(path))

    if missing:
        st.error("Some required result files are missing.")
        st.code("\n".join(missing))
        st.stop()


def format_int(value):
    try:
        return f"{int(value):,}"
    except Exception:
        return value


def format_percent(value):
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return value


def get_first_row_value(df, col):
    if col not in df.columns:
        return None
    return df[col].iloc[0]


# -----------------------------
# Load data
# -----------------------------
check_files()

summary_df = load_csv(SUMMARY_PATH)
monthly_df = load_csv(MONTHLY_PATH)
line_df = load_csv(LINE_PATH)

summary = summary_df.iloc[0]

total_rows = summary["total_rows"]
zscore_count = summary["zscore_anomaly_count"]
iforest_count = summary["iforest_v3_anomaly_count"]
both_count = summary["both_methods_anomaly_count"]
only_zscore = summary["only_zscore_anomaly_count"]
only_iforest = summary["only_iforest_v3_anomaly_count"]
zscore_rate = summary["zscore_anomaly_rate"]
iforest_rate = summary["iforest_v3_anomaly_rate"]
overlap_rate = summary["method_overlap_rate"]


# -----------------------------
# Header
# -----------------------------
st.title("🚇 Scalable Anomaly Detection in Urban Transportation Data")
st.caption(
    "Apache Spark + Google Cloud Dataproc pipeline for detecting unusual hourly passenger demand patterns."
)

st.divider()


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Dashboard Controls")

min_obs = st.sidebar.slider(
    "Minimum observations for line analysis",
    min_value=0,
    max_value=100_000,
    value=10_000,
    step=1_000,
)

top_n = st.sidebar.slider(
    "Number of top lines to display",
    min_value=5,
    max_value=30,
    value=10,
    step=1,
)

st.sidebar.divider()

st.sidebar.markdown("### Methods")
st.sidebar.markdown(
    """
    **Z-score**  
    Contextual statistical baseline.

    **Contextual Isolation Forest V3**  
    Multivariate anomaly detection extension.
    """
)


# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_comparison, tab_monthly, tab_lines, tab_interpretation = st.tabs(
    [
        "Project Overview",
        "Method Comparison",
        "Monthly Analysis",
        "Line Analysis",
        "Interpretation",
    ]
)


# -----------------------------
# Project Overview
# -----------------------------
with tab_overview:
    st.header("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Observations", format_int(total_rows))
    col2.metric("Z-score Anomalies", format_int(zscore_count), format_percent(zscore_rate))
    col3.metric("Contextual Isolation Forest V3 Anomalies", format_int(iforest_count), format_percent(iforest_rate))
    col4.metric("Common Anomalies", format_int(both_count), format_percent(overlap_rate))

    st.subheader("Problem Definition")
    st.markdown(
        """
        This project focuses on detecting **unusual hourly passenger demand patterns**
        in Istanbul public transportation data.

        The core question is:

        > Which transportation lines and time periods show passenger demand that deviates from expected behavior?

        The project uses a scalable data processing pipeline based on **Apache Spark**
        and **Google Cloud Dataproc**. The final anomaly detection results are visualized
        in this dashboard.
        """
    )

    st.subheader("Pipeline")
    st.markdown(
        """
        **Raw Data → Spark Preprocessing → Hourly Aggregation → Feature Engineering → Z-score → Contextual Isolation Forest V3 → Evaluation**
        """
    )

    pipeline_df = pd.DataFrame(
        {
            "Stage": [
                "Data Source",
                "Cloud Storage",
                "Processing Framework",
                "Baseline Method",
                "ML Extension",
                "Final Output",
            ],
            "Description": [
                "IMM hourly public transportation dataset",
                "Google Cloud Storage bucket",
                "Apache Spark on Google Cloud Dataproc",
                "Contextual Z-score anomaly detection",
                "Contextual Isolation Forest V3 with refined features",
                "Summary tables, monthly analysis, line-based analysis, dashboard",
            ],
        }
    )

    st.dataframe(pipeline_df, use_container_width=True, hide_index=True)


# -----------------------------
# Method Comparison
# -----------------------------
with tab_comparison:
    st.header("Z-score vs Contextual Isolation Forest V3")

    comparison_df = pd.DataFrame(
        {
            "Method": ["Z-score", "Contextual Isolation Forest V3"],
            "Anomaly Count": [zscore_count, iforest_count],
            "Anomaly Rate": [zscore_rate, iforest_rate],
        }
    )

    col1, col2 = st.columns([1.1, 1])

    with col1:
        fig = px.bar(
            comparison_df,
            x="Method",
            y="Anomaly Count",
            text="Anomaly Count",
            title="Detected Anomaly Count by Method",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(yaxis_title="Anomaly Count", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Comparison Summary")
        st.dataframe(
            comparison_df.assign(
                **{
                    "Anomaly Count": comparison_df["Anomaly Count"].map(lambda x: f"{int(x):,}"),
                    "Anomaly Rate": comparison_df["Anomaly Rate"].map(lambda x: f"{x * 100:.2f}%"),
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "Both methods detect anomalies at a similar overall rate, which makes the comparison more balanced."
        )

    st.subheader("Overlap Between Methods")

    overlap_df = pd.DataFrame(
        {
            "Category": [
                "Detected by Both",
                "Only Z-score",
                "Only Contextual Isolation Forest V3",
            ],
            "Count": [
                both_count,
                only_zscore,
                only_iforest,
            ],
        }
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        fig = px.pie(
            overlap_df,
            names="Category",
            values="Count",
            title="Overlap of Detected Anomalies",
            hole=0.35,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.dataframe(
            overlap_df.assign(
                Count=overlap_df["Count"].map(lambda x: f"{int(x):,}")
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            """
            The overlap is relatively low, which indicates that the two methods capture
            **different anomaly perspectives**.

            - **Z-score** focuses on contextual passenger-count deviation.
            - **Contextual Isolation Forest V3** captures multivariate irregular patterns.
            """
        )


# -----------------------------
# Monthly Analysis
# -----------------------------
with tab_monthly:
    st.header("Monthly Anomaly Analysis")

    monthly_plot_df = monthly_df.copy()
    monthly_plot_df["month"] = monthly_plot_df["month"].astype(int)

    monthly_long = monthly_plot_df.melt(
        id_vars=["month"],
        value_vars=["zscore_anomaly_count", "iforest_v3_anomaly_count"],
        var_name="Method",
        value_name="Anomaly Count",
    )

    monthly_long["Method"] = monthly_long["Method"].replace(
        {
            "zscore_anomaly_count": "Z-score",
            "iforest_v3_anomaly_count": "Contextual Isolation Forest V3",
        }
    )

    fig = px.line(
        monthly_long,
        x="month",
        y="Anomaly Count",
        color="Method",
        markers=True,
        title="Monthly Anomaly Count: Z-score vs Contextual Isolation Forest V3",
    )
    fig.update_layout(xaxis_title="Month", yaxis_title="Anomaly Count")
    st.plotly_chart(fig, use_container_width=True)

    monthly_rate_long = monthly_plot_df.melt(
        id_vars=["month"],
        value_vars=["zscore_anomaly_rate", "iforest_v3_anomaly_rate"],
        var_name="Method",
        value_name="Anomaly Rate",
    )

    monthly_rate_long["Method"] = monthly_rate_long["Method"].replace(
        {
            "zscore_anomaly_rate": "Z-score",
            "iforest_v3_anomaly_rate": "Contextual Isolation Forest V3",
        }
    )

    fig = px.bar(
        monthly_rate_long,
        x="month",
        y="Anomaly Rate",
        color="Method",
        barmode="group",
        title="Monthly Anomaly Rate by Method",
    )
    fig.update_layout(xaxis_title="Month", yaxis_title="Anomaly Rate")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Result Table")
    display_monthly = monthly_plot_df.copy()
    display_monthly["zscore_anomaly_rate"] = display_monthly["zscore_anomaly_rate"].map(lambda x: f"{x * 100:.2f}%")
    display_monthly["iforest_v3_anomaly_rate"] = display_monthly["iforest_v3_anomaly_rate"].map(lambda x: f"{x * 100:.2f}%")
    st.dataframe(display_monthly, use_container_width=True, hide_index=True)


# -----------------------------
# Line Analysis
# -----------------------------
with tab_lines:
    st.header("Line-Based Anomaly Analysis")

    filtered_lines = line_df[line_df["total_obs"] >= min_obs].copy()

    if filtered_lines.empty:
        st.warning("No lines match the selected minimum observation threshold.")
    else:
        filtered_lines = filtered_lines.sort_values(
            "iforest_v3_anomaly_rate",
            ascending=False,
        )

        top_lines = filtered_lines.head(top_n)

        st.caption(
            f"Showing top {top_n} lines with at least {min_obs:,} observations."
        )

        fig = px.bar(
            top_lines.sort_values("iforest_v3_anomaly_rate"),
            x="iforest_v3_anomaly_rate",
            y="line",
            orientation="h",
            title="Top Lines by Contextual Isolation Forest V3 Anomaly Rate",
            text="iforest_v3_anomaly_rate",
        )
        fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
        fig.update_layout(
            xaxis_title="Contextual Isolation Forest V3 Anomaly Rate",
            yaxis_title="Transportation Line",
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(
            filtered_lines,
            x="zscore_anomaly_rate",
            y="iforest_v3_anomaly_rate",
            size="total_obs",
            hover_name="line",
            title="Line-Level Method Comparison",
        )
        fig.update_layout(
            xaxis_title="Z-score Anomaly Rate",
            yaxis_title="Contextual Isolation Forest V3 Anomaly Rate",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Filtered Line Comparison Table")

        display_lines = top_lines.copy()
        display_lines["zscore_anomaly_rate"] = display_lines["zscore_anomaly_rate"].map(lambda x: f"{x * 100:.2f}%")
        display_lines["iforest_v3_anomaly_rate"] = display_lines["iforest_v3_anomaly_rate"].map(lambda x: f"{x * 100:.2f}%")
        display_lines["total_obs"] = display_lines["total_obs"].map(lambda x: f"{int(x):,}")
        display_lines["zscore_anomaly_count"] = display_lines["zscore_anomaly_count"].map(lambda x: f"{int(x):,}")
        display_lines["iforest_v3_anomaly_count"] = display_lines["iforest_v3_anomaly_count"].map(lambda x: f"{int(x):,}")

        st.dataframe(display_lines, use_container_width=True, hide_index=True)


# -----------------------------
# Interpretation
# -----------------------------
with tab_interpretation:
    st.header("Interpretation and Key Findings")

    st.subheader("1. Why Z-score is a strong baseline")
    st.markdown(
        """
        Z-score is simple, but it directly matches the main anomaly definition:
        passenger demand that deviates strongly from expected contextual behavior.

        In this project, Z-scores are computed within comparable groups such as
        transportation line, hour of day, and day of week. This makes the method
        highly interpretable.
        """
    )

    st.subheader("2. What Contextual Isolation Forest V3 adds")
    st.markdown(
        """
        Contextual Isolation Forest V3 provides a multivariate perspective. Instead of only
        checking whether passenger count is far from the group mean, it considers
        multiple features such as:

        - transformed passenger volume,
        - transformed passage count,
        - record count,
        - temporal features,
        - group mean and group standard deviation,
        - relative deviation from expected behavior.

        Therefore, it can detect anomaly patterns that are not captured by Z-score alone.
        """
    )

    st.subheader("3. Why the overlap is low")
    st.markdown(
        f"""
        The methods overlap on **{format_int(both_count)}** observations.

        This relatively low overlap does not necessarily mean one method is wrong.
        It means the methods detect different types of anomalies:

        - Z-score detects extreme univariate deviations.
        - Contextual Isolation Forest V3 detects unusual multivariate combinations.
        """
    )

    st.subheader("4. Limitation")
    st.markdown(
        """
        Since the dataset does not contain ground-truth anomaly labels, the project
        cannot claim that one method is objectively more accurate than the other.
        Instead, the evaluation focuses on anomaly counts, anomaly rates, monthly
        distributions, line-based patterns, and interpretability.
        """
    )

    st.subheader("Final Conclusion")
    st.success(
        """
        Z-score is the most interpretable and reliable statistical baseline.
        Contextual Isolation Forest V3 complements it by revealing additional multivariate
        anomaly patterns. Together, they provide a stronger analysis than either
        method alone.
        """
    )