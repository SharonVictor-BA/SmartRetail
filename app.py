import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import plotly.express as px
import plotly.graph_objects as go

from statsmodels.tsa.holtwinters import ExponentialSmoothing


# --------------------------------------
# STREAMLIT CONFIG
# --------------------------------------
st.set_page_config(
    page_title="Smart Retail Inventory Intelligence",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Smart-Retail: Predictive Analytics for Inventory Optimization")
st.markdown("Machine-learning powered PCA, clustering, and demand forecasting for FMCG inventory.")


# --------------------------------------
# DATA LOADERS
# --------------------------------------
@st.cache_data
def load_uploaded(file):
    return pd.read_csv(file)


@st.cache_data
def load_default():
    return pd.read_csv("SmartRetail_Forecasting_CompleteDataset.csv")


def prepare(df):
    # Ensure date field exists
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")

    # Derive margin if missing
    if "margin" not in df.columns and {"unit_price", "unit_cost"}.issubset(df.columns):
        df["margin"] = (df["unit_price"] - df["unit_cost"]) / df["unit_price"].replace(0, np.nan)

    # Fill numeric NaNs
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].fillna(df[num].median())

    return df


# --------------------------------------
# SIDEBAR: DATA INPUT
# --------------------------------------
st.sidebar.header("📁 Data Source")

uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded_file:
    df_raw = load_uploaded(uploaded_file)
    st.sidebar.success("Uploaded dataset loaded.")
else:
    df_raw = load_default()
    st.sidebar.info("Using default dataset: SmartRetail_Forecasting_CompleteDataset.csv")

df = prepare(df_raw)


# --------------------------------------
# SIDEBAR: SELECT MODE
# --------------------------------------
mode = st.sidebar.radio("Choose Mode", ["Dataset Overview", "PCA & Clustering", "Forecasting"])


# -------------------------------------------------------------------
# 1️⃣ DATASET OVERVIEW
# -------------------------------------------------------------------
if mode == "Dataset Overview":
    st.subheader("📊 Dataset Overview")

    st.markdown("### Preview")
    st.dataframe(df.head(20), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rows", df.shape[0])
    with c2:
        st.metric("Columns", df.shape[1])
    with c3:
        if "date" in df.columns:
            st.metric("Date Range", f"{df['date'].min().date()} → {df['date'].max().date()}")

    st.markdown("### Summary Statistics")
    st.dataframe(df.describe().T, use_container_width=True)

    st.markdown("### Category Distributions")
    candidates = [c for c in ["sku_id", "store_id", "category"] if c in df.columns]
    if candidates:
        col = st.selectbox("Choose Column", candidates)
        fig = px.bar(df[col].value_counts(), title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------------
# 2️⃣ PCA & CLUSTERING
# -------------------------------------------------------------------
elif mode == "PCA & Clustering":
    st.subheader("🧠 PCA & K-Means Clustering")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    st.markdown("### Select Features for PCA")
    selected = st.multiselect("Choose numeric columns", numeric_cols,
        default=[
            "sales_qty", "sales_revenue", "unit_cost", "unit_price", "margin",
            "lead_time_days", "delivery_reliability", "obsolescence_risk",
            "criticality_score", "sales_frequency", "sales_volatility",
            "rolling_avg_4w", "rolling_std_4w"
        ]
    )

    if len(selected) < 2:
        st.warning("Please select at least 2 features for PCA.")
        st.stop()

    n_components = st.slider("Number of PCA components", 2, min(6, len(selected)), 3)
    k = st.slider("Number of clusters (K)", 2, 10, 4)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[selected])

    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    pca_cols = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(X_pca, columns=pca_cols)

    # Clustering
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    df["cluster"] = kmeans.fit_predict(pca_df)

    # Show variance
    st.metric("Variance Explained", f"{pca.explained_variance_ratio_.sum()*100:.2f}%")

    # Loadings
    st.markdown("### PCA Loadings (Feature Weights)")
    load_df = pd.DataFrame(pca.components_.T, index=selected, columns=pca_cols)
    st.dataframe(load_df.style.background_gradient("Blues"), use_container_width=True)

    # Plot PC1 vs PC2
    if len(pca_cols) >= 2:
        fig = px.scatter(
            df.assign(**pca_df),
            x="PC1",
            y="PC2",
            color="cluster",
            hover_data=["sku_id", "store_id"],
            title="PCA Scatterplot with K-Means Clusters"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Cluster Summary
    st.markdown("### Cluster Profiles")
    st.dataframe(df.groupby("cluster")[selected].mean().round(2), use_container_width=True)


# -------------------------------------------------------------------
# 3️⃣ FORECASTING
# -------------------------------------------------------------------
elif mode == "Forecasting":
    st.subheader("📈 SKU-Level Forecasting")

    if "date" not in df.columns:
        st.error("Dataset must contain a 'date' column.")
        st.stop()

    # SKU selection
    sku_list = sorted(df["sku_id"].unique())
    sku = st.selectbox("Choose SKU", sku_list)

    store_list = sorted(df["store_id"].unique())
    store = st.selectbox("Choose Store", store_list)

    # Filter
    ts = df[(df["sku_id"] == sku) & (df["store_id"] == store)]
    ts = ts.groupby("date")["sales_qty"].sum().asfreq("W").fillna(0)

    st.markdown("### Historical Sales")
    st.line_chart(ts)

    horizon = st.slider("Forecast Horizon (weeks)", 4, 52, 12)

    # Model
    model = ExponentialSmoothing(ts, trend="add", seasonal=None).fit()
    fc = model.forecast(horizon)

    st.markdown("### Forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode="lines", name="History"))
    fig.add_trace(go.Scatter(x=fc.index, y=fc.values, mode="lines+markers", name="Forecast"))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(fc.reset_index().rename(columns={"index": "date", 0: "forecast"}))
