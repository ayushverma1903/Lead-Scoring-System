import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dataset Overview", page_icon="📊", layout="wide")
st.title("📊 Dataset Overview")
st.markdown("A complete summary of the raw dataset used to train the Lead Scoring model.")

DATA_PATH = "data/raw/Lead Scoring.csv"

if not os.path.exists(DATA_PATH):
    st.warning("Raw dataset not found at `data/raw/Lead Scoring.csv`.")
    st.stop()

@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# ── Top KPI strip ──────────────────────────────────────────────────────────────
total       = len(df)
converted   = int(df["Converted"].sum()) if "Converted" in df.columns else 0
conv_rate   = converted / total * 100 if total else 0
n_features  = df.shape[1]
missing_pct = df.isnull().mean().mean() * 100

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Leads",        f"{total:,}")
c2.metric("Converted Leads",    f"{converted:,}")
c3.metric("Conversion Rate",    f"{conv_rate:.1f}%")
c4.metric("Total Features",     f"{n_features}")
c5.metric("Overall Missing %",  f"{missing_pct:.1f}%")

st.divider()

# ── Raw Data Preview ────────────────────────────────────────────────────────────
with st.expander("🔍 Raw Data Preview (first 100 rows)", expanded=False):
    st.dataframe(df.head(100), use_container_width=True)

st.divider()

# ── Two-column section ──────────────────────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("📋 Column Data Types")
    dtype_df = pd.DataFrame({
        "Column":   df.dtypes.index,
        "Dtype":    df.dtypes.astype(str).values,
        "Non-Null": df.notnull().sum().values,
        "Null":     df.isnull().sum().values,
        "Null %":   (df.isnull().mean() * 100).round(1).astype(str) + "%"
    })
    st.dataframe(dtype_df, use_container_width=True, height=400)

with right:
    st.subheader("📈 Conversion Rate by Lead Source")
    if "Lead Source" in df.columns and "Converted" in df.columns:
        src_conv = (
            df.groupby("Lead Source")["Converted"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "Conv Rate", "count": "Count"})
            .sort_values("Conv Rate", ascending=False)
            .head(10)
        )
        src_conv["Conv Rate %"] = (src_conv["Conv Rate"] * 100).round(1)
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(src_conv.index, src_conv["Conv Rate %"], color="#4C72B0")
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
        ax.set_xlabel("Conversion Rate (%)")
        ax.set_title("Top Lead Sources by Conversion Rate")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

st.divider()

# ── Numeric stats ───────────────────────────────────────────────────────────────
st.subheader("📐 Numeric Feature Statistics")
numeric_stats = df.describe().T.round(2)
st.dataframe(numeric_stats, use_container_width=True)
