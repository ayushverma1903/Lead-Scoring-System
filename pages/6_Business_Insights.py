import streamlit as st
import pandas as pd
import numpy as np
import requests, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Business Insights", page_icon="💼", layout="wide")
st.title("💼 Business Insights & Operations")
st.markdown("Real KPIs computed from the loaded dataset, plus pipeline controls.")

API_URL = os.getenv("API_URL", "http://localhost:8000")
DEFAULT_DATA_PATH = "data/raw/Lead Scoring.csv"

# ── Load data from session_state ──────────────────────────────────────────────
if "user_df" not in st.session_state:
    if os.path.exists(DEFAULT_DATA_PATH):
        st.session_state["user_df"] = pd.read_csv(DEFAULT_DATA_PATH)
        st.session_state["data_source"] = "📁 Default: Lead Scoring.csv"

if "user_df" not in st.session_state:
    st.warning("No dataset loaded. Please go to the **Home** page and upload a dataset.")
    st.stop()

df = st.session_state["user_df"]
source = st.session_state.get("data_source", "")
st.caption(f"**Data source:** {source} — {df.shape[0]:,} rows × {df.shape[1]} columns")

if "Converted" not in df.columns:
    st.error("❌ Column 'Converted' not found. Business Insights requires a 'Converted' column (0/1).")
    st.stop()

# ── KPI strip ──────────────────────────────────────────────────────────────────
st.subheader("📌 Key Performance Indicators")

total     = len(df)
converted = int(df["Converted"].sum())
conv_rate = converted / total * 100
avg_visits = df["TotalVisits"].median() if "TotalVisits" in df.columns else None
avg_time   = df["Total Time Spent on Website"].median() if "Total Time Spent on Website" in df.columns else None

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Leads",       f"{total:,}")
c2.metric("Total Conversions", f"{converted:,}")
c3.metric("Conversion Rate",   f"{conv_rate:.1f}%")
c4.metric("Median Visits",     f"{avg_visits:.0f}" if avg_visits is not None else "N/A")
c5.metric("Median Time (min)", f"{avg_time:.0f}" if avg_time is not None else "N/A")

st.divider()

# ── Charts ─────────────────────────────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("🏆 Best Lead Sources")
    if "Lead Source" in df.columns:
        src_conv = df.groupby("Lead Source")["Converted"].mean().sort_values(ascending=False)
        top_src  = src_conv.head(8) * 100
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh(top_src.index[::-1], top_src.values[::-1], color="#2ecc71")
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
        ax.set_xlabel("Conversion Rate (%)")
        ax.set_title("Top 8 Lead Sources by Conversion Rate")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("'Lead Source' column not found.")

with col_r:
    st.subheader("📍 Lead Origin Breakdown")
    if "Lead Origin" in df.columns:
        origin_counts = df["Lead Origin"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.pie(origin_counts.values, labels=origin_counts.index,
                autopct="%1.1f%%", startangle=90,
                colors=["#3498db", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"])
        ax2.set_title("Lead Origin Distribution")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
    else:
        st.info("'Lead Origin' column not found.")

st.divider()

# ── Time spent vs conversion ───────────────────────────────────────────────────
if "Total Time Spent on Website" in df.columns:
    st.subheader("⏱️ Time Spent on Website vs Conversion")
    time_conv = (
        df.groupby("Converted")["Total Time Spent on Website"]
        .mean()
        .rename({0: "Not Converted", 1: "Converted"})
    )
    fig3, ax3 = plt.subplots(figsize=(5, 3))
    colors = ["#e74c3c", "#2ecc71"]
    ax3.bar(time_conv.index, time_conv.values, color=colors, width=0.4)
    ax3.set_ylabel("Average Minutes on Site")
    ax3.set_title("Avg Time Spent: Converted vs Not Converted")
    for i, v in enumerate(time_conv.values):
        ax3.text(i, v + 0.5, f"{v:.0f} min", ha="center", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    st.divider()

# ── Specialization breakdown ───────────────────────────────────────────────────
if "Specialization" in df.columns:
    st.subheader("🎓 Top Specializations by Conversion Rate")
    spec_conv = (
        df[df["Specialization"] != "Unknown"]
        .groupby("Specialization")["Converted"].mean()
        .sort_values(ascending=False).head(10) * 100
    )
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    bars4 = ax4.bar(spec_conv.index, spec_conv.values, color="#9b59b6")
    ax4.bar_label(bars4, fmt="%.1f%%", padding=3, fontsize=8)
    ax4.set_ylabel("Conversion Rate (%)")
    ax4.set_title("Top 10 Specializations by Conversion Rate")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    st.divider()

# ── Retraining control ─────────────────────────────────────────────────────────
st.subheader("🔄 Automated Retraining Pipeline")
st.markdown(
    "Drop new lead CSVs in **`data/new_data/`** on the server, then trigger retraining below. "
    "The new model will replace the current one only if it outperforms it."
)
if st.button("🚀 Trigger Retraining Pipeline"):
    try:
        resp = requests.post(f"{API_URL}/retrain", timeout=10)
        if resp.status_code == 200:
            st.success("✅ Retraining task started in the background. Check API logs for progress.")
        else:
            st.error(f"Failed: {resp.text}")
    except Exception as e:
        st.error(f"Could not reach API: {e}")
