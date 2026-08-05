import streamlit as st
import pandas as pd
import numpy as np
import requests, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Business Insights", page_icon="💼", layout="wide")
st.title("💼 Business Insights & Operations")
st.markdown("Real KPIs computed from the dataset, plus pipeline controls.")

API_URL   = os.getenv("API_URL", "http://localhost:8000")
DATA_PATH = "data/raw/Lead Scoring.csv"

@st.cache_data(show_spinner="Computing business metrics…")
def compute_kpis():
    df = pd.read_csv(DATA_PATH)
    total        = len(df)
    converted    = int(df["Converted"].sum())
    conv_rate    = converted / total * 100
    avg_visits   = df["TotalVisits"].median()
    avg_time     = df["Total Time Spent on Website"].median()

    # Lead Origin breakdown
    origin_conv  = df.groupby("Lead Origin")["Converted"].mean().sort_values(ascending=False)

    # Best lead source
    src_conv     = df.groupby("Lead Source")["Converted"].mean().sort_values(ascending=False)
    best_source  = src_conv.index[0] if not src_conv.empty else "N/A"
    best_rate    = src_conv.iloc[0] * 100 if not src_conv.empty else 0

    return {
        "total":       total,
        "converted":   converted,
        "conv_rate":   conv_rate,
        "avg_visits":  avg_visits,
        "avg_time":    avg_time,
        "origin_conv": origin_conv,
        "src_conv":    src_conv,
        "best_source": best_source,
        "best_rate":   best_rate,
        "df":          df,
    }

if not os.path.exists(DATA_PATH):
    st.warning("Raw dataset not found.")
    st.stop()

kpis = compute_kpis()
df   = kpis["df"]

# ── KPI strip ──────────────────────────────────────────────────────────────────
st.subheader("📌 Key Performance Indicators")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Leads",       f"{kpis['total']:,}")
c2.metric("Total Conversions", f"{kpis['converted']:,}")
c3.metric("Conversion Rate",   f"{kpis['conv_rate']:.1f}%")
c4.metric("Median Visits",     f"{kpis['avg_visits']:.0f}")
c5.metric("Median Time (min)", f"{kpis['avg_time']:.0f}")

st.divider()

# ── Charts ─────────────────────────────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("🏆 Best Lead Sources")
    top_src = kpis["src_conv"].head(8) * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(top_src.index[::-1], top_src.values[::-1], color="#2ecc71")
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    ax.set_xlabel("Conversion Rate (%)")
    ax.set_title("Top 8 Lead Sources by Conversion Rate")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col_r:
    st.subheader("📍 Lead Origin Breakdown")
    origin_counts = df["Lead Origin"].value_counts()
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.pie(origin_counts.values, labels=origin_counts.index,
            autopct="%1.1f%%", startangle=90,
            colors=["#3498db","#e74c3c","#f39c12","#9b59b6","#1abc9c"])
    ax2.set_title("Lead Origin Distribution")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

st.divider()

# ── Time spent vs conversion ───────────────────────────────────────────────────
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
