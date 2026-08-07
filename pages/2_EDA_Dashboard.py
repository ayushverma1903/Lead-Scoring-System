import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="EDA Dashboard", page_icon="📈", layout="wide")
st.title("📈 Exploratory Data Analysis")

# ── Load data from session_state ──────────────────────────────────────────────
DEFAULT_DATA_PATH = "data/raw/Lead Scoring.csv"

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

# Check for required EDA columns
if "Converted" not in df.columns:
    st.error("❌ Column 'Converted' not found. EDA requires a 'Converted' column (0/1).")
    st.stop()

# ── Conversion Rate by Lead Source ────────────────────────────────────────────
st.markdown("### 📊 Conversion Rate by Lead Source")
if "Lead Source" in df.columns:
    lead_source_conv = (
        df.groupby("Lead Source")["Converted"].mean()
        .sort_values(ascending=False).head(10)
    )
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    bars = ax1.bar(lead_source_conv.index, lead_source_conv.values * 100, color="#4C72B0")
    ax1.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax1.set_ylabel("Conversion Rate (%)")
    ax1.set_title("Top 10 Lead Sources by Conversion Rate")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)
else:
    st.info("'Lead Source' column not found in dataset.")

st.divider()

# ── Lead Origin Breakdown ─────────────────────────────────────────────────────
if "Lead Origin" in df.columns:
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### 📍 Lead Origin Distribution")
        origin_counts = df["Lead Origin"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.pie(origin_counts.values, labels=origin_counts.index,
                autopct="%1.1f%%", startangle=90,
                colors=["#3498db", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"])
        ax2.set_title("Lead Origin Distribution")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    with col_r:
        st.markdown("### 🏆 Conversion Rate by Lead Origin")
        origin_conv = (
            df.groupby("Lead Origin")["Converted"].mean()
            .sort_values(ascending=False) * 100
        )
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        bars3 = ax3.barh(origin_conv.index, origin_conv.values, color="#2ecc71")
        ax3.bar_label(bars3, fmt="%.1f%%", padding=3, fontsize=9)
        ax3.set_xlabel("Conversion Rate (%)")
        ax3.set_title("Conversion Rate by Lead Origin")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

st.divider()

# ── Total Visits vs Conversion ────────────────────────────────────────────────
st.markdown("### 🌐 Average Visits: Converted vs Not Converted")
if "TotalVisits" in df.columns:
    visits_df = df[df["TotalVisits"] < df["TotalVisits"].quantile(0.95)].dropna(subset=["TotalVisits", "Converted"])
    avg_visits = visits_df.groupby("Converted")["TotalVisits"].mean().rename({0: "Not Converted", 1: "Converted"})
    fig4, ax4 = plt.subplots(figsize=(5, 3))
    colors = ["#e74c3c", "#2ecc71"]
    bars4 = ax4.bar(avg_visits.index, avg_visits.values, color=colors, width=0.4)
    for i, v in enumerate(avg_visits.values):
        ax4.text(i, v + 0.05, f"{v:.1f}", ha="center", fontweight="bold")
    ax4.set_ylabel("Average Visits")
    ax4.set_title("Avg Total Visits: Converted vs Not Converted")
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)
else:
    st.info("'TotalVisits' column not found in dataset.")

st.divider()

# ── Time Spent vs Conversion ───────────────────────────────────────────────────
st.markdown("### ⏱️ Time Spent on Website vs Conversion")
if "Total Time Spent on Website" in df.columns:
    time_conv = (
        df.groupby("Converted")["Total Time Spent on Website"]
        .mean()
        .rename({0: "Not Converted", 1: "Converted"})
    )
    fig5, ax5 = plt.subplots(figsize=(5, 3))
    colors5 = ["#e74c3c", "#2ecc71"]
    ax5.bar(time_conv.index, time_conv.values, color=colors5, width=0.4)
    for i, v in enumerate(time_conv.values):
        ax5.text(i, v + 0.5, f"{v:.0f} min", ha="center", fontweight="bold")
    ax5.set_ylabel("Average Minutes on Site")
    ax5.set_title("Avg Time Spent: Converted vs Not Converted")
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)
else:
    st.info("'Total Time Spent on Website' column not found in dataset.")

st.divider()

# ── Missing Values ─────────────────────────────────────────────────────────────
st.markdown("### 🔍 Top 15 Missing Value Columns")
missing = df.isnull().sum().sort_values(ascending=False).head(15)
missing = missing[missing > 0]
if len(missing) > 0:
    fig6, ax6 = plt.subplots(figsize=(10, 4))
    bars6 = ax6.bar(missing.index, missing.values, color="#e74c3c")
    ax6.bar_label(bars6, padding=3, fontsize=9)
    ax6.set_ylabel("Missing Count")
    ax6.set_title("Columns with Most Missing Values")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close(fig6)
else:
    st.success("✅ No missing values in the dataset!")
