import streamlit as st
import pandas as pd
import numpy as np
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(page_title="Business Insights", page_icon="💼", layout="wide")
st.title("💼 Business Insights & Operations")
st.markdown("Real KPIs computed from the loaded dataset, plus pipeline controls.")

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
has_converted = "Converted" in df.columns

st.caption(f"**Data source:** {source} — {df.shape[0]:,} rows × {df.shape[1]} columns")

if not has_converted:
    st.info("ℹ️ No `Converted` column found — showing **volume-based insights** only. Upload a labelled dataset to see conversion metrics.")

# ── KPI strip ──────────────────────────────────────────────────────────────────
st.subheader("📌 Key Performance Indicators")

total = len(df)
avg_visits = df["TotalVisits"].median() if "TotalVisits" in df.columns else None
avg_time = df["Total Time Spent on Website"].median() if "Total Time Spent on Website" in df.columns else None

if has_converted:
    converted = int(df["Converted"].sum())
    conv_rate = converted / total * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Leads",       f"{total:,}")
    c2.metric("Total Conversions", f"{converted:,}")
    c3.metric("Conversion Rate",   f"{conv_rate:.1f}%")
    c4.metric("Median Visits",     f"{avg_visits:.0f}" if avg_visits is not None else "N/A")
    c5.metric("Median Time (min)", f"{avg_time:.0f}" if avg_time is not None else "N/A")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Leads",       f"{total:,}")
    c2.metric("Median Visits",     f"{avg_visits:.0f}" if avg_visits is not None else "N/A")
    c3.metric("Median Time (min)", f"{avg_time:.0f}" if avg_time is not None else "N/A")

st.divider()

# ── Charts ─────────────────────────────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("🏆 Lead Sources")
    if "Lead Source" in df.columns:
        if has_converted:
            src_conv = df.groupby("Lead Source")["Converted"].mean().sort_values(ascending=False)
            top_src  = src_conv.head(8) * 100
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.barh(top_src.index[::-1], top_src.values[::-1], color="#2ecc71")
            ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
            ax.set_xlabel("Conversion Rate (%)")
            ax.set_title("Top 8 Lead Sources by Conversion Rate")
        else:
            src_vol = df["Lead Source"].value_counts().head(8)
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.barh(src_vol.index[::-1], src_vol.values[::-1], color="#3498db")
            ax.bar_label(bars, padding=3, fontsize=8)
            ax.set_xlabel("Number of Leads")
            ax.set_title("Top 8 Lead Sources by Volume")
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
    st.subheader("⏱️ Time Spent on Website")
    if has_converted:
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
    else:
        time_data = df["Total Time Spent on Website"].dropna()
        fig3, ax3 = plt.subplots(figsize=(8, 3))
        ax3.hist(time_data, bins=30, color="#e67e22", edgecolor="white", alpha=0.85)
        ax3.set_xlabel("Minutes on Site")
        ax3.set_ylabel("Count")
        ax3.set_title("Distribution of Time Spent on Website")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    st.divider()

# ── Do Not Email analysis ─────────────────────────────────────────────────────
if "Do Not Email" in df.columns:
    st.subheader("📧 Do Not Email Breakdown")
    email_counts = df["Do Not Email"].value_counts()
    c1, c2 = st.columns([1, 2])
    with c1:
        for label, count in email_counts.items():
            st.metric(f"Do Not Email = {label}", f"{count:,}")
    with c2:
        fig4, ax4 = plt.subplots(figsize=(4, 3))
        ax4.pie(email_counts.values, labels=email_counts.index, autopct="%1.1f%%",
                colors=["#2ecc71", "#e74c3c"], startangle=90)
        ax4.set_title("Do Not Email Distribution")
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
    with st.spinner("Running retraining pipeline... This may take a moment."):
        try:
            from src.retrain import retrain
            result = retrain()
            
            status = result.get("status", "error")
            message = result.get("message", "Unknown result")
            
            if status == "success":
                st.success(f"✅ {message}")
                metrics = result.get("metrics", {})
                if metrics:
                    mc1, mc2 = st.columns(2)
                    mc1.metric("New Model Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%")
                    mc2.metric("New Model AUC", f"{metrics.get('auc', 0):.4f}")
                st.info("🔄 Restart the app to use the new model for predictions.")
            elif status == "no_data":
                st.warning(f"⚠️ {message}")
            elif status == "not_improved":
                st.info(f"ℹ️ {message}")
            else:
                st.error(f"❌ {message}")
                
        except Exception as e:
            st.error(f"❌ Retraining failed: {str(e)}")
