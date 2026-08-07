import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="EDA Dashboard", page_icon="📈", layout="wide")
st.title("📈 Exploratory Data Analysis")

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
    st.info("ℹ️ No `Converted` column found — showing **distribution analysis** only. Upload a labelled dataset to see conversion rates.")
else:
    st.success("✅ `Converted` column detected — showing full conversion analysis.")

st.divider()

# ── Lead Source ───────────────────────────────────────────────────────────────
if "Lead Source" in df.columns:
    st.markdown("### 📊 Lead Source Analysis")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Volume by Lead Source**")
        src_counts = df["Lead Source"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh(src_counts.index[::-1], src_counts.values[::-1], color="#4C72B0")
        ax.bar_label(bars, padding=3, fontsize=9)
        ax.set_xlabel("Number of Leads")
        ax.set_title("Top 10 Lead Sources by Volume")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_r:
        if has_converted:
            st.markdown("**Conversion Rate by Lead Source**")
            src_conv = (
                df.groupby("Lead Source")["Converted"].mean()
                .sort_values(ascending=False).head(10) * 100
            )
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            bars2 = ax2.barh(src_conv.index[::-1], src_conv.values[::-1], color="#2ecc71")
            ax2.bar_label(bars2, fmt="%.1f%%", padding=3, fontsize=9)
            ax2.set_xlabel("Conversion Rate (%)")
            ax2.set_title("Conversion Rate by Lead Source")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.markdown("**Lead Source Distribution (Pie)**")
            src_counts2 = df["Lead Source"].value_counts().head(6)
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.pie(src_counts2.values, labels=src_counts2.index, autopct="%1.1f%%", startangle=90)
            ax2.set_title("Lead Source Distribution")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

    st.divider()

# ── Lead Origin ───────────────────────────────────────────────────────────────
if "Lead Origin" in df.columns:
    st.markdown("### 📍 Lead Origin Analysis")
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        origin_counts = df["Lead Origin"].value_counts()
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        ax3.pie(origin_counts.values, labels=origin_counts.index,
                autopct="%1.1f%%", startangle=90,
                colors=["#3498db", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"])
        ax3.set_title("Lead Origin Distribution")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    with col_r2:
        if has_converted:
            origin_conv = (
                df.groupby("Lead Origin")["Converted"].mean()
                .sort_values(ascending=False) * 100
            )
            fig4, ax4 = plt.subplots(figsize=(5, 4))
            bars4 = ax4.barh(origin_conv.index, origin_conv.values, color="#2ecc71")
            ax4.bar_label(bars4, fmt="%.1f%%", padding=3, fontsize=9)
            ax4.set_xlabel("Conversion Rate (%)")
            ax4.set_title("Conversion Rate by Lead Origin")
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)
        else:
            origin_vol = df["Lead Origin"].value_counts()
            fig4, ax4 = plt.subplots(figsize=(5, 4))
            bars4 = ax4.barh(origin_vol.index[::-1], origin_vol.values[::-1], color="#9b59b6")
            ax4.bar_label(bars4, padding=3, fontsize=9)
            ax4.set_xlabel("Count")
            ax4.set_title("Lead Count by Origin")
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)

    st.divider()

# ── Visits Distribution ───────────────────────────────────────────────────────
if "TotalVisits" in df.columns:
    st.markdown("### 🌐 Total Visits Distribution")
    col_l3, col_r3 = st.columns(2)

    with col_l3:
        visit_data = df["TotalVisits"].dropna()
        visit_data = visit_data[visit_data < visit_data.quantile(0.95)]
        fig5, ax5 = plt.subplots(figsize=(5, 3))
        ax5.hist(visit_data, bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
        ax5.set_xlabel("Total Visits")
        ax5.set_ylabel("Count")
        ax5.set_title("Distribution of Total Visits")
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)

    with col_r3:
        if has_converted:
            avg_visits = (
                df[df["TotalVisits"] < df["TotalVisits"].quantile(0.95)]
                .dropna(subset=["TotalVisits", "Converted"])
                .groupby("Converted")["TotalVisits"].mean()
                .rename({0: "Not Converted", 1: "Converted"})
            )
            fig6, ax6 = plt.subplots(figsize=(5, 3))
            ax6.bar(avg_visits.index, avg_visits.values, color=["#e74c3c", "#2ecc71"], width=0.4)
            for i, v in enumerate(avg_visits.values):
                ax6.text(i, v + 0.05, f"{v:.1f}", ha="center", fontweight="bold")
            ax6.set_ylabel("Average Visits")
            ax6.set_title("Avg Visits: Converted vs Not")
            plt.tight_layout()
            st.pyplot(fig6)
            plt.close(fig6)
        else:
            st.metric("Median Total Visits", f"{df['TotalVisits'].median():.0f}")
            st.metric("Mean Total Visits", f"{df['TotalVisits'].mean():.1f}")
            st.metric("Max Total Visits", f"{df['TotalVisits'].max():.0f}")

    st.divider()

# ── Time Spent ────────────────────────────────────────────────────────────────
if "Total Time Spent on Website" in df.columns:
    st.markdown("### ⏱️ Time Spent on Website")
    col_l4, col_r4 = st.columns(2)

    with col_l4:
        time_data = df["Total Time Spent on Website"].dropna()
        fig7, ax7 = plt.subplots(figsize=(5, 3))
        ax7.hist(time_data, bins=30, color="#e67e22", edgecolor="white", alpha=0.85)
        ax7.set_xlabel("Minutes")
        ax7.set_ylabel("Count")
        ax7.set_title("Distribution of Time Spent")
        plt.tight_layout()
        st.pyplot(fig7)
        plt.close(fig7)

    with col_r4:
        if has_converted:
            time_conv = (
                df.groupby("Converted")["Total Time Spent on Website"]
                .mean()
                .rename({0: "Not Converted", 1: "Converted"})
            )
            fig8, ax8 = plt.subplots(figsize=(5, 3))
            ax8.bar(time_conv.index, time_conv.values, color=["#e74c3c", "#2ecc71"], width=0.4)
            for i, v in enumerate(time_conv.values):
                ax8.text(i, v + 0.5, f"{v:.0f} min", ha="center", fontweight="bold")
            ax8.set_ylabel("Avg Minutes")
            ax8.set_title("Avg Time: Converted vs Not")
            plt.tight_layout()
            st.pyplot(fig8)
            plt.close(fig8)
        else:
            st.metric("Median Time on Site", f"{time_data.median():.0f} min")
            st.metric("Mean Time on Site",   f"{time_data.mean():.1f} min")
            st.metric("Max Time on Site",    f"{time_data.max():.0f} min")

    st.divider()

# ── Missing Values ─────────────────────────────────────────────────────────────
st.markdown("### 🔍 Top 15 Columns with Missing Values")
missing = df.isnull().sum().sort_values(ascending=False).head(15)
missing = missing[missing > 0]
if len(missing) > 0:
    fig9, ax9 = plt.subplots(figsize=(10, 4))
    bars9 = ax9.bar(missing.index, missing.values, color="#e74c3c")
    ax9.bar_label(bars9, padding=3, fontsize=9)
    ax9.set_ylabel("Missing Count")
    ax9.set_title("Columns with Most Missing Values")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig9)
    plt.close(fig9)
else:
    st.success("✅ No missing values in the dataset!")
