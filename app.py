import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Lead Scoring System",
    page_icon="🎯",
    layout="wide",
)

# ── Global Dataset Loader (Sidebar) ────────────────────────────────────────────
DEFAULT_DATA_PATH = "data/raw/Lead Scoring.csv"
REQUIRED_COLS = [
    "TotalVisits", "Total Time Spent on Website", "Page Views Per Visit",
    "Lead Origin", "Lead Source", "Do Not Email"
]

with st.sidebar:
    st.header("📂 Dataset")
    st.markdown("Upload your own lead scoring CSV, or use the default training data.")

    uploaded = st.file_uploader("Upload Lead Scoring CSV", type=["csv"], key="global_uploader")

    if uploaded is not None:
        try:
            df_uploaded = pd.read_csv(uploaded)
            missing_cols = [c for c in REQUIRED_COLS if c not in df_uploaded.columns]
            if missing_cols:
                st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
            else:
                st.session_state["user_df"] = df_uploaded
                st.session_state["data_source"] = f"📤 Uploaded: {uploaded.name} ({len(df_uploaded):,} rows)"
                st.success(f"✅ {len(df_uploaded):,} leads loaded from upload!")
        except Exception as e:
            st.error(f"Could not read file: {e}")
    else:
        # Load default if not already loaded
        if "user_df" not in st.session_state:
            if os.path.exists(DEFAULT_DATA_PATH):
                st.session_state["user_df"] = pd.read_csv(DEFAULT_DATA_PATH)
                st.session_state["data_source"] = "📁 Default: Lead Scoring.csv"

    # Show current data source
    if "user_df" in st.session_state:
        df = st.session_state["user_df"]
        source = st.session_state.get("data_source", "")
        st.info(source)
        st.caption(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")

        if st.button("🔄 Reset to Default Dataset"):
            if "user_df" in st.session_state:
                del st.session_state["user_df"]
            if "data_source" in st.session_state:
                del st.session_state["data_source"]
            st.rerun()

# ── Main Home Page ──────────────────────────────────────────────────────────────
st.title("🎯 Lead Scoring System")
st.markdown("""
Welcome to the **Lead Scoring System**! 

This application predicts the conversion probability of leads for the LMS platform, helping the sales team prioritize their efforts on hot leads.

### 👈 Navigation
Use the sidebar to navigate through the application:
- **Dataset Overview:** View raw and processed data.
- **EDA Dashboard:** Interactive visualizations of the data.
- **Predict Lead:** Make single or batch predictions using the ML model.
- **Model Performance:** View evaluation metrics of the currently trained model.
- **SHAP Explainability:** Understand why the model makes certain predictions.
- **Business Insights:** High-level insights and KPI cards.
- **About:** Information about the project.

### Quick KPI
""")

# Live KPIs from current dataset
if "user_df" in st.session_state:
    df = st.session_state["user_df"]
    total = len(df)
    converted = int(df["Converted"].sum()) if "Converted" in df.columns else "N/A"
    conv_rate = f"{converted / total * 100:.1f}%" if isinstance(converted, int) else "N/A"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Leads", f"{total:,}")
    col2.metric("Converted Leads", f"{converted:,}" if isinstance(converted, int) else "N/A")
    col3.metric("Conversion Rate", conv_rate)
    col4.metric("Features", f"{df.shape[1]}")
else:
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Total Leads Processed", value="9,240")
    col2.metric(label="Overall Conversion Rate", value="38.5%")
    col3.metric(label="Model Accuracy", value="82.4%")

st.info("📤 **Upload your own dataset** from the sidebar, or use the default training data. All pages will update automatically!")
