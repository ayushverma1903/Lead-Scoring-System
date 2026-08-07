import streamlit as st
import joblib
import shap
import pandas as pd
import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocess import full_preprocess_pipeline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="SHAP Explainability", page_icon="🧠", layout="wide")
st.title("🧠 SHAP Explainability")

st.markdown("""
Understanding **why** the model makes certain predictions is crucial for building trust. 
This page uses SHAP (SHapley Additive exPlanations) values to explain the model's behavior.
""")

MODEL_PATH    = "models/lead_scoring_model.pkl"
SCALER_PATH   = "models/scaler.pkl"
FEATURES_PATH = "models/feature_columns.pkl"
DEFAULT_DATA_PATH = "data/raw/Lead Scoring.csv"

# ── Load data from session_state ──────────────────────────────────────────────
if "user_df" not in st.session_state:
    if os.path.exists(DEFAULT_DATA_PATH):
        st.session_state["user_df"] = pd.read_csv(DEFAULT_DATA_PATH)
        st.session_state["data_source"] = "📁 Default: Lead Scoring.csv"

if "user_df" not in st.session_state:
    st.warning("No dataset loaded. Please go to the **Home** page and upload a dataset.")
    st.stop()

if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]):
    st.warning("Model files not found. Please run the training notebooks first.")
    st.stop()

df = st.session_state["user_df"]
source = st.session_state.get("data_source", "")
st.caption(f"**Data source:** {source} — SHAP computed on a sample of {min(300, len(df))} leads")

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    model    = joblib.load(MODEL_PATH)
    scaler   = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, scaler, features

def compute_shap(model, scaler, features, raw_df):
    # Drop target column if present
    raw_df = raw_df.drop(columns=["Converted"], errors="ignore")
    processed = full_preprocess_pipeline(raw_df, features)

    # Sample for speed
    sample   = processed.sample(min(300, len(processed)), random_state=42)
    X_scaled = scaler.transform(sample)

    explainer   = shap.LinearExplainer(model, X_scaled)
    shap_values = explainer.shap_values(X_scaled)
    return shap_values, X_scaled, features

try:
    model, scaler, features = load_model()

    with st.spinner("Computing SHAP values on current dataset…"):
        shap_values, X_scaled, feat_cols = compute_shap(model, scaler, features, df.copy())

    # ── Global Feature Importance ──────────────────────────────────────────────
    st.subheader("Global Feature Importance")
    st.markdown("Mean absolute SHAP value — how much each feature moves the prediction on average.")

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = (
        pd.DataFrame({"Feature": feat_cols, "Mean |SHAP|": mean_abs})
        .sort_values("Mean |SHAP|", ascending=True)
        .tail(15)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(importance_df["Feature"], importance_df["Mean |SHAP|"], color="#4F81BD")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top-15 Feature Importances (Current Dataset)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # ── SHAP Beeswarm / Summary Plot ──────────────────────────────────────────
    st.subheader("SHAP Summary Plot (Beeswarm)")
    st.markdown("Each dot is one sample. **Red = high feature value**, **Blue = low**. Rightward = pushes toward conversion.")

    shap.summary_plot(shap_values, X_scaled,
                      feature_names=feat_cols,
                      max_display=15,
                      show=False)
    st.pyplot(plt.gcf())
    plt.close("all")

    st.divider()

    # ── Top 3 features bar breakdown ──────────────────────────────────────────
    st.subheader("Top 3 Most Influential Features")
    top3 = importance_df.tail(3)
    c1, c2, c3 = st.columns(3)
    for col, (_, row) in zip([c1, c2, c3], top3[::-1].iterrows()):
        col.metric(row["Feature"], f"{row['Mean |SHAP|']:.4f}")

    st.info("Features like **Total Time Spent on Website** and **TotalVisits** typically have the largest impact on lead conversion probability.")

except Exception as e:
    st.error(f"Could not compute SHAP values: {e}")
