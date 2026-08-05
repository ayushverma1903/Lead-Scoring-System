import streamlit as st
import joblib
import shap
import pandas as pd
import numpy as np
import sys, os

# Ensure src/ is importable
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
DATA_PATH     = "data/raw/Lead Scoring.csv"

@st.cache_resource(show_spinner="Loading model & computing SHAP values…")
def compute_shap():
    model    = joblib.load(MODEL_PATH)
    scaler   = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)

    # Load raw data and run the FULL preprocessing pipeline
    raw_df   = pd.read_csv(DATA_PATH)
    # Drop target column if present
    raw_df   = raw_df.drop(columns=['Converted'], errors='ignore')
    processed = full_preprocess_pipeline(raw_df, features)

    # Sample for speed
    sample    = processed.sample(min(300, len(processed)), random_state=42)
    X_scaled  = scaler.transform(sample)

    explainer   = shap.LinearExplainer(model, X_scaled)
    shap_values = explainer.shap_values(X_scaled)
    return shap_values, X_scaled, features

missing = not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, FEATURES_PATH, DATA_PATH])

if missing:
    st.warning("Model files or raw data not found. Please run the training notebooks first.")
else:
    try:
        shap_values, X_scaled, features = compute_shap()

        # ── Global Feature Importance ──
        st.subheader("Global Feature Importance")
        st.markdown("Mean absolute SHAP value — how much each feature moves the prediction on average.")

        mean_abs = np.abs(shap_values).mean(axis=0)
        importance_df = (
            pd.DataFrame({"Feature": features, "Mean |SHAP|": mean_abs})
            .sort_values("Mean |SHAP|", ascending=True)
            .tail(15)
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(importance_df["Feature"], importance_df["Mean |SHAP|"], color="#4F81BD")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Top-15 Feature Importances")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ── SHAP Beeswarm / Summary Plot ──
        st.subheader("SHAP Summary Plot (Beeswarm)")
        st.markdown("Each dot is one sample. **Red = high feature value**, **Blue = low**. Rightward = pushes toward conversion.")

        shap.summary_plot(shap_values, X_scaled,
                          feature_names=features,
                          max_display=15,
                          show=False)
        st.pyplot(plt.gcf())
        plt.close("all")

        st.info("Features like **Total Time Spent on Website** and **TotalVisits** typically have the largest impact on lead conversion probability.")

    except Exception as e:
        st.error(f"Could not compute SHAP values: {e}")
