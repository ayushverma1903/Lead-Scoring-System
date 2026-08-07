import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocess import full_preprocess_pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix
)

st.set_page_config(page_title="Model Performance", page_icon="📊", layout="wide")
st.title("📊 Model Performance Metrics")

MODEL_PATH    = "models/lead_scoring_model.pkl"
SCALER_PATH   = "models/scaler.pkl"
FEATURES_PATH = "models/feature_columns.pkl"
DEFAULT_DATA_PATH = "data/raw/Lead Scoring.csv"
METRICS_PATH  = "outputs/metrics.json"

if "user_df" not in st.session_state:
    if os.path.exists(DEFAULT_DATA_PATH):
        st.session_state["user_df"] = pd.read_csv(DEFAULT_DATA_PATH)
        st.session_state["data_source"] = "📁 Default: Lead Scoring.csv"

if "user_df" not in st.session_state:
    st.warning("No dataset loaded. Please go to the **Home** page and upload a dataset.")
    st.stop()

if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]):
    st.warning("Model files missing. Run the training notebooks first.")
    st.stop()

df = st.session_state["user_df"]
source = st.session_state.get("data_source", "")
has_converted = "Converted" in df.columns

st.caption(f"**Data source:** {source} — evaluating on {df.shape[0]:,} leads")

# ── Mode: With labels vs Without labels ───────────────────────────────────────
if has_converted:
    st.markdown("Live evaluation of the production model on the loaded dataset (has ground truth labels).")
else:
    st.info("ℹ️ No `Converted` column found — showing **prediction distribution only**. Upload a labelled dataset to see accuracy, ROC, and confusion matrix.")

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    model    = joblib.load(MODEL_PATH)
    scaler   = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, scaler, features

try:
    model, scaler, features = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ── Predict on the data ───────────────────────────────────────────────────────
try:
    with st.spinner("Running model predictions on current dataset…"):
        X_raw = df.drop(columns=["Converted"], errors="ignore")
        X     = full_preprocess_pipeline(X_raw, features)
        X_sc  = scaler.transform(X)

        y_pred = model.predict(X_sc)
        y_prob = model.predict_proba(X_sc)[:, 1]
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# MODE A: Has labels → full metrics
# ══════════════════════════════════════════════════════════════════════════════
if has_converted:
    y = df["Converted"].values

    metrics = {
        "accuracy":  round(accuracy_score(y, y_pred),  4),
        "precision": round(precision_score(y, y_pred), 4),
        "recall":    round(recall_score(y, y_pred),    4),
        "f1_score":  round(f1_score(y, y_pred),        4),
        "roc_auc":   round(roc_auc_score(y, y_prob),   4),
    }

    os.makedirs("outputs", exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f)

    fpr, tpr, _ = roc_curve(y, y_prob)
    cm = confusion_matrix(y, y_pred)

    # KPI strip
    st.subheader("Key Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{metrics['accuracy']*100:.1f}%")
    c2.metric("Precision", f"{metrics['precision']*100:.1f}%")
    c3.metric("Recall",    f"{metrics['recall']*100:.1f}%")
    c4.metric("F1 Score",  f"{metrics['f1_score']*100:.1f}%")
    c5.metric("ROC AUC",   f"{metrics['roc_auc']:.3f}")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("ROC Curve")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"AUC = {metrics['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_right:
        st.subheader("Confusion Matrix")
        fig2, ax2 = plt.subplots(figsize=(4, 3.5))
        im = ax2.imshow(cm, cmap="Blues")
        ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
        ax2.set_xticklabels(["Not Converted", "Converted"])
        ax2.set_yticklabels(["Not Converted", "Converted"])
        plt.setp(ax2.get_xticklabels(), rotation=15, ha="right")
        for i in range(2):
            for j in range(2):
                ax2.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black",
                         fontsize=14, fontweight="bold")
        ax2.set_title("Confusion Matrix")
        plt.colorbar(im, ax=ax2)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# COMMON: Prediction Score Distribution (always shown)
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Prediction Score Distribution")

# Summary KPIs
hot  = int((y_prob >= 0.70).sum())
warm = int(((y_prob >= 0.40) & (y_prob < 0.70)).sum())
cold = int((y_prob < 0.40).sum())

c1, c2, c3 = st.columns(3)
c1.metric("🔥 Hot Leads (≥70%)",   f"{hot:,}")
c2.metric("🌡️ Warm Leads (40-70%)", f"{warm:,}")
c3.metric("❄️ Cold Leads (<40%)",    f"{cold:,}")

fig3, ax3 = plt.subplots(figsize=(8, 3))
ax3.hist(y_prob, bins=50, color="#4C72B0", edgecolor="white", alpha=0.85)
ax3.axvline(0.5, color="red", linestyle="--", label="Decision threshold (0.5)")
ax3.axvline(0.7, color="orange", linestyle="--", label="Hot lead threshold (0.7)")
ax3.axvline(0.4, color="blue", linestyle="--", label="Warm lead threshold (0.4)")
ax3.set_xlabel("Predicted Probability of Conversion")
ax3.set_ylabel("Number of Leads")
ax3.set_title("Distribution of Predicted Scores")
ax3.legend(fontsize=8)
plt.tight_layout()
st.pyplot(fig3)
plt.close(fig3)

if has_converted:
    st.info("Metrics are computed live against the currently loaded dataset.")
else:
    st.info("Upload a labelled dataset (with `Converted` column) from the Home page sidebar to see full accuracy metrics, ROC curve, and confusion matrix.")
