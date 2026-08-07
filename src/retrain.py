"""
retrain.py
----------
Automated script for model retraining.
1. Checks for new data in data/new_data/
2. Combines with old data or just uses new data (based on logic).
3. Preprocesses data.
4. Trains a new model.
5. Evaluates and replaces old model if better.
"""

import os
import glob
import shutil
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from src.preprocess import full_preprocess_pipeline
from src.logger import get_logger

logger = get_logger(__name__)

NEW_DATA_DIR = "data/new_data"
MODEL_PATH = "models/lead_scoring_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/feature_columns.pkl"
ARCHIVE_DIR = "models/archive"

# Minimum number of samples required for retraining
MIN_SAMPLES = 20


def load_new_data():
    files = glob.glob(os.path.join(NEW_DATA_DIR, "*.csv"))
    if not files:
        return None
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Could not read {f}: {e}")
    
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def evaluate_model(model, scaler, feature_columns, X_test, y_test):
    # Already processed X_test
    X_test_scaled = scaler.transform(X_test)
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, probs)
    except ValueError:
        # roc_auc_score fails if only one class is present in y_test
        auc = 0.0
        logger.warning("ROC AUC could not be computed (only one class present in test set).")
    return acc, auc


def retrain():
    """
    Run the retraining pipeline.
    
    Returns a dict with keys:
        - 'status': 'success' | 'no_data' | 'error' | 'not_improved'
        - 'message': human-readable description
        - 'metrics': dict with 'accuracy' and 'auc' (only on success)
    """
    logger.info("Starting retraining process...")
    
    try:
        new_data = load_new_data()
    except Exception as e:
        msg = f"Failed to load new data: {e}"
        logger.error(msg)
        return {"status": "error", "message": msg}
    
    if new_data is None or len(new_data) == 0:
        msg = "No new data found in data/new_data/ for retraining."
        logger.info(msg)
        return {"status": "no_data", "message": msg}
    
    if 'Converted' not in new_data.columns:
        msg = "New data is missing the target column 'Converted'. Cannot retrain."
        logger.error(msg)
        return {"status": "error", "message": msg}
    
    if len(new_data) < MIN_SAMPLES:
        msg = (
            f"New data has only {len(new_data)} rows. "
            f"At least {MIN_SAMPLES} are required for retraining."
        )
        logger.error(msg)
        return {"status": "error", "message": msg}
        
    logger.info(f"Loaded {len(new_data)} new records.")
    
    try:
        y = new_data['Converted']
        X_raw = new_data.drop(columns=['Converted'])
        
        # Check that both classes are present
        if y.nunique() < 2:
            msg = "New data contains only one class. Need both converted and non-converted leads."
            logger.error(msg)
            return {"status": "error", "message": msg}
        
        # Load old feature columns to keep consistency
        if not os.path.exists(FEATURES_PATH):
            msg = f"Feature columns file not found at {FEATURES_PATH}."
            logger.error(msg)
            return {"status": "error", "message": msg}
        
        old_features = joblib.load(FEATURES_PATH)
        
        X = full_preprocess_pipeline(X_raw, old_features)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        acc, auc = evaluate_model(model, scaler, old_features, X_test, y_test)
        logger.info(f"New Model - Accuracy: {acc:.4f}, AUC: {auc:.4f}")
        
        # Evaluate old model on new data for comparison
        try:
            old_model = joblib.load(MODEL_PATH)
            old_scaler = joblib.load(SCALER_PATH)
            old_acc, old_auc = evaluate_model(old_model, old_scaler, old_features, X_test, y_test)
            logger.info(f"Old Model - Accuracy: {old_acc:.4f}, AUC: {old_auc:.4f}")
            
            if auc <= old_auc:
                msg = (
                    f"New model (AUC={auc:.4f}) did not outperform "
                    f"old model (AUC={old_auc:.4f}). Keeping old model."
                )
                logger.info(msg)
                # Clear new data
                for f in glob.glob(os.path.join(NEW_DATA_DIR, "*.csv")):
                    os.remove(f)
                return {"status": "not_improved", "message": msg}
        except Exception as e:
            logger.warning(f"Could not load old model for comparison: {e}")
            logger.info("Proceeding to save new model anyway.")
        
        # Save new model
        logger.info("Saving new model...")
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        if os.path.exists(MODEL_PATH):
            shutil.move(MODEL_PATH, os.path.join(ARCHIVE_DIR, "lead_scoring_model_old.pkl"))
        if os.path.exists(SCALER_PATH):
            shutil.move(SCALER_PATH, os.path.join(ARCHIVE_DIR, "scaler_old.pkl"))
            
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        # feature columns remain same
        
        # Clear new data
        for f in glob.glob(os.path.join(NEW_DATA_DIR, "*.csv")):
            os.remove(f)
        
        msg = f"Retraining successful! New model — Accuracy: {acc:.4f}, AUC: {auc:.4f}"
        logger.info(msg)
        return {
            "status": "success",
            "message": msg,
            "metrics": {"accuracy": round(acc, 4), "auc": round(auc, 4)},
        }
        
    except Exception as e:
        msg = f"Retraining failed: {str(e)}"
        logger.error(msg)
        return {"status": "error", "message": msg}


if __name__ == "__main__":
    result = retrain()
    print(result["message"])
