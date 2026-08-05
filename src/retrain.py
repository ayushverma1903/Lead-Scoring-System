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

def load_new_data():
    files = glob.glob(os.path.join(NEW_DATA_DIR, "*.csv"))
    if not files:
        return None
    
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def evaluate_model(model, scaler, feature_columns, X_test, y_test):
    # Already processed X_test
    X_test_scaled = scaler.transform(X_test)
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    return acc, auc

def retrain():
    logger.info("Starting retraining process...")
    new_data = load_new_data()
    
    if new_data is None:
        logger.info("No new data found for retraining.")
        return
    
    if 'Converted' not in new_data.columns:
        logger.error("New data is missing the target column 'Converted'.")
        return
        
    logger.info(f"Loaded {len(new_data)} new records.")
    
    # In a real scenario, we might combine this with old data. 
    # For simplicity, we train on the new data provided.
    
    y = new_data['Converted']
    X_raw = new_data.drop(columns=['Converted'])
    
    # Load old feature columns to keep consistency
    old_features = joblib.load(FEATURES_PATH)
    
    X = full_preprocess_pipeline(X_raw, old_features)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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
            logger.info("New model did not outperform old model. Discarding.")
            # Clear new data
            for f in glob.glob(os.path.join(NEW_DATA_DIR, "*.csv")):
                os.remove(f)
            return
    except Exception as e:
        logger.warning(f"Could not load old model for comparison: {e}")
        logger.info("Proceeding to save new model anyway.")
    
    # Save new model
    logger.info("Saving new model...")
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH):
        os.rename(MODEL_PATH, os.path.join(ARCHIVE_DIR, "lead_scoring_model_old.pkl"))
    if os.path.exists(SCALER_PATH):
        os.rename(SCALER_PATH, os.path.join(ARCHIVE_DIR, "scaler_old.pkl"))
        
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    # feature columns remain same
    
    # Clear new data
    for f in glob.glob(os.path.join(NEW_DATA_DIR, "*.csv")):
        os.remove(f)
        
    logger.info("Retraining process completed successfully.")

if __name__ == "__main__":
    retrain()
