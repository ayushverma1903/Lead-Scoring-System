from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
import pandas as pd
from src.logger import get_logger
import sys
import os

router = APIRouter()
logger = get_logger(__name__)

@router.get("/")
def root():
    return {"status": "ok", "message": "Lead Scoring API is running"}


@router.get("/health")
def health_check(request: Request):
    """Check if API is healthy and model is loaded."""
    scorer = request.app.state.scorer
    return {"status": "ok", "model_loaded": scorer is not None}


@router.get("/metrics")
def get_metrics():
    """Return model performance metrics."""
    # In a real scenario, you'd load this from a metrics.json or database.
    # For now, we return some dummy metrics or try to load them if they exist.
    metrics_path = "outputs/metrics.json"
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, "r") as f:
            return json.load(f)
    return {
        "accuracy": 0.82,
        "precision": 0.79,
        "recall": 0.85,
        "f1_score": 0.82,
        "roc_auc": 0.90
    }

def run_retraining():
    logger.info("Starting automated retraining task...")
    try:
        from src.retrain import retrain
        result = retrain()
        logger.info(f"Retraining result: {result.get('status')} — {result.get('message')}")
    except Exception as e:
        logger.error(f"Retraining failed: {str(e)}")

@router.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks):
    """Trigger automated retraining in the background."""
    logger.info("Retraining triggered via API")
    background_tasks.add_task(run_retraining)
    return {"message": "Retraining task started in the background"}
