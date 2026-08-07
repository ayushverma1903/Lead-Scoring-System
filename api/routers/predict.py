from fastapi import APIRouter, HTTPException, Request
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from api.schemas import RawLead, RawLeadsBatch
from src.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


def _sanitize_for_json(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN and Inf values with None for valid JSON serialization."""
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.where(pd.notnull(df), other=None)
    return df


@router.post("/predict")
def predict_single_lead(lead: RawLead, request: Request):
    """Score a single raw lead and return its conversion probability."""
    scorer = request.app.state.scorer
    if scorer is None:
        logger.error("Predict endpoint called but model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        df = pd.DataFrame([lead.data])
        report = scorer.score_raw_leads(df)
        report = _sanitize_for_json(report)
        result = report.iloc[0].to_dict()
        logger.info(f"Single prediction successful. Result: {result.get('Priority')}")
        return result
    except Exception as e:
        logger.error(f"Error during single prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch_predict")
def predict_batch_leads(batch: RawLeadsBatch, request: Request):
    """Score multiple raw leads at once and return a ranked report."""
    scorer = request.app.state.scorer
    if scorer is None:
        logger.error("Batch predict endpoint called but model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if len(batch.leads) == 0:
        raise HTTPException(status_code=400, detail="No leads provided")

    try:
        df = pd.DataFrame([{k: v for k, v in lead.items() if v is not None} for lead in batch.leads])
        report = scorer.score_raw_leads(df)
        # Replace NaN/Inf with None so the JSON response is always valid
        report = _sanitize_for_json(report)
        logger.info(f"Batch prediction successful for {len(batch.leads)} leads.")
        return report.reset_index(drop=True).to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
