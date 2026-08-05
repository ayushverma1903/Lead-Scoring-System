"""
main.py — Lead Scoring API
---------------------------
Run locally with:
    uvicorn api.main:app --reload --port 8000

Then visit http://127.0.0.1:8000/docs for interactive API docs.
"""

import sys
import os
from contextlib import asynccontextmanager

# Add parent directory to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from api.routers import predict, system
from src.predict import LeadScorer
from src.logger import get_logger

logger = get_logger(__name__)

MODEL_PATH = "models/lead_scoring_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/feature_columns.pkl"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    logger.info("Loading Lead Scoring Model...")
    try:
        app.state.scorer = LeadScorer(MODEL_PATH, SCALER_PATH, FEATURES_PATH)
        logger.info("✅ Lead scoring model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        app.state.scorer = None
    yield
    # Clean up resources if needed
    logger.info("Shutting down API...")

app = FastAPI(
    title="Lead Scoring API",
    description="Predicts the conversion probability of leads for the LMS platform.",
    version="1.0.0",
    lifespan=lifespan
)

# Include routers
app.include_router(system.router)
app.include_router(predict.router)

