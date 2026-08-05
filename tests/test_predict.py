"""
test_predict.py
----------------
Basic tests for the Lead Scoring pipeline.
Run with: pytest tests/
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from src.predict import LeadScorer

MODEL_PATH = "models/lead_scoring_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/feature_columns.pkl"


@pytest.fixture(scope="module")
def scorer():
    return LeadScorer(MODEL_PATH, SCALER_PATH, FEATURES_PATH)


def test_model_loads(scorer):
    assert scorer.model is not None
    assert scorer.scaler is not None
    assert len(scorer.feature_columns) == 30


def test_predict_returns_valid_probability(scorer):
    # Build a minimal fake lead using the model's expected feature columns
    fake_lead = pd.DataFrame([{col: 0 for col in scorer.feature_columns}])
    probabilities, predictions = scorer.predict(fake_lead)

    assert 0.0 <= probabilities[0] <= 1.0
    assert predictions[0] in [0, 1]


def test_score_raw_leads_returns_report(scorer):
    raw_lead = pd.DataFrame([{
        "TotalVisits": 5,
        "Total Time Spent on Website": 800,
        "Page Views Per Visit": 2.5,
        "Lead Origin": "Landing Page Submission",
        "Lead Source": "Google",
        "Last Activity": "Email Opened",
        "Country": "India",
        "Specialization": "Unknown",
        "What is your current occupation": "Working Professional",
        "City": "Mumbai",
        "Do Not Email": "No",
        "Do Not Call": "No",
    }])

    report = scorer.score_raw_leads(raw_lead)

    assert "Conversion_Probability" in report.columns
    assert "Priority" in report.columns
    assert report.iloc[0]["Priority"] in ["Hot", "Warm", "Cold"]
