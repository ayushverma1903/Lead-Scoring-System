"""
predict.py
----------
Reusable prediction pipeline for the Lead Scoring System.
Mirrors the logic built in 10_final_prediction.ipynb.

Usage:
    from src.predict import LeadScorer

    scorer = LeadScorer(
        model_path="models/lead_scoring_model.pkl",
        scaler_path="models/scaler.pkl",
        features_path="models/feature_columns.pkl"
    )

    # For data that is ALREADY in the engineered/top-30-feature format:
    prob, pred = scorer.predict(new_lead_df)

    # For RAW lead data (same columns as the original Kaggle CSV):
    report = scorer.score_raw_leads(raw_leads_df)
"""

import numpy as np
import pandas as pd
import joblib

from .preprocess import full_preprocess_pipeline


class LeadScorer:
    def __init__(self, model_path: str, scaler_path: str, features_path: str):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_columns = joblib.load(features_path)

    def predict(self, leads_df: pd.DataFrame):
        """
        Predict on data that ALREADY matches the model's expected
        feature columns (e.g. output of 06_feature_selection.ipynb).
        Returns (probabilities, predictions).
        """
        aligned = leads_df.reindex(columns=self.feature_columns, fill_value=0)
        scaled = self.scaler.transform(aligned)

        probabilities = self.model.predict_proba(scaled)[:, 1]
        predictions = self.model.predict(scaled)
        return probabilities, predictions

    def score_raw_leads(self, raw_leads_df: pd.DataFrame) -> pd.DataFrame:
        """
        Full pipeline: takes RAW lead data (same format as the original
        'Lead Scoring.csv') and returns a ranked scoring report.
        """
        processed = full_preprocess_pipeline(raw_leads_df, self.feature_columns)
        probabilities, predictions = self.predict(processed)

        report = pd.DataFrame(index=raw_leads_df.index)
        report['Conversion_Probability'] = (probabilities * 100).round(2)
        report['Prediction'] = np.where(predictions == 1, 'Will Convert', 'Will Not Convert')
        report['Priority'] = report['Conversion_Probability'].apply(self._priority_tier)

        return report.sort_values(by='Conversion_Probability', ascending=False)

    @staticmethod
    def _priority_tier(prob: float) -> str:
        if prob >= 70:
            return 'Hot'
        elif prob >= 40:
            return 'Warm'
        else:
            return 'Cold'
