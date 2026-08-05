"""
test_pipeline.py
----------------
Tests for the preprocessing pipeline.
"""

import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import clean_raw_lead_data, engineer_features, drop_leakage_columns, full_preprocess_pipeline

def test_clean_raw_lead_data():
    raw_df = pd.DataFrame([
        {"Prospect ID": "123", "Lead Source": "Google", "TotalVisits": None, "City": None},
        {"Prospect ID": "456", "Lead Source": "Select", "TotalVisits": 5, "City": "Mumbai"}
    ])
    
    cleaned_df = clean_raw_lead_data(raw_df)
    
    assert "Prospect ID" not in cleaned_df.columns
    assert cleaned_df.loc[0, "City"] == "Unknown"
    assert not pd.isna(cleaned_df.loc[0, "TotalVisits"]) # should be imputed with median
    assert pd.isna(cleaned_df.loc[1, "Lead Source"]) # "Select" replaced with nan

def test_engineer_features():
    df = pd.DataFrame([
        {"Do Not Email": "Yes", "Tags": "Ringing"},
        {"Do Not Email": "No", "Tags": "Will revert after reading the email"}
    ])
    
    engineered_df = engineer_features(df)
    
    assert engineered_df.loc[0, "Do Not Email"] == 1
    assert engineered_df.loc[1, "Do Not Email"] == 0
    
def test_full_pipeline():
    raw_df = pd.DataFrame([{
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
    feature_columns = ["Total Time Spent on Website", "TotalVisits", "Lead Origin_Landing Page Submission"]
    processed = full_preprocess_pipeline(raw_df, feature_columns)
    
    assert list(processed.columns) == feature_columns
    assert len(processed) == 1
