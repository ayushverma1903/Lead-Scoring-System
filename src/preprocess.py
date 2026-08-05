"""
preprocess.py
--------------
Replicates the data cleaning + feature engineering pipeline built in:
  - 04_data_cleaning.ipynb
  - 05_feature_engineering.ipynb
  - 06_feature_selection.ipynb

Use this to transform RAW lead data (same columns as the original
Kaggle "Lead Scoring.csv") into the exact feature format the model expects.
"""

import numpy as np
import pandas as pd

# Columns dropped during cleaning (IDs + constant + high-missing columns)
DROP_COLS_CLEANING = [
    'Prospect ID', 'Lead Number', 'Magazine',
    'How did you hear about X Education',
    'Lead Profile',
    'Asymmetrique Activity Index',
    'Asymmetrique Profile Index',
    'Asymmetrique Activity Score',
    'Asymmetrique Profile Score'
]

# Categorical columns imputed with "Unknown" (30-70% missing)
FILL_UNKNOWN_COLS = [
    'Lead Quality', 'City', 'Specialization', 'Tags',
    'What matters most to you in choosing a course',
    'What is your current occupation', 'Country'
]

# Binary Yes/No columns converted to 0/1
BINARY_COLS = [
    'Do Not Email', 'Do Not Call', 'Search', 'Newspaper Article',
    'X Education Forums', 'Newspaper', 'Digital Advertisement',
    'Through Recommendations', 'Receive More Updates About Our Courses',
    'Update me on Supply Chain Content', 'Get updates on DM Content',
    'I agree to pay the amount through cheque',
    'A free copy of Mastering The Interview'
]

# High-cardinality columns that had rare categories grouped into "Other"
HIGH_CARDINALITY_COLS = [
    'Country', 'Tags', 'Lead Source', 'Specialization',
    'Last Activity', 'Last Notable Activity'
]

# Columns dropped for data-leakage reasons (post sales-contact info)
LEAKAGE_PREFIXES = ('Tags_', 'Lead Quality_', 'Last Activity_', 'Last Notable Activity_')


def clean_raw_lead_data(df: pd.DataFrame) -> pd.DataFrame:
    """Step 1: Mirrors 04_data_cleaning.ipynb"""
    df = df.copy()

    # Drop useless / high-missing columns (ignore ones that may not exist in new data)
    df.drop(columns=[c for c in DROP_COLS_CLEANING if c in df.columns], inplace=True, errors='ignore')

    # Treat literal "Select" as missing
    df.replace('Select', np.nan, inplace=True)

    # Impute categorical columns with "Unknown"
    for col in FILL_UNKNOWN_COLS:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    # Impute remaining numeric columns with median
    for col in ['TotalVisits', 'Page Views Per Visit']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Impute any remaining categorical columns with mode
    for col in df.select_dtypes(include='object').columns:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])

    return df


def engineer_features(df: pd.DataFrame, rare_category_threshold: float = 0.01) -> pd.DataFrame:
    """Step 2: Mirrors 05_feature_engineering.ipynb"""
    df = df.copy()

    # Convert binary Yes/No columns to 0/1
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Group rare categories into "Other" for high-cardinality columns
    for col in HIGH_CARDINALITY_COLS:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True)
            rare = freq[freq < rare_category_threshold].index
            df[col] = df[col].replace(rare, 'Other')

    # One-hot encode remaining categorical columns
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Convert any bool columns to int
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: Mirrors the leakage-column removal in 06_feature_selection.ipynb"""
    leakage_cols = [c for c in df.columns if c.startswith(LEAKAGE_PREFIXES)]
    return df.drop(columns=leakage_cols, errors='ignore')


def align_to_model_features(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """
    Final step: ensure the dataframe has EXACTLY the columns the model
    was trained on (top 30 features), in the same order.
    Missing columns are filled with 0; extra columns are dropped.
    """
    return df.reindex(columns=feature_columns, fill_value=0)


def full_preprocess_pipeline(raw_df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """
    Runs the complete pipeline end-to-end:
    raw lead data -> cleaned -> engineered -> leakage-free -> aligned to model features
    """
    df = clean_raw_lead_data(raw_df)
    df = engineer_features(df)
    df = drop_leakage_columns(df)
    df = align_to_model_features(df, feature_columns)
    return df
