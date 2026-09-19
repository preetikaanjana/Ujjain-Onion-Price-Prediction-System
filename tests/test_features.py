import pytest
import pandas as pd
from src.preprocessing import load_and_preprocess
from src.feature_engineering import create_features, get_feature_matrix, FEATURE_COLUMNS, TARGET_COLUMN


def test_feature_generation_and_columns():
    cleaned = load_and_preprocess(save_processed=False)
    feat_df = create_features(cleaned, drop_na=True)
    
    for col in FEATURE_COLUMNS:
        assert col in feat_df.columns, f"Missing feature column: {col}"
    assert TARGET_COLUMN in feat_df.columns
    assert feat_df[FEATURE_COLUMNS].isnull().sum().sum() == 0, "Feature matrix must not have NaN values"


def test_no_data_leakage():
    cleaned = load_and_preprocess(save_processed=False)
    feat_df = create_features(cleaned, drop_na=True)
    
    # price_lag_1 at row index 1 must equal modal_price at row index 0
    assert feat_df['price_lag_1'].iloc[1] == feat_df[TARGET_COLUMN].iloc[0]
