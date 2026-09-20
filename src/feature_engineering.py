"""
Feature Engineering Module for Ujjain Onion Price Prediction System.
Generates comprehensive time-series features (Lags 1-30, Rolling Statistics 7-30,
Calendar Signals, and Momentum Indicators) with strict zero-leakage guarantees.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


FEATURE_COLUMNS = [
    'price_lag_1',
    'price_lag_2',
    'price_lag_3',
    'price_lag_7',
    'price_lag_14',
    'price_lag_30',
    'rolling_mean_7',
    'rolling_mean_14',
    'rolling_mean_30',
    'rolling_std_7',
    'rolling_std_14',
    'rolling_std_30',
    'price_change_1d',
    'price_change_7d',
    'price_change_30d',
    'day_of_week',
    'day_of_month',
    'month',
    'quarter',
    'year'
]

TARGET_COLUMN = 'modal_price'


def create_features(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """
    Generate strictly backward-looking time-series features.
    
    ZERO DATA LEAKAGE GUARANTEE:
    - All lags are shifted by k >= 1.
    - All rolling calculations are strictly shifted by 1 before windowing.
    - No current or future modal price is part of any feature.
    """
    df = df.copy()
    if 'date' in df.columns and not np.issubdtype(df['date'].dtype, np.datetime64):
        df['date'] = pd.to_datetime(df['date'])
        
    df = df.sort_values('date').reset_index(drop=True)
    target = df[TARGET_COLUMN]
    past_series = target.shift(1)
    
    # 1. Lag features
    df['price_lag_1'] = target.shift(1)
    df['price_lag_2'] = target.shift(2)
    df['price_lag_3'] = target.shift(3)
    df['price_lag_7'] = target.shift(7)
    df['price_lag_14'] = target.shift(14)
    df['price_lag_30'] = target.shift(30)
    
    # 2. Rolling window features (computed on past_series only)
    df['rolling_mean_7'] = past_series.rolling(window=7, min_periods=3).mean()
    df['rolling_mean_14'] = past_series.rolling(window=14, min_periods=7).mean()
    df['rolling_mean_30'] = past_series.rolling(window=30, min_periods=14).mean()
    
    df['rolling_std_7'] = past_series.rolling(window=7, min_periods=3).std().fillna(0)
    df['rolling_std_14'] = past_series.rolling(window=14, min_periods=7).std().fillna(0)
    df['rolling_std_30'] = past_series.rolling(window=30, min_periods=14).std().fillna(0)
    
    # 3. Momentum / Trend features
    df['price_change_1d'] = df['price_lag_1'] - df['price_lag_2']
    df['price_change_7d'] = df['price_lag_1'] - df['price_lag_7']
    df['price_change_30d'] = df['price_lag_1'] - df['price_lag_30']
    
    # 4. Calendar features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    
    if drop_na:
        df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
        
    return df


def get_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Extract feature matrix X, target vector y, and column names."""
    feat_df = create_features(df, drop_na=True)
    X = feat_df[FEATURE_COLUMNS]
    y = feat_df[TARGET_COLUMN]
    return X, y, FEATURE_COLUMNS


if __name__ == '__main__':
    from src.preprocessing import load_and_preprocess
    clean_df = load_and_preprocess(save_processed=False)
    feat_df = create_features(clean_df)
    X, y, cols = get_feature_matrix(clean_df)
    
    out_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'ujjain_onion_features.csv')
    feat_df.to_csv(out_path, index=False)
    
    print("Feature Engineering Verification:")
    print(f"  Input daily rows: {len(clean_df)}")
    print(f"  Rows after 30-period lag warm-up: {len(feat_df)}")
    print(f"  Feature count: {X.shape[1]}")
    print(f"  Feature columns: {cols}")
    print("  Zero NaN values in features:", X.isnull().sum().sum() == 0)
    print(f"  Saved full 20-feature dataset to: {out_path}")
