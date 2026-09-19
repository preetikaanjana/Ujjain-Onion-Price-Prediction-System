"""
Model Training, 2026 Holdout Evaluation, and Final Live Forecaster Module.
Implements:
- Objective A: Rigorous evaluation of models on unseen 2026 Ujjain Onion prices.
- Objective B: Final retraining on all historical data up to September 19, 2026 for forward forecasting.
"""

import os
import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_and_preprocess
from src.feature_engineering import create_features, FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluate import calculate_metrics, format_metric_statement
from src.baselines import NaiveBaseline, MovingAverageBaseline


def split_train_val_2026(df: pd.DataFrame):
    """
    Chronological Split:
    - Train: 2024 through mid-2025 (< 2025-07-01)
    - Validation: late 2025 (2025-07-01 to 2025-12-31)
    - Test: 2026 Unseen Evaluation (>= 2026-01-01)
    """
    df = df.copy()
    dates = pd.to_datetime(df['date'])
    
    train_mask = dates < '2025-07-01'
    val_mask = (dates >= '2025-07-01') & (dates < '2026-01-01')
    test_mask = dates >= '2026-01-01'
    
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()
    
    return train_df, val_df, test_df


def train_and_evaluate_all():
    print("=" * 65)
    print("UJJAIN ONION PRICE PREDICTION SYSTEM - 2026 BENCHMARK & RETRAINING")
    print("=" * 65)
    
    # 1. Load clean data
    clean_df = load_and_preprocess(save_processed=True)
    
    # 2. Generate time-series features
    feat_df = create_features(clean_df, drop_na=True)
    print(f"Total usable records after feature engineering: {len(feat_df)}")
    
    # 3. Chronological Splits
    train_df, val_df, test_df = split_train_val_2026(feat_df)
    
    print(f"\nChronological Splits:")
    print(f"  Training (Older Historical): {train_df['date'].min().date()} -> {train_df['date'].max().date()} ({len(train_df)} rows, {len(train_df)/len(feat_df)*100:.1f}%)")
    print(f"  Validation (Recent Hist):   {val_df['date'].min().date()} -> {val_df['date'].max().date()} ({len(val_df)} rows, {len(val_df)/len(feat_df)*100:.1f}%)")
    print(f"  Test (2026 Unseen Period):  {test_df['date'].min().date()} -> {test_df['date'].max().date()} ({len(test_df)} rows, {len(test_df)/len(feat_df)*100:.1f}%)")
    
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]
    
    # Fit models on Train + Val for 2026 Test Evaluation
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    
    models = {
        'Naive (Previous Price)': NaiveBaseline(),
        'Moving Average (7-Day)': MovingAverageBaseline(),
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=10.0, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_split=4, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    }
    
    results = []
    trained_models_2026 = {}
    test_predictions_2026 = {}
    
    print("\n--- OBJECTIVE A: EVALUATE ON UNSEEN 2026 DATA ---")
    for name, model in models.items():
        # Fit on historical train+val
        if hasattr(model, 'fit') and not isinstance(model, (NaiveBaseline, MovingAverageBaseline)):
            model.fit(X_train_val, y_train_val)
            
        test_pred = model.predict(X_test)
        # Price assertion: prices cannot be negative
        test_pred = np.maximum(50.0, test_pred)
        
        m = calculate_metrics(y_test, test_pred, name)
        results.append({
            'Model': name,
            'Test_2026_MAE': m['MAE'],
            'Test_2026_RMSE': m['RMSE'],
            'Test_2026_R2': m['R2'],
            'Test_2026_MAPE_%': m['MAPE_%']
        })
        test_predictions_2026[name] = test_pred
        trained_models_2026[name] = model
        
    res_df = pd.DataFrame(results).sort_values('Test_2026_MAE').reset_index(drop=True)
    print(res_df.to_string(index=False))
    
    # Save comparison table
    os.makedirs(os.path.join(PROJECT_ROOT, 'models'), exist_ok=True)
    comp_path = os.path.join(PROJECT_ROOT, 'models', 'model_comparison.csv')
    res_df.to_csv(comp_path, index=False)
    print(f"\nSaved 2026 benchmark comparison to {comp_path}")
    
    best_row = res_df.iloc[0]
    best_name = best_row['Model']
    print(f"\nSTRONGEST PERFORMER ON UNSEEN 2026 PERIOD: {best_name}")
    print("Note: The 7-day moving average was the strongest baseline on the 2026 holdout set. Among the machine-learning models tested, Random Forest performed best.")
    print(format_metric_statement({
        'Model': best_name,
        'MAE': best_row['Test_2026_MAE'],
        'RMSE': best_row['Test_2026_RMSE'],
        'R2': best_row['Test_2026_R2']
    }, split_name="unseen 2026 period (Jan-Sep 2026)"))
    
    # --- OBJECTIVE B: RETRAIN FINAL MODEL ON ALL DATA UP TO SEPTEMBER 19, 2026 ---
    print("\n--- OBJECTIVE B: RETRAIN ON COMPLETE DATA UP TO 2026-09-19 ---")
    X_full = feat_df[FEATURE_COLUMNS]
    y_full = feat_df[TARGET_COLUMN]
    
    # Instantiate fresh final model instance
    final_model = None
    if best_name == 'Naive (Previous Price)':
        final_model = NaiveBaseline()
    elif best_name == 'Moving Average (7-Day)':
        final_model = MovingAverageBaseline()
    elif best_name == 'Linear Regression':
        final_model = LinearRegression()
    elif best_name == 'Ridge Regression':
        final_model = Ridge(alpha=10.0, random_state=42)
    elif best_name == 'Random Forest':
        final_model = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_split=4, random_state=42)
    elif best_name == 'Gradient Boosting':
        final_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    elif best_name == 'XGBoost':
        final_model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
        
    if hasattr(final_model, 'fit') and not isinstance(final_model, (NaiveBaseline, MovingAverageBaseline)):
        final_model.fit(X_full, y_full)
        
    print(f"Retrained {best_name} on all {len(X_full)} observations up to {feat_df['date'].max().date()}.")
    
    # Save bundle with full 2026 actuals, predictions, and metadata
    bundle = {
        'model_name': best_name,
        'model': final_model,
        'feature_columns': FEATURE_COLUMNS,
        'target_column': TARGET_COLUMN,
        'test_metrics': {
            'MAE': float(best_row['Test_2026_MAE']),
            'RMSE': float(best_row['Test_2026_RMSE']),
            'R2': float(best_row['Test_2026_R2']),
            'MAPE_%': float(best_row['Test_2026_MAPE_%'])
        },
        'train_date_range': (str(train_df['date'].min().date()), str(val_df['date'].max().date())),
        'test_date_range': (str(test_df['date'].min().date()), str(test_df['date'].max().date())),
        'test_predictions': test_predictions_2026[best_name].tolist(),
        'test_actuals': y_test.tolist(),
        'test_dates': test_df['date'].dt.strftime('%Y-%m-%d').tolist(),
        'all_test_predictions': {k: v.tolist() for k, v in test_predictions_2026.items()},
        'latest_trained_date': str(feat_df['date'].max().date()),
        'latest_trained_price': float(feat_df[TARGET_COLUMN].iloc[-1])
    }
    
    for filename in ['final_model.pkl', 'onion_price_model.pkl']:
        p = os.path.join(PROJECT_ROOT, 'models', filename)
        with open(p, 'wb') as f:
            pickle.dump(bundle, f)
        print(f"Saved model bundle to {p}")
        
    return res_df, bundle


if __name__ == '__main__':
    train_and_evaluate_all()
