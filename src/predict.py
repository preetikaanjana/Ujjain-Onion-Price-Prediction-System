"""
Prediction and Live Forecasting Module for Ujjain Onion Price Prediction System.
Forecasts future wholesale onion mandi prices in 2026 (1 Day, 7 Days, 14 Days)
from the latest verified 2026 observation (September 19, 2026).
"""

import os
import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_and_preprocess
from src.feature_engineering import FEATURE_COLUMNS
from src.baselines import NaiveBaseline, MovingAverageBaseline


def load_model_bundle(model_path: str = None) -> Dict[str, Any]:
    """Load the final trained model bundle."""
    candidates = [
        model_path,
        os.path.join(PROJECT_ROOT, 'models', 'final_model.pkl'),
        os.path.join(PROJECT_ROOT, 'models', 'onion_price_model.pkl')
    ]
    for p in candidates:
        if p and os.path.exists(p):
            with open(p, 'rb') as f:
                return pickle.load(f)
    raise FileNotFoundError("Model bundle not found in models/ directory. Please run src/train.py first.")


def predict_future(horizon_days: int = 1, model_bundle: Dict[str, Any] = None, df_history: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Generate live future forecasts forward from the latest 2026 observation.
    
    Supports:
        horizon_days: 1, 7, or 14 days.
    """
    if horizon_days not in [1, 7, 14]:
        raise ValueError(f"Unsupported horizon {horizon_days}. Expected 1, 7, or 14.")
        
    bundle = model_bundle or load_model_bundle()
    model = bundle['model']
    rmse_2026 = bundle['test_metrics']['RMSE']
    
    if df_history is None:
        df_history = load_and_preprocess(save_processed=False)
        
    df = df_history.copy().sort_values('date').reset_index(drop=True)
    latest_known_date = pd.to_datetime(df['date'].iloc[-1])
    latest_known_price = float(df['modal_price'].iloc[-1])
    
    prices = list(df['modal_price'].values)
    dates = list(pd.to_datetime(df['date']).values)
    
    forecast_results = []
    current_date = latest_known_date
    
    for step in range(1, horizon_days + 1):
        current_date = current_date + pd.Timedelta(days=1)
        
        # Build backward-looking feature vector
        p_lag_1 = prices[-1]
        p_lag_2 = prices[-2] if len(prices) >= 2 else p_lag_1
        p_lag_3 = prices[-3] if len(prices) >= 3 else p_lag_2
        p_lag_7 = prices[-7] if len(prices) >= 7 else prices[0]
        p_lag_14 = prices[-14] if len(prices) >= 14 else prices[0]
        p_lag_30 = prices[-30] if len(prices) >= 30 else prices[0]
        
        rolling_7 = np.mean(prices[-7:]) if len(prices) >= 7 else np.mean(prices)
        rolling_14 = np.mean(prices[-14:]) if len(prices) >= 14 else np.mean(prices)
        rolling_30 = np.mean(prices[-30:]) if len(prices) >= 30 else np.mean(prices)
        
        std_7 = np.std(prices[-7:]) if len(prices) >= 7 else 0.0
        std_14 = np.std(prices[-14:]) if len(prices) >= 14 else 0.0
        std_30 = np.std(prices[-30:]) if len(prices) >= 30 else 0.0
        
        feat_dict = {
            'price_lag_1': [p_lag_1],
            'price_lag_2': [p_lag_2],
            'price_lag_3': [p_lag_3],
            'price_lag_7': [p_lag_7],
            'price_lag_14': [p_lag_14],
            'price_lag_30': [p_lag_30],
            'rolling_mean_7': [rolling_7],
            'rolling_mean_14': [rolling_14],
            'rolling_mean_30': [rolling_30],
            'rolling_std_7': [std_7],
            'rolling_std_14': [std_14],
            'rolling_std_30': [std_30],
            'price_change_1d': [p_lag_1 - p_lag_2],
            'price_change_7d': [p_lag_1 - p_lag_7],
            'price_change_30d': [p_lag_1 - p_lag_30],
            'day_of_week': [current_date.dayofweek],
            'day_of_month': [current_date.day],
            'month': [current_date.month],
            'quarter': [current_date.quarter],
            'year': [current_date.year]
        }
        X_step = pd.DataFrame(feat_dict)[FEATURE_COLUMNS]
        
        pred_val = float(model.predict(X_step)[0])
        # Assert non-negative price
        pred_val = max(100.0, pred_val)
        
        # Empirical error-based uncertainty ranges derived from historical RMSE (not statistically calibrated confidence intervals)
        margin = rmse_2026 * np.sqrt(step) * 0.95
        lower_bound = max(100.0, round(pred_val - margin, 2))
        upper_bound = round(pred_val + margin, 2)
        
        forecast_results.append({
            'step': step,
            'forecast_date': current_date.strftime('%Y-%m-%d'),
            'predicted_price': round(pred_val, 2),
            'estimated_lower': lower_bound,
            'estimated_upper': upper_bound
        })
        
        # Recursive step: feed prediction into trailing buffer
        prices.append(pred_val)
        dates.append(current_date)
        
    final_pred = forecast_results[-1]
    
    reliability_notes = {
        1: "1-day forward lookahead based on latest verified September 19, 2026 price data.",
        7: "7-day recursive forecast using intermediate predicted values.",
        14: "14-day recursive horizon with widening empirical error-based uncertainty ranges."
    }
    
    return {
        'horizon_days': horizon_days,
        'latest_actual_date': latest_known_date.strftime('%Y-%m-%d'),
        'latest_actual_price': latest_known_price,
        'target_date': final_pred['forecast_date'],
        'predicted_price': final_pred['predicted_price'],
        'price_delta': round(final_pred['predicted_price'] - latest_known_price, 2),
        'percentage_delta': round(((final_pred['predicted_price'] - latest_known_price) / latest_known_price) * 100.0, 2),
        'estimated_range': (final_pred['estimated_lower'], final_pred['estimated_upper']),
        'daily_forecasts': forecast_results,
        'model_name': bundle['model_name'],
        'reliability_disclosure': reliability_notes.get(horizon_days, "")
    }


if __name__ == '__main__':
    print("Testing live 2026 prediction module...")
    for h in [1, 7, 14]:
        res = predict_future(horizon_days=h)
        print(f"\n--- Live Forecast: {h} Day(s) Forward ---")
        print(f"Latest Actual 2026 Date:  {res['latest_actual_date']} (Rs. {res['latest_actual_price']:.2f}/Q)")
        print(f"Forecast Target Date:     {res['target_date']}")
        print(f"Predicted Price:          Rs. {res['predicted_price']:.2f}/Q")
        print(f"Estimated Price Range:    Rs. {res['estimated_range'][0]:.0f} - Rs. {res['estimated_range'][1]:.0f}/Q")
        print(f"Reliability Notice:       {res['reliability_disclosure']}")
