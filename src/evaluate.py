"""
Evaluation Module for Ujjain Onion Price Prediction System.
Calculates continuous regression metrics: MAE, RMSE, and R-squared in Rs./Quintal.
Strictly avoids confusing R2 with 'accuracy'.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, Any]:
    """
    Compute standard regression evaluation metrics.
    
    Args:
        y_true: Actual modal prices (Rs./Quintal).
        y_pred: Predicted modal prices (Rs./Quintal).
        model_name: Identifier string for reporting.
        
    Returns:
        Dictionary with MAE (Rs./Q), RMSE (Rs./Q), and R2.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Also calculate mean absolute percentage error (MAPE) as auxiliary metric
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0
    
    return {
        'Model': model_name,
        'MAE': round(float(mae), 2),
        'RMSE': round(float(rmse), 2),
        'R2': round(float(r2), 4),
        'MAPE_%': round(float(mape), 2)
    }


def format_metric_statement(metrics: Dict[str, Any], split_name: str = "held-out test period") -> str:
    """
    Generate an honest, technically rigorous evaluation summary string.
    """
    model = metrics.get('Model', 'The model')
    mae = metrics.get('MAE', 0.0)
    rmse = metrics.get('RMSE', 0.0)
    r2 = metrics.get('R2', 0.0)
    
    statement = (
        f"{model} achieved an MAE of Rs. {mae}/quintal, an RMSE of Rs. {rmse}/quintal, "
        f"and an R? of {r2:.4f} on the {split_name}."
    )
    return statement


if __name__ == '__main__':
    # Test with sample numbers
    y_t = np.array([1200.0, 1350.0, 1500.0, 1420.0])
    y_p = np.array([1220.0, 1310.0, 1490.0, 1460.0])
    m = calculate_metrics(y_t, y_p, "Test-Model")
    print(m)
    print(format_metric_statement(m))
