"""
Baseline Forecasting Estimators for Ujjain Mandi Price Prediction.
Defined here to ensure consistent pickling and unpickling across modules.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class NaiveBaseline(BaseEstimator, RegressorMixin):
    """Predicts the previous observed trading price: y_hat_t = y_{t-1}"""
    def fit(self, X, y=None):
        return self
        
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            return X['price_lag_1'].values
        return np.asarray(X)[:, 0]


class MovingAverageBaseline(BaseEstimator, RegressorMixin):
    """Predicts the trailing 7-period rolling average: y_hat_t = rolling_mean_7"""
    def fit(self, X, y=None):
        return self
        
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            return X['rolling_mean_7'].values
        # Column 5 is rolling_mean_7
        return np.asarray(X)[:, 5]
