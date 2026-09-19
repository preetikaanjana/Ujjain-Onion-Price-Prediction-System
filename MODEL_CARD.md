# Model Card: Ujjain Onion Price Prediction System (2026)

## 1. Model Overview
- **Model Name:** Ujjain Onion Price Forecaster (2026 Edition)
- **Version:** 2.0.0
- **Model Type:** Time-Series Autoregressive Baselines & Machine Learning Regression Models
- **Primary Target Variable:** Wholesale `Modal_Price` (Indian Rupees per Quintal, ₹ / 100 kg)
- **Supported Forecast Horizons:** 1-Day, 7-Day, and 14-Day forward recursive forecasting
- **Data Coverage:** 2024 to September 19, 2026 (699 daily observations, 1,086 market price records)
- **Latest Training Date:** September 19, 2026
- **Developer:** Machine Learning Engineering Portfolio

---

## 2. Intended Use
- **Intended Purpose:** Forward price forecasting for wholesale onion mandi prices at Ujjain APMC, Madhya Pradesh.
- **Out of Scope:** Guaranteed pricing contracts, financial derivatives arbitrage, or farmer net income claims.
- **Uncertainty Disclosure:** Forecast ranges are empirical error-based uncertainty ranges derived from historical forecasting error and are not statistically calibrated confidence intervals.

---

## 3. Training & Evaluation Methodology
- **Chronological Out-of-Sample Evaluation:**
  - **Train (Earlier Historical):** `2024-02-08` to `2025-06-30` (386 rows, 57.7%)
  - **Validation (Later 2025):** `2025-07-01` to `2025-12-31` (115 rows, 17.2%)
  - **Test (Unseen 2026 Period):** `2026-01-01` to `2026-09-19` (168 rows, 25.1%)
  - No random shuffling was used because random splitting of time-series observations can introduce temporal leakage.
- **Final Model Retraining:**
  - Final model retrained on all 669 usable observations up to `2026-09-19` for live forward projection.

---

## 4. Benchmark Performance on Unseen 2026 Test Period (Jan–Sep 2026)

| Model | Model Type | 2026 Test MAE (₹/Q) | 2026 Test RMSE (₹/Q) | 2026 Test R² | 2026 Test MAPE (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Moving Average (7-Day)** | Baseline | **₹ 232.67** | **₹ 322.72** | **0.8576** | **20.38%** |
| **Random Forest** | Machine Learning | ₹ 252.37 | ₹ 327.49 | 0.8534 | 25.75% |
| **Ridge Regression** | Machine Learning | ₹ 253.76 | ₹ 353.25 | 0.8294 | 21.21% |
| **Naive (Previous Price)** | Baseline | ₹ 258.20 | ₹ 368.39 | 0.8145 | 21.61% |
| **XGBoost** | Machine Learning | ₹ 259.19 | ₹ 336.22 | 0.8455 | 26.26% |
| **Linear Regression** | Machine Learning | ₹ 263.06 | ₹ 364.61 | 0.8183 | 21.65% |
| **Gradient Boosting** | Machine Learning | ₹ 267.49 | ₹ 338.41 | 0.8435 | 26.31% |

### Performance Insights:
- **Baseline vs. ML:** The 7-day moving average was the strongest baseline on the 2026 holdout set. Among the machine-learning models tested, Random Forest performed best.
- **R² Score:** An R² of 0.8576 means the model explains approximately 85.76% of the variance in the held-out 2026 target values relative to the standard R² baseline.
- **Price Dependence:** The historical price series showed strong short-term dependence, making recent prices useful predictors.

---

## 5. Input Features (Zero-Leakage Guarantee)
- 20 backward-looking features:
  - Lags: `price_lag_1`, `price_lag_2`, `price_lag_3`, `price_lag_7`, `price_lag_14`, `price_lag_30`
  - Rolling Stats: `rolling_mean_7`, `rolling_mean_14`, `rolling_mean_30`, `rolling_std_7`, `rolling_std_14`, `rolling_std_30`
  - Momentum: `price_change_1d`, `price_change_7d`, `price_change_30d`
  - Calendar: `day_of_week`, `day_of_month`, `month`, `quarter`, `year`
