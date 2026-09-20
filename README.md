# Ujjain Onion Price Prediction System (2026)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![Source](https://img.shields.io/badge/Data%20Source-AGMARKNET%202.0%20API-green.svg)](https://agmarknet.gov.in/)
[![Dataset](https://img.shields.io/badge/Dataset-2024--2026%20Verified-orange.svg)](#3-data-source)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An empirical, end-to-end machine learning time-series forecasting application predicting wholesale onion mandi prices specifically for **Ujjain APMC, Madhya Pradesh, India**, using 100% verified historical Indian government records through **September 19, 2026**.

---

## 1. Project Overview
Agricultural price volatility in India severely impacts farmer revenues and market stability. Onion (*Allium cepa*) is especially price-volatile due to seasonal arrival gluts, storage loss, and rainfall disruptions. The **Ujjain Onion Price Prediction System** provides an end-to-end, leak-free machine learning forecasting solution that ingests authentic government mandi records, benchmarks multiple regression algorithms on an unseen 2026 test period, and provides live multi-horizon forecasts forward from September 19, 2026 with empirical error-based uncertainty ranges.

---

## 2. Motivation
- **Ujjain Focus:** Ujjain is a major commercial mandi in Madhya Pradesh's Malwa belt. Rather than generic national averages, this project models localized micro-market pricing.
- **Genuine 2026 Data:** Built directly on 1,086 official market price records (699 daily observations) from January 1, 2024 to September 19, 2026.
- **End-to-End ML Engineering:** Leakage-free feature pipelines, chronological out-of-sample holdout validation, and modular architecture.

---

## 3. Data Source
- **Origin:** Directorate of Marketing & Inspection (DMI), Ministry of Agriculture & Farmers Welfare, Government of India.
- **Portal & API:** [AGMARKNET 2.0](https://agmarknet.gov.in/) & [data.gov.in](https://data.gov.in/)
- **API Endpoint:** `https://api.agmarknet.gov.in/v1/prices-and-arrivals/date-wise/specific-commodity`
- **Market:** `Ujjain APMC` (District: Ujjain, State: Madhya Pradesh)
- **Commodity:** `Onion` (Variety: `Onion`)
- **Price Unit:** Indian Rupees per Quintal (₹ / 100 kg)

---

## 4. Data Authenticity
- **100% Real Records:** Zero fake rows, zero simulated prices, zero Faker.
- **Automated Validation:** Strict runtime assertion in `src/data_collection.py` halts with `ValueError` if zero verified Ujjain onion records exist.

---

## 5. Dataset Description
- **Total Market Records:** 1,086 market price records (699 unique daily trading sessions)
- **2026 Records:** 289 market price records across 168 active trading days
- **Date Range:** `2024-01-01` to `2026-09-19`
- **Missing Values:** 0 null values across all fields
- **Primary Target:** `modal_price` (₹ / Quintal)
- **Daily Aggregation:** Aggregates multiple available market-price records for the same trading day into a daily price series using the implemented aggregation rule (mean for modal price, min for min price, max for max price, sum for arrivals).

| Metric | Min Price (₹/Q) | Max Price (₹/Q) | Modal Price (₹/Q) | Arrivals (Tonnes) |
| :--- | :--- | :--- | :--- | :--- |
| **Mean** | ₹ 491.04 | ₹ 1,964.77 | ₹ 1,451.26 | 134.28 Tonnes |
| **Min** | ₹ 80.00 | ₹ 249.00 | ₹ 100.00 | 0.05 Tonnes |
| **Max** | ₹ 4,041.00 | ₹ 11,831.00 | ₹ 5,000.00 | 1,420.00 Tonnes |
| **Latest (19 Sep 2026)** | ₹ 1,377.18 | ₹ 3,549.96 | **₹ 3,549.96** | 69.82 Tonnes |

---

## 6. Exploratory Data Analysis (EDA)
Full notebooks available in `notebooks/`:
- `01_data_collection.ipynb`: API extraction and validation.
- `02_eda.ipynb`: Multi-year trends, 2026 price movements, seasonality, and volume arrivals.
- `03_feature_engineering.ipynb`: Lag creation and leakage prevention checks.
- `04_model_training.ipynb`: 2026 benchmark experiments.

---

## 7. Feature Engineering
Generated 20 backward-looking time-series features in `src/feature_engineering.py`:
- **Lags:** `price_lag_1`, `price_lag_2`, `price_lag_3`, `price_lag_7`, `price_lag_14`, `price_lag_30`
- **Rolling Statistics:** `rolling_mean_7`, `rolling_mean_14`, `rolling_mean_30`, `rolling_std_7`, `rolling_std_14`, `rolling_std_30`
- **Momentum:** `price_change_1d`, `price_change_7d`, `price_change_30d`
- **Calendar:** `day_of_week`, `day_of_month`, `month`, `quarter`, `year`
- **Zero Leakage:** All rolling and lag features are strictly shifted by at least 1 period before computation. No random shuffling was used because random splitting of time-series observations can introduce temporal leakage.

---

## 8. Models Evaluated

### Baseline Methods
1. **Naive Persistence:** $y_t = y_{t-1}$ (previous-day naive prediction)
2. **Moving Average (7-Day):** Trailing 7-session rolling mean

### Machine Learning Models
3. **Linear Regression:** Standard ordinary least squares (OLS)
4. **Ridge Regression:** L2 regularized linear regression ($\alpha=10$)
5. **Random Forest Regressor:** 100 trees, max depth 6
6. **Gradient Boosting Regressor:** 100 stages, learning rate 0.05
7. **XGBoost Regressor:** Gradient-boosted trees

---

## 9. Model Evaluation on Unseen 2026 Test Period
Models were trained on historical data (2024 to mid-2025, 386 rows), validated on late 2025 (115 rows), and evaluated on the **unseen 2026 period (Jan 1 to Sep 19, 2026, 168 trading days)**:

| Model | Model Type | 2026 Test MAE (₹/Q) | 2026 Test RMSE (₹/Q) | 2026 Test R² | 2026 Test MAPE (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Moving Average (7-Day)** | Baseline | **₹ 232.67** | **₹ 322.72** | **0.8576** | **20.38%** |
| **Random Forest** | Machine Learning | ₹ 252.37 | ₹ 327.49 | 0.8534 | 25.75% |
| **Ridge Regression** | Machine Learning | ₹ 253.76 | ₹ 353.25 | 0.8294 | 21.21% |
| **Naive (Previous Price)** | Baseline | ₹ 258.20 | ₹ 368.39 | 0.8145 | 21.61% |
| **XGBoost** | Machine Learning | ₹ 259.19 | ₹ 336.22 | 0.8455 | 26.26% |
| **Linear Regression** | Machine Learning | ₹ 263.06 | ₹ 364.61 | 0.8183 | 21.65% |
| **Gradient Boosting** | Machine Learning | ₹ 267.49 | ₹ 338.41 | 0.8435 | 26.31% |

### Key Benchmark Findings:
- **Baseline vs. ML:** The 7-day moving average was the strongest baseline on the 2026 holdout set. Among the machine-learning models tested, Random Forest performed best.
- **R² Interpretation:** An R² of 0.8576 means the model explains approximately 85.76% of the variance in the held-out 2026 target values relative to the standard R² baseline.
- **Uncertainty Intervals:** Forecast intervals are empirical error-based uncertainty ranges derived from historical forecasting error and are not statistically calibrated confidence intervals.
- **Price Dependence:** The historical price series showed strong short-term dependence, making recent prices useful predictors. The 7-day moving average provided the strongest performance among the baseline methods tested.

---

## 10. Deployment on Render

This project is configured specifically for deployment on **Render** as a Python Web Service.

### Option A: Automatic Blueprint Deployment
1. Connect your GitHub repository to [Render](https://render.com/).
2. Create a new **Blueprint** instance. Render will automatically detect [`render.yaml`](render.yaml) and configure all build and start parameters.

### Option B: Manual Web Service Setup
1. On your Render dashboard, click **New +** -> **Web Service**.
2. Select your repository.
3. Configure settings:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
   - **Plan:** `Free`
4. Click **Deploy Web Service**.

> **Note:** Streamlit Community Cloud deploy buttons, toolbars, and menus have been hidden via `.streamlit/config.toml` and CSS. The app runs cleanly on Render without external cloud popups.

---

## 11. Limitations
- Agricultural prices depend on monsoon rainfall, hail damage, interstate logistics, and export duties.
- Predictions are empirical statistical estimates, not financial guarantees.
- APMC auction modal prices do not reflect production costs and must never be cited as farmer net profit.
- Multi-step recursive forecasts (7 and 14 days) carry widening empirical uncertainty bands.

---

## 12. Future Improvements
1. Real-time satellite precipitation from IMD.
2. Ingest daily arrival volume forecasts.
3. Cross-market price signals from Indore and Lasalgaon.

---

## Project Structure
```
OnionPrice AI/
├── data/
│   ├── raw/ujjain_onion_raw.csv           # 1,086 raw market price records (2024-2026)
│   └── processed/ujjain_onion_clean.csv   # 699 daily aggregated records
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training.ipynb
├── src/
│   ├── data_collection.py                 # Live AGMARKNET 2.0 REST API ingestion
│   ├── preprocessing.py                   # Daily aggregation & cleaning
│   ├── feature_engineering.py             # 20 zero-leakage time-series features
│   ├── baselines.py                       # Modular baseline estimators
│   ├── train.py                           # 2026 holdout benchmark & retraining
│   ├── evaluate.py                        # MAE, RMSE, R² in ₹/Quintal
│   └── predict.py                         # 1, 7, 14-day live forecasting
├── models/
│   ├── final_model.pkl                    # Retrained model bundle
│   └── model_comparison.csv              # 2026 model benchmark table
├── app/
│   └── streamlit_app.py                   # Streamlit web application
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   └── test_prediction.py
├── app.py
├── requirements.txt
├── render.yaml
└── README.md
```
