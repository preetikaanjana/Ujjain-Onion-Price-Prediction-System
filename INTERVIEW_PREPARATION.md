# Machine Learning Interview Preparation Guide (2026 Edition)
## Ujjain Onion Price Prediction System

Truthful, technically accurate answers to all core interview questions reflecting the authentic 2024–2026 implementation.

---

### 1. What problem does your project solve?
**Answer:**  
Wholesale onion prices in Indian mandis fluctuate significantly due to weather events, storage degradation, and arrival patterns. Farmers and traders lack forward-looking price visibility. This project builds an end-to-end machine learning time-series application predicting expected future modal prices (in ₹/Quintal) for Ujjain APMC, Madhya Pradesh, with empirical error-based uncertainty ranges.

### 2. Why did you choose onion?
**Answer:**  
Onion (*Allium cepa*) is one of India's most price-volatile staple crops. Unlike wheat or paddy, onions have no regular government Minimum Support Price (MSP) buffer procurement. They trade predominantly via daily wholesale auctions in APMC yards, making price forecasting a practical, challenging time-series problem.

### 3. Why Ujjain?
**Answer:**  
Ujjain is a premier commercial onion trading hub in Madhya Pradesh's Malwa agricultural belt. Focusing strictly on Ujjain APMC captures local micro-market auction dynamics rather than obscuring localized trends in generic national averages.

### 4. Where did you get the data?
**Answer:**  
Directly from the Government of India's official **AGMARKNET 2.0 REST API** (`https://api.agmarknet.gov.in/v1/`) under the Directorate of Marketing & Inspection (DMI), Ministry of Agriculture & Farmers Welfare.

### 5. Is your data real?
**Answer:**  
Yes, 100% authentic government records. Zero synthetic, generated, or fake rows. Zero Faker. The pipeline includes runtime assertions that raise `ValueError` if zero verified Ujjain onion records exist.

### 6. How much 2026 data did you have?
**Answer:**  
We extracted **289 market price records** representing **168 active trading days in 2026**, spanning from **January 1, 2026 to September 19, 2026**.

### 7. What is your target?
**Answer:**  
`modal_price` in **Indian Rupees per Quintal (₹ / 100 kg)**.

### 8. Why modal price?
**Answer:**  
Mandi trading sessions feature transactions across varying produce qualities. Minimum price reflects lower-grade or damaged produce; maximum reflects top-tier quality. The **Modal Price** is the reported price where the highest volume of transactions occurred on that day, representing the central market clearing price.

### 9. Why is this a regression problem?
**Answer:**  
Because the target (`modal_price`) is a continuous numeric price value on a continuous scale, not a discrete class label.

### 10. How are multiple records on the same day handled?
**Answer:**  
The pipeline aggregates multiple available market-price records for the same trading day into a daily price series using the implemented aggregation rule: arithmetic mean for modal price, minimum for min price, maximum for max price, and sum for arrival volume.

### 11. How did you evaluate and predict 2026?
**Answer:**  
We executed a two-objective workflow:
- **Objective A (Model Evaluation):** Trained models on earlier historical data (2024 to mid-2025), validated on late 2025, and evaluated on the **entire unseen 2026 period (Jan 1 to Sep 19, 2026, 168 trading days)** to assess out-of-sample generalization.
- **Objective B (Current Forecasting):** Retrained the final model on all data up to September 19, 2026, and forecasted 1-day, 7-day, and 14-day future prices forward.

### 12. How did you prevent data leakage?
**Answer:**  
1. All lag features ($t-1, t-2, t-3, t-7, t-14, t-30$) and rolling statistics (7, 14, 30 days) were strictly calculated on shifted historical data ($t-1$ downwards). No price from day $t$ or later is ever present in feature inputs.
2. Chronological splitting: The 2026 test set was held out in the future; zero random shuffling (`shuffle=False`).

### 13. Why didn't you randomly split the data?
**Answer:**  
Because this is time-series data. Random splitting could allow information from future periods to influence training. I therefore used chronological train, validation and unseen 2026 test periods.

### 14. What models did you compare?
**Answer:**  
We evaluated two baseline methods and five machine learning models:
- **Baselines:** Previous-day naive prediction ($y_t = y_{t-1}$), 7-day moving average
- **Machine Learning Models:** Linear Regression (OLS), Ridge Regression (L2 regularized), Random Forest Regressor, Gradient Boosting Regressor, and XGBoost Regressor.

### 15. Which model performed best?
**Answer:**  
The 7-day moving average was the strongest baseline on the 2026 holdout set. Among the machine-learning models tested, Random Forest performed best.

### 16. Why did the moving average perform better than some ML models?
**Answer:**  
The recent historical price itself was a strong predictor in this dataset. The 7-day moving average smoothed short-term noise and performed better than the other baseline and ML models on the 2026 holdout period.

### 17. What is the accuracy of your model?
**Answer:**  
This is a time-series regression problem, so I evaluate it using MAE, RMSE, R² and MAPE rather than classification accuracy. On the unseen 2026 test period, the 7-day moving-average baseline achieved an MAE of ₹232.67 per quintal, while Random Forest was the best-performing ML model with an MAE of ₹252.37 per quintal.

### 18. What is MAE?
**Answer:**  
Mean Absolute Error is the average absolute difference between actual and predicted prices: $\text{MAE} = \frac{1}{n} \sum_{t=1}^{n} |y_t - \hat{y}_t|$. In our test set, an MAE of ₹ 232.67 means predictions differed by an average of ₹ 232.67 per quintal.

### 19. What is RMSE?
**Answer:**  
Root Mean Squared Error: $\text{RMSE} = \sqrt{\frac{1}{n} \sum_{t=1}^{n} (y_t - \hat{y}_t)^2}$. It penalizes larger errors more heavily. Our baseline RMSE was ₹ 322.72/Q, and Random Forest RMSE was ₹ 327.49/Q.

### 20. What is R² and what does 0.8576 mean?
**Answer:**  
Coefficient of determination: $R^2 = 1 - \frac{\sum (y_t - \hat{y}_t)^2}{\sum (y_t - \bar{y})^2}$. An R² of 0.8576 means the model explains approximately 85.76% of the variance in the held-out 2026 target values relative to the standard R² baseline.

### 21. Are your uncertainty intervals confidence intervals?
**Answer:**  
No. They are empirical error-based uncertainty ranges derived from historical forecasting errors. They are not statistically calibrated confidence intervals.

### 22. What are the limitations of the system?
**Answer:**  
Mandi prices depend on monsoon rainfall, unseasonal weather, transport logistics, and government export-import policies. The model produces empirical statistical estimates, not financial guarantees. Furthermore, wholesale modal prices represent auction clearing rates and do not measure farmer net profit.

### 23. How would you retrain and update the model as new data arrives?
**Answer:**  
The ingestion script in `src/data_collection.py` fetches newly published trading days from the AGMARKNET API, validates them, appends them to `data/processed/ujjain_onion_clean.csv`, recalculates lag buffers, and periodically refits the model pipeline on updated historical data.
