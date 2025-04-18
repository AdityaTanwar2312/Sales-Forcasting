### Revenue Pridiction using XGBoost, LSTM and ARIMA

## 📊 ARIMA Model: Coffee Sales Forecasting

**Model Used:** `ARIMA(1,1,1)`  
**Library:** `statsmodels`

### 🔄 Preprocessing Steps:
- **Data Resampling:** Aggregated sales data to daily frequency (`df_daily`).
- **Stationarity Check:** Conducted the Augmented Dickey-Fuller (ADF) test to confirm non-stationarity.
- **Differencing:** Applied first-order differencing (`d=1`) to make the time series stationary and suitable for ARIMA modeling.
- **Validation Approach:** Used **walk-forward validation** with an expanding training window on the last **30 days** of data.

### 📈 Evaluation Metrics:
| Metric | Value |
|--------|--------|
| MAE    | 300.74 |
| RMSE   | 449.92 |
| MAPE   | 5.54%  |

### 🧠 Inference:
- ARIMA(1,1,1) provided a **strong baseline forecast** with low percentage error (MAPE).
- It effectively modeled the **overall trend** in the data but may underperform on sudden spikes or irregular patterns due to its linear nature.
- Walk-forward validation ensured realistic performance estimation on unseen data.


## 🧠 LSTM-Based Time Series Forecasting

This project implements a Long Short-Term Memory (LSTM) neural network to forecast **daily revenue** from transactional sales data.

### 📊 Dataset

- Sales data including:
  - `transaction_date`, `transaction_time`
  - `transaction_qty`, `unit_price`
- Derived revenue column: `revenue = transaction_qty * unit_price`

### Preprocessing:
- Timestamp creation and grouping by `store_location` and `product_category`
- Resampled to daily frequency
- Feature Engineering:
  - Lag features: `revenue_lag1`, `revenue_lag7`, `revenue_lag30`
  - Rolling statistics: `rolling_mean_7`, `rolling_std_7`, `rolling_mean_30`
  - `cumulative_revenue`, `dayofweek`, and `is_weekend`
- One-hot encoding for categorical features

### 🧮 Model Architecture

- Framework: Keras (TensorFlow backend)
- Lookback window: 7 days
- Layers:
  - LSTM → Dropout → LSTM → Dense
- Optimizer: Adam
- Loss Function: Mean Absolute Percentage Error (MAPE)

### 📈 Evaluation Metrics

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error

### 📉 Results

- The model effectively captures seasonal patterns and daily fluctuations.
- Good generalization on unseen validation data.
- Residuals are symmetrically distributed, showing no major bias.

### 📌 Graphs

- **Actual vs Predicted** revenue plots
- **Loss curves** over training epochs
- **Error distribution** visualization


## 📌 Inference: XGBoost Model Performance Summary

The **XGBoost** model demonstrated strong performance in predicting **daily revenue**, particularly after training on a well-engineered dataset with reduced dimensionality.


### ✅ Model Accuracy & Fit

- **R² Score**: `0.7706`  
  The model explains **77% of the variance** in daily revenue, which reflects a solid fit for a financial time series dataset that typically contains a high degree of noise and irregularity.


### 📊 Evaluation Metrics

| Metric | Value |
|--------|-------|
| **MAE** (Mean Absolute Error) | `77.91` |
| **RMSE** (Root Mean Squared Error) | `114.04` |

- **MAE** indicates that the model's predictions deviate by around ₹78 from actual values on average — a low absolute error relative to expected daily revenue figures.
- **RMSE**, being slightly higher due to squaring larger deviations, still confirms that prediction errors remain low and consistent.


### 🔍 Feature Effectiveness

- The model used **7 features**, selected from an original **14-feature set**, demonstrating that **feature reduction preserved predictive power**.
- Key feature types:
  - **Lag values** (e.g., revenue_lag1, lag7, lag30)
  - **Rolling statistics** (e.g., rolling_mean_7, rolling_std_30)
  - **Temporal features** (e.g., day of week, is_weekend)

- **No feature scaling required** — XGBoost handles raw magnitudes effectively, simplifying preprocessing for deployment.


### ⚖️ Comparison to Baseline

- The XGBoost model **clearly outperforms naive baselines** like:
  - **Last-value persistence**
  - **Simple moving averages**
- With high R² and low error metrics, it provides **reliable, scalable predictions**, suitable for production use in retail or revenue forecasting pipelines.
