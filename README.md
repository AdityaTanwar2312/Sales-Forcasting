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
