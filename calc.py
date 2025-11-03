"""
Calculation module for HAR-RV model and realized variance computations.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
from typing import Dict, Optional


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """Calculate log returns: log(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1))


def calculate_realized_variance(returns: pd.Series) -> float:
    """Calculate realized variance as sum of squared returns."""
    return (returns ** 2).sum()


def calculate_rv_components(data: pd.DataFrame, current_idx: int) -> Optional[Dict[str, float]]:
    """Calculate cascading RV components (5min, hourly, daily) up to current index."""
    if current_idx < 78:
        return None
    
    prices_before_current = data['Close'].iloc[:current_idx]
    returns = calculate_log_returns(prices_before_current).dropna()
    
    if len(returns) < 78:
        return None
    
    rv_5min = calculate_realized_variance(returns.iloc[-1:])
    
    rv_hourly = calculate_realized_variance(returns.iloc[-12:]) / 12
    
    rv_daily = calculate_realized_variance(returns.iloc[-78:]) / 78
    
    return {
        'rv_5min': rv_5min,
        'rv_hourly': rv_hourly,
        'rv_daily': rv_daily
    }


def prepare_training_data(data: pd.DataFrame, target_date: str) -> pd.DataFrame:
    """Prepare training dataset using all data before target date."""
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    
    training_data = data[data.index.date < target_dt.date()].copy()
    
    if len(training_data) < 78:
        raise ValueError(
            f"Insufficient training data: {len(training_data)} bars available, "
            f"need at least 78 bars (1 day)"
        )
    
    training_records = []
    print(training_data)
    
    for i in range(78, len(training_data) - 1):
        components = calculate_rv_components(training_data, i)
        
        if components is None:
            continue
        
        next_return = calculate_log_returns(training_data['Close'].iloc[i:i+2]).iloc[-1]
        rv_next = next_return ** 2
        
        training_records.append({
            'rv_5min': components['rv_5min'],
            'rv_hourly': components['rv_hourly'],
            'rv_daily': components['rv_daily'],
            'rv_next': rv_next
        })
    
    training_df = pd.DataFrame(training_records)
    
    if len(training_df) == 0:
        raise ValueError("No valid training samples could be created")
    
    print(f"Prepared {len(training_df)} training samples from historical data")
    
    return training_df


def fit_har_model(training_data: pd.DataFrame) -> LinearRegression:
    """Fit HAR-RV model using OLS regression."""
    X = training_data[['rv_5min', 'rv_hourly', 'rv_daily']].values
    y = training_data['rv_next'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    print(f"HAR Model Coefficients:")
    print(f"  Intercept: {model.intercept_:.2e}")
    print(f"  RV_5min:   {model.coef_[0]:.4f}")
    print(f"  RV_hourly: {model.coef_[1]:.4f}")
    print(f"  RV_daily:  {model.coef_[2]:.4f}")
    
    return model


def simulate_streaming(data: pd.DataFrame, target_date: str) -> pd.DataFrame:
    """Simulate streaming predictions for target date."""
    print("Preparing training data...")
    training_data = prepare_training_data(data, target_date)
    
    print("Fitting HAR model...")
    model = fit_har_model(training_data)
    
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    target_day_data = data[data.index.date == target_dt.date()].copy()
    
    if len(target_day_data) == 0:
        raise ValueError(f"No data available for target date {target_date}")
    
    print(f"Simulating streaming predictions for {len(target_day_data) - 1} timestamps...")
    
    predictions = []
    
    pre_target_data = data[data.index.date < target_dt.date()]
    combined_data = pd.concat([pre_target_data, target_day_data])
    combined_data = combined_data.sort_index()
    
    start_idx = len(pre_target_data)
    
    for i in range(start_idx, start_idx + len(target_day_data) - 1):
        components = calculate_rv_components(combined_data, i)
        
        if components is None:
            continue
        
        X_pred = np.array([[
            components['rv_5min'],
            components['rv_hourly'],
            components['rv_daily']
        ]])
        predicted_rv = model.predict(X_pred)[0]
        
        next_return = calculate_log_returns(
            combined_data['Close'].iloc[i:i+2]
        ).iloc[-1]
        actual_rv = next_return ** 2
        
        timestamp = combined_data.index[i]
        
        predictions.append({
            'timestamp': timestamp,
            'predicted_rv': predicted_rv,
            'actual_rv': actual_rv
        })
    
    predictions_df = pd.DataFrame(predictions)
    
    if len(predictions_df) == 0:
        raise ValueError("No predictions could be generated for target date")
    
    print(f"Generated {len(predictions_df)} predictions")
    
    return predictions_df

