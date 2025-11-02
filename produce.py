"""
Output module for displaying 5-minute realized variance predictions and statistics.
"""

import pandas as pd
import numpy as np


def output_results(predictions: pd.DataFrame) -> None:
    """
    Display prediction results with formatted table and summary statistics.
    
    Args:
        predictions: DataFrame with columns: timestamp, predicted_rv, actual_rv
    """
    if len(predictions) == 0:
        print("No predictions to display")
        return
    
    predictions = predictions.copy()
    
    predictions['abs_error'] = abs(predictions['predicted_rv'] - predictions['actual_rv'])
    predictions['pct_error'] = (predictions['abs_error'] / predictions['actual_rv']) * 100
    
    _print_predictions_table(predictions)
    _print_summary_statistics(predictions)


def _print_predictions_table(predictions: pd.DataFrame) -> None:
    """
    Print formatted table of predictions.
    
    Args:
        predictions: DataFrame with predictions and calculated errors
    """
    print("\n" + "=" * 80)
    print("5-Minute Realized Variance Predictions")
    print("=" * 80)
    print(f"{'Timestamp':<20} | {'Predicted RV':<12} | {'Actual RV':<12} | {'Error (%)':<10}")
    print("-" * 80)
    
    for _, row in predictions.iterrows():
        timestamp_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        predicted = row['predicted_rv']
        actual = row['actual_rv']
        error_pct = row['pct_error']
        
        print(f"{timestamp_str:<20} | {predicted:>12.9f} | {actual:>12.9f} | {error_pct:>9.2f}%")
    
    print("=" * 80)


def _print_summary_statistics(predictions: pd.DataFrame) -> None:
    """
    Print summary statistics for predictions.
    
    Args:
        predictions: DataFrame with predictions and calculated errors
    """
    total_predictions = len(predictions)
    
    mean_abs_pct_error = predictions['pct_error'].mean()
    median_abs_pct_error = predictions['pct_error'].median()
    std_pct_error = predictions['pct_error'].std()
    
    rmse = np.sqrt((predictions['abs_error'] ** 2).mean())
    
    mae = predictions['abs_error'].mean()
    
    min_error = predictions['pct_error'].min()
    max_error = predictions['pct_error'].max()
    
    print("\nSummary Statistics")
    print("=" * 80)
    print(f"Total Predictions:            {total_predictions:>6}")
    print(f"Mean Absolute % Error (MAPE): {mean_abs_pct_error:>6.2f}%")
    print(f"Median Absolute % Error:      {median_abs_pct_error:>6.2f}%")
    print(f"Std Dev of % Errors:          {std_pct_error:>6.2f}%")
    print(f"Min % Error:                  {min_error:>6.2f}%")
    print(f"Max % Error:                  {max_error:>6.2f}%")
    print(f"Root Mean Squared Error:      {rmse:>6.9f}")
    print(f"Mean Absolute Error:          {mae:>6.9f}")
    print("=" * 80)
    print()

