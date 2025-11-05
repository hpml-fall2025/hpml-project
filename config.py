"""
Configuration file for 5-minute Realized Variance Prediction System.
"""

TICKER = "SPY"

TEST_INTERVAL = ["2025-10-27", "2025-10-31"]

VALIDATION_INTERVAL = ["2025-10-20", "2025-10-24"]

TRAINING_INTERVAL = ["2025-09-15", "2025-10-17"]

TIME_INTERVAL = "5m"

PREDICTION_HORIZON = "5min"

BARS_PER_DAY = 78

MODEL_CONFIGS = {
        'features': ['1b'],
        'reg_type': 'ridge',
        'ridge_alpha': 1000,
}

ALPHA_CONFIGS = [0, 0.01, 0.1, 1.0, 10, 100, 1000, 10e3, 10e4, 10e5]
FEATURE_CONFIGS = ['1b', '12b', '78b', '390b', '1D', '1W']