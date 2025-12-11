"""
Configuration file for 5-minute Realized Variance Prediction System.
"""

TICKER = "SPY"

TEST_INTERVAL = ["2024-01-01", "2025-11-01"]

VALIDATION_INTERVAL = ["2025-11-02", "2025-12-02"]

TRAINING_INTERVAL = ["2025-05-01", "2025-10-31"]

TIME_INTERVAL = "1d"

PREDICTION_HORIZON = "1d"

BARS_PER_DAY = 1

MODEL_CONFIGS = {
        'features': ['1b'],
        'reg_type': 'ridge',
        'ridge_alpha': 1000,
}

ALPHA_CONFIGS = [0, 10e-5, 10e-4, 10e-3, 0.01, 0.1, 1.0, 10, 100]
FEATURE_CONFIGS = ['1D', '1W', '1M']