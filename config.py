"""
Configuration file for 5-minute Realized Variance Prediction System.
"""

TICKER = "SPY"

TEST_INTERVAL = ["2025-11-10", "2025-11-14"]

VALIDATION_INTERVAL = ["2025-11-03", "2025-11-07"]

TRAINING_INTERVAL = ["2025-09-29", "2025-10-31"]

TIME_INTERVAL = "5m"

PREDICTION_HORIZON = "5min"

BARS_PER_DAY = 78

MODEL_CONFIGS = {
        'features': ['1b'],
        'reg_type': 'ridge',
        'ridge_alpha': 1000,
}

ALPHA_CONFIGS = [0, 10e-5, 10e-4, 10e-3, 0.01, 0.1, 1.0, 10, 100]
FEATURE_CONFIGS = ['1b', '12b', '78b', '390b', '1D', '1W']