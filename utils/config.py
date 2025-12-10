import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data storage
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_FILE = os.path.join(DATA_DIR, "rv_data.parquet")

# Simulation settings
SEED = 42
REFRESH_RATE = 1.0  # seconds
