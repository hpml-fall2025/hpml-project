"""
Configuration file for FinBERT sentiment analysis pipeline.
"""

import torch
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
FINBERT_MODEL_DIR = MODELS_DIR / "finbert"

# Device Configuration
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Data Collection
TICKER = "SPY"
START_DATE = "2022-01-01"
END_DATE = "2024-10-31"

# Train/Val/Test Split Dates
TRAIN_START = "2022-01-01"
TRAIN_END = "2023-06-30"
VAL_START = "2023-07-01"
VAL_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2024-10-31"

# NewsAPI Configuration (optional - set your API key here)
NEWS_API_KEY = None  # Set to your NewsAPI key or leave None to skip

# FinBERT Model Configuration
MODEL_NAME = "bert-base-uncased"
MAX_SEQ_LENGTH = 128
NUM_LABELS = 1  # Continuous sentiment score output
HIDDEN_DROPOUT_PROB = 0.1
ATTENTION_PROBS_DROPOUT_PROB = 0.1

# Training Hyperparameters
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 32
NUM_EPOCHS = 4
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# Learning Rate Scheduler
LR_SCHEDULER_TYPE = "linear"  # Options: linear, cosine, constant

# Layer Freezing
FREEZE_EMBEDDINGS = False
FREEZE_ENCODER_LAYERS = 0  # Number of encoder layers to freeze (0 = no freezing, 12 = freeze all)

# Mixed Precision Training
USE_AMP = True  # Automatic Mixed Precision for MPS/CUDA

# Logging and Checkpointing
LOGGING_STEPS = 50
EVAL_STEPS = 200
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 3  # Keep only best 3 checkpoints

# Sentiment Inference
INFERENCE_BATCH_SIZE = 32

# Random Seed
RANDOM_SEED = 42

# Volatility Calculation
RV_WINDOW_DAILY = 22  # ~1 month
RV_WINDOW_WEEKLY = 5  # ~1 week
RV_WINDOW_MONTHLY = 22  # ~1 month

# Label Mapping for Financial PhraseBank
# Original: negative=0, neutral=1, positive=2
# We'll map these to continuous scores: -1, 0, 1
LABEL_TO_SENTIMENT = {
    0: -1.0,  # negative
    1: 0.0,   # neutral
    2: 1.0    # positive
}

