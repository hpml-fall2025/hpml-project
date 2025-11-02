"""
Configuration file for FinBERT-CNN volatility forecasting
"""
from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    # Data
    data_path: str = "data/headlines_rv.csv"
    date_col: str = "date"
    headline_col: str = "headline"
    rv_col: str = "realized_volatility"
    
    # Train/val/test split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # FinBERT
    finbert_model: str = "ProsusAI/finbert"
    max_length: int = 512
    embedding_dim: int = 768  # FinBERT output dimension
    
    # CNN Architecture (following the paper)
    filter_sizes: List[int] = None  # [1, 2, 3] for unigram, bigram, trigram
    num_filters: int = 50  # Number of filters per size
    dropout_rate: float = 0.5
    
    # FCNN
    hidden_dim: int = 128
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 50
    weight_decay: float = 3.0  # L2 regularization (paper uses 3)
    
    # Device
    device: str = "cuda"  # or "cpu"
    
    # Reproducibility
    random_seed: int = 42
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5  # Save every N epochs
    
    def __post_init__(self):
        if self.filter_sizes is None:
            self.filter_sizes = [1, 2, 3]
