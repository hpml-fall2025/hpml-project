"""
Data loading and preprocessing for headline-based RV forecasting
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Tuple
from sklearn.model_selection import train_test_split

class HeadlineRVDataset(Dataset):
    """Dataset for headline text and realized volatility pairs"""
    
    def __init__(self, headlines, rv_values, tokenizer, max_length=512):
        """
        Args:
            headlines: List of headline strings
            rv_values: Array of realized volatility values
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.headlines = headlines
        self.rv_values = torch.FloatTensor(rv_values)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.headlines)
    
    def __getitem__(self, idx):
        headline = str(self.headlines[idx])
        rv = self.rv_values[idx]
        
        # Tokenize (returns input_ids, attention_mask)
        encoding = self.tokenizer(
            headline,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'rv': rv
        }


def load_data(config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and split data into train/val/test sets
    
    Args:
        config: Configuration object
        
    Returns:
        train_df, val_df, test_df
    """
    # Load CSV
    df = pd.read_csv(config.data_path)
    
    # Ensure date column is datetime
    df[config.date_col] = pd.to_datetime(df[config.date_col])
    
    # Sort by date (temporal ordering is important!)
    df = df.sort_values(config.date_col).reset_index(drop=True)
    
    # Calculate split indices
    n = len(df)
    train_end = int(n * config.train_ratio)
    val_end = train_end + int(n * config.val_ratio)
    
    # Split preserving temporal order
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    print(f"Data splits:")
    print(f"  Train: {len(train_df)} samples ({train_df[config.date_col].min()} to {train_df[config.date_col].max()})")
    print(f"  Val:   {len(val_df)} samples ({val_df[config.date_col].min()} to {val_df[config.date_col].max()})")
    print(f"  Test:  {len(test_df)} samples ({test_df[config.date_col].min()} to {test_df[config.date_col].max()})")
    
    return train_df, val_df, test_df


def create_dataloaders(config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train/val/test
    
    Args:
        config: Configuration object
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load data splits
    train_df, val_df, test_df = load_data(config)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.finbert_model)
    
    # Create datasets
    train_dataset = HeadlineRVDataset(
        train_df[config.headline_col].values,
        train_df[config.rv_col].values,
        tokenizer,
        config.max_length
    )
    
    val_dataset = HeadlineRVDataset(
        val_df[config.headline_col].values,
        val_df[config.rv_col].values,
        tokenizer,
        config.max_length
    )
    
    test_dataset = HeadlineRVDataset(
        test_df[config.headline_col].values,
        test_df[config.rv_col].values,
        tokenizer,
        config.max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,  # Shuffle within epochs
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
