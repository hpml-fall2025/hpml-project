"""
PyTorch Dataset classes for FinBERT training and inference.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
import config


class FinancialSentimentDataset(Dataset):
    """
    PyTorch Dataset for Financial PhraseBank sentiment data.
    Tokenizes sentences and returns input_ids, attention_mask, and sentiment labels.
    """
    
    def __init__(self, csv_path, tokenizer, max_length=None):
        """
        Args:
            csv_path: Path to CSV file with 'sentence' and 'sentiment_score' columns
            tokenizer: HuggingFace BertTokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length or config.MAX_SEQ_LENGTH
        
        # Verify required columns
        if 'sentence' not in self.data.columns:
            raise ValueError("CSV must contain 'sentence' column")
        if 'sentiment_score' not in self.data.columns:
            raise ValueError("CSV must contain 'sentiment_score' column")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            input_ids: Token IDs (torch.LongTensor)
            attention_mask: Attention mask (torch.LongTensor)
            label: Sentiment score as float (torch.FloatTensor)
        """
        row = self.data.iloc[idx]
        sentence = str(row['sentence'])
        sentiment = float(row['sentiment_score'])
        
        # Tokenize
        encoding = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(sentiment, dtype=torch.float32)
        }


class HeadlineDataset(Dataset):
    """
    PyTorch Dataset for financial headlines (inference only).
    Used during sentiment scoring phase.
    """
    
    def __init__(self, headlines, tokenizer, max_length=None):
        """
        Args:
            headlines: List of headline strings
            tokenizer: HuggingFace BertTokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.headlines = headlines
        self.tokenizer = tokenizer
        self.max_length = max_length or config.MAX_SEQ_LENGTH
    
    def __len__(self):
        return len(self.headlines)
    
    def __getitem__(self, idx):
        """
        Returns:
            input_ids: Token IDs (torch.LongTensor)
            attention_mask: Attention mask (torch.LongTensor)
        """
        headline = str(self.headlines[idx])
        
        # Tokenize
        encoding = self.tokenizer(
            headline,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


def create_dataloaders(train_dataset, val_dataset, batch_size=None, shuffle_train=True):
    """
    Create DataLoaders for training and validation.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size (uses config.BATCH_SIZE if None)
        shuffle_train: Whether to shuffle training data
    
    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader
    
    batch_size = batch_size or config.BATCH_SIZE
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=0,  # MPS doesn't work well with multiprocessing
        pin_memory=False  # Not needed for MPS
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, val_loader


def load_finbert_data():
    """
    Load and prepare FinBERT training data from Financial PhraseBank.
    
    Returns:
        train_dataset, val_dataset, tokenizer
    """
    print("Loading FinBERT training data...")
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Load Financial PhraseBank
    phrasebank_path = config.RAW_DATA_DIR / "financial_phrasebank.csv"
    
    if not phrasebank_path.exists():
        raise FileNotFoundError(
            f"Financial PhraseBank not found at {phrasebank_path}. "
            "Run data_collection.py first."
        )
    
    # Load data
    df = pd.read_csv(phrasebank_path)
    
    # Split into train/val (80/20 split)
    # Use fixed random state for reproducibility
    from sklearn.model_selection import train_test_split
    
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=config.RANDOM_SEED,
        stratify=df['label']  # Stratify by original label for balanced splits
    )
    
    # Save temporary split files
    train_path = config.PROCESSED_DATA_DIR / "finbert_train.csv"
    val_path = config.PROCESSED_DATA_DIR / "finbert_val.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    # Create datasets
    train_dataset = FinancialSentimentDataset(train_path, tokenizer)
    val_dataset = FinancialSentimentDataset(val_path, tokenizer)
    
    print(f"✓ Data loaded:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    
    return train_dataset, val_dataset, tokenizer


if __name__ == "__main__":
    """Test the data loader"""
    try:
        train_dataset, val_dataset, tokenizer = load_finbert_data()
        
        print(f"\nTesting dataset...")
        sample = train_dataset[0]
        print(f"  Input IDs shape: {sample['input_ids'].shape}")
        print(f"  Attention mask shape: {sample['attention_mask'].shape}")
        print(f"  Label: {sample['labels'].item():.2f}")
        
        # Decode sample
        decoded = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        print(f"  Decoded text: {decoded[:100]}...")
        
        print("\n✓ Data loader test passed!")
        
    except Exception as e:
        print(f"\n✗ Data loader test failed: {e}")
        raise

