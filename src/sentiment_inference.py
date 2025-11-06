"""
Sentiment inference pipeline for financial headlines.
Performs batch inference and aggregates daily sentiment features.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import ast
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.finbert_model import FinBERT
from src.data_loader import HeadlineDataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer


class SentimentInference:
    """
    Inference engine for sentiment scoring of financial headlines.
    """
    
    def __init__(self, model_path, device=None):
        """
        Args:
            model_path: Path to trained FinBERT model directory
            device: Device to run inference on (default: config.DEVICE)
        """
        self.device = device or config.DEVICE
        self.model_path = Path(model_path)
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = FinBERT.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        tokenizer_path = config.FINBERT_MODEL_DIR / 'tokenizer'
        if not tokenizer_path.exists():
            print(f"Warning: Tokenizer not found at {tokenizer_path}, using default")
            tokenizer_path = config.MODEL_NAME
        
        self.tokenizer = BertTokenizer.from_pretrained(str(tokenizer_path))
        
        print(f"✓ Model loaded on {self.device}")
    
    def predict_sentiment(self, headlines, batch_size=None):
        """
        Predict sentiment scores for a list of headlines.
        
        Args:
            headlines: List of headline strings
            batch_size: Batch size for inference (default: config.INFERENCE_BATCH_SIZE)
        
        Returns:
            numpy array of sentiment scores
        """
        if not headlines:
            return np.array([])
        
        batch_size = batch_size or config.INFERENCE_BATCH_SIZE
        
        # Create dataset and dataloader
        dataset = HeadlineDataset(headlines, self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Inference
        all_predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                predictions = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                all_predictions.extend(predictions.cpu().numpy())
        
        return np.array(all_predictions)
    
    def aggregate_daily_sentiment(self, sentiment_scores):
        """
        Aggregate sentiment scores for a single day.
        Calculates mean squared sentiment and variance as per plan specifications.
        
        Args:
            sentiment_scores: Array of sentiment scores for one day
        
        Returns:
            Dict with aggregated features
        """
        if len(sentiment_scores) == 0:
            return {
                'mean_square_sentiment': 0.0,
                'sentiment_variance': 0.0,
                'headline_count': 0
            }
        
        return {
            'mean_square_sentiment': float(np.mean(sentiment_scores ** 2)),
            'sentiment_variance': float(np.var(sentiment_scores)),
            'headline_count': len(sentiment_scores)
        }
    
    def process_aligned_dataset(self, aligned_csv_path, output_path=None):
        """
        Process entire aligned dataset and generate sentiment features.
        
        Args:
            aligned_csv_path: Path to aligned dataset CSV
            output_path: Path to save sentiment features (optional)
        
        Returns:
            DataFrame with sentiment features
        """
        print(f"\nProcessing aligned dataset from {aligned_csv_path}...")
        
        # Load aligned data
        df = pd.read_csv(aligned_csv_path, index_col=0, parse_dates=True)
        
        # Parse headline lists (they may be stored as strings)
        if isinstance(df['headline'].iloc[0], str):
            df['headline'] = df['headline'].apply(ast.literal_eval)
        
        sentiment_features = []
        
        # Process each day
        for date, row in tqdm(df.iterrows(), total=len(df), desc="Scoring headlines"):
            headlines = row['headline']
            
            # Predict sentiment for all headlines
            if headlines and len(headlines) > 0:
                sentiment_scores = self.predict_sentiment(headlines)
            else:
                sentiment_scores = np.array([])
            
            # Aggregate
            features = self.aggregate_daily_sentiment(sentiment_scores)
            features['date'] = date
            
            sentiment_features.append(features)
        
        # Create DataFrame
        features_df = pd.DataFrame(sentiment_features)
        features_df.set_index('date', inplace=True)
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            features_df.to_csv(output_path)
            print(f"✓ Sentiment features saved to {output_path}")
        
        # Print statistics
        print(f"\nSentiment Feature Statistics:")
        print(f"  Total days: {len(features_df)}")
        print(f"  Mean square sentiment: {features_df['mean_square_sentiment'].mean():.4f} ± {features_df['mean_square_sentiment'].std():.4f}")
        print(f"  Sentiment variance: {features_df['sentiment_variance'].mean():.4f} ± {features_df['sentiment_variance'].std():.4f}")
        print(f"  Avg headlines/day: {features_df['headline_count'].mean():.1f}")
        
        return features_df
    
    def process_splits(self):
        """
        Process train/val/test splits and generate sentiment features for each.
        """
        print("\n" + "=" * 60)
        print("Processing Train/Val/Test Splits")
        print("=" * 60)
        
        splits = ['train', 'val', 'test']
        
        for split in splits:
            print(f"\nProcessing {split} split...")
            
            input_path = config.SPLITS_DIR / f"{split}.csv"
            output_path = config.PROCESSED_DATA_DIR / f"sentiment_features_{split}.csv"
            
            if not input_path.exists():
                print(f"  ⚠ {split}.csv not found, skipping...")
                continue
            
            self.process_aligned_dataset(input_path, output_path)
        
        print("\n" + "=" * 60)
        print("✓ All splits processed!")
        print("=" * 60)


def main():
    """
    Main inference function.
    """
    print("=" * 60)
    print("FinBERT Sentiment Inference Pipeline")
    print("=" * 60)
    
    # Check if model exists
    model_path = config.FINBERT_MODEL_DIR / 'best_model'
    
    if not model_path.exists():
        print(f"\n✗ Model not found at {model_path}")
        print("Please train the model first using train_finbert.py")
        return
    
    # Create inference engine
    inference = SentimentInference(model_path)
    
    # Process aligned dataset
    aligned_path = config.PROCESSED_DATA_DIR / "aligned_dataset.csv"
    
    if aligned_path.exists():
        print("\nProcessing full aligned dataset...")
        output_path = config.PROCESSED_DATA_DIR / "sentiment_features.csv"
        inference.process_aligned_dataset(aligned_path, output_path)
    
    # Process splits
    if config.SPLITS_DIR.exists():
        inference.process_splits()
    
    print("\n✓ Sentiment inference complete!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Inference failed: {e}")
        raise

