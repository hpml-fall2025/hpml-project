import os
import sys

# Setup paths to fix module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'finBERT'))
sys.path.append(os.path.dirname(current_dir))

import numpy as np
import pandas as pd
from finbert.finbert import predict
from transformers import AutoModelForSequenceClassification
from pipelines.base import Pipeline

class NewsPipeline(Pipeline):
    def __init__(self):
        # FinBERT paths
        current_file = os.path.abspath(__file__)
        pipeline_dir = os.path.dirname(current_file)
        finbert_root = os.path.join(pipeline_dir, 'finBERT')

        # Load Model
        # Using the relative path to the model based on the user instructions and file structure
        model_path = os.path.join(finbert_root, 'models', 'sentiment')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3, cache_dir=None)
        
        use_gpu = True
        
        # Load Data
        data_path = os.path.join(finbert_root, 'data', 'sentiment_data', 'test.csv')
        # Reading tab-separated file
        df = pd.read_csv(data_path, sep='\t')
            
        if 'text' not in df.columns:
            # Handle case where first column is index but unnamed in header (common in some csv exports)
            # The user provided file has "\ttext\tlabel" which pandas usually parses with an empty first col name
            pass
            
        # Combine text for batch prediction
        # We join with a period to ensure sent_tokenize splits them correctly
        texts = df['text'].astype(str).tolist()
        full_text = ". ".join(texts)
        
        # Run prediction
        print("Running FinBERT prediction on test data...")
        self.results_df = predict(full_text, self.model, use_gpu=True)
        self.sentiment_scores = self.results_df['sentiment_score'].values
        self.current_idx = 0
        
        print(f"NewsPipeline initialized. Loaded {len(self.sentiment_scores)} sentiment scores.")

    def get_latest_data(self) -> dict:
        noise = self.rng.normal(0.0, self.sigma)
        new_log_val = self.mu + self.phi * (self.last_log_val - self.mu) + noise
        self.last_log_val = new_log_val
        return {"news_rv": float(np.exp(new_log_val))}

    def get_headline(self):
        headlines = [
            "SPY rallies on strong tech earnings",
            "Fed signals potential rate hike, markets jittery",
            "Inflation data comes in lower than expected",
            "Energy sector drags SPY lower",
            "Global supply chain issues persist, affecting outlook",
            "Consumer confidence hits 5-year high",
            "Tech sell-off continues as yields rise",
            "SPY steady ahead of jobs report",
            "Geopolitical tensions rise, impacting volatility",
            "Analysts upgrade S&P 500 price target"
        ]
        return self.rng.choice(headlines)
        if len(self.sentiment_scores) == 0:
            return {"news_rv": 0.0}
            
        score = self.sentiment_scores[self.current_idx]
        
        news_rv = score ** 2
        
        self.current_idx = (self.current_idx + 1) % len(self.sentiment_scores)
        
        return {"news_rv": float(news_rv)}
