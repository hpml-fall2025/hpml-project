import os
import sys

# Setup paths to fix module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'finBERT'))
sys.path.append(os.path.dirname(current_dir))

import numpy as np
import pandas as pd
from finBERT.finbert.finbert import predict
from transformers import AutoModelForSequenceClassification
from pipelines.base import Pipeline
import datetime 
import csv

class NewsPipeline(Pipeline):
    def __init__(self, use_gpu = True):
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
        
        # Load Data
        data_path = "data/headlines.csv"
        # Reading tab-separated file
        self.df = pd.read_csv(data_path)
        
        self.use_gpu = use_gpu

        self.df = self.df.drop('URL', axis=1)
        self.df['Timestamp'] = pd.to_datetime(
            self.df['Timestamp'],
            format='%Y-%m-%dT%H:%M:%SZ',
            errors='raise'
        )
        self.df['Timestamp'] = self.df['Timestamp'].apply(lambda x : x.date())
            
    def get_latest_data(self, dates) -> dict:
        #dates -> date time objects
                # Combine text for batch prediction
        # We join with a period to ensure sent_tokenize splits them correctly
        mask = self.df["Timestamp"].isin(dates)
        rows = self.df.loc[mask]
        headlines = rows["Headline"].tolist()
        full_text = ". ".join(headlines)

        self.results_df = predict(full_text, self.model, use_gpu=self.use_gpu)
        self.sentiment_scores = self.results_df['sentiment_score'].values
        
        vol =  sum(self.sentiment_scores**2)  / len(self.sentiment_scores) if len(self.sentiment_scores) > 0 else 0.0

        return vol

#Example usage:
# news_pipe = NewsPipeline()
# dates = [datetime.datetime.strptime(date_string, "%Y-%m-%d").date() for date_string in ['2021-01-04', '2021-01-06', '2021-02-07']]
# news_pipe.get_latest_data(dates)