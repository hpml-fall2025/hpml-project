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
    
    def predict_news_vol(self, date) -> (float, float):
        """
        predicts volatility wrt news headlines with custom drop-off weighting
        """
        day_weights = [0.5, 0.25, 0.13, 0.07, 0.03]  
        
        vol = 0
        num_headlines = 0
        
        for i in range(len(day_weights)):
            check_date = date - datetime.timedelta(days=i)
            mask = self.df["Timestamp"] == check_date
            day_rows = self.df.loc[mask]
            
            if len(day_rows)==0:
                continue
            
            day_headlines = day_rows["Headline"].tolist()
            model_batch = " .".join(day_headlines) #we do this because the model splits by "." to figure out the seperate headlines
            
            results_df = predict(model_batch, self.model, use_gpu=self.use_gpu)
            day_sentiment_scores = results_df['sentiment_score'].values
            day_avg_ss = sum(day_sentiment_scores ** 2) / len(day_sentiment_scores) #average squared sentiment score for a fixed day
            num_headlines += len(day_sentiment_scores)
            vol += day_weights[i] * day_avg_ss
        
        return vol, num_headlines

            
    def get_latest_data(self, query_date) -> dict:
        day_weights = [0.5, 0.25, 0.13, 0.07, 0.03]  
        
        vol = 0
        
        for i in range(len(day_weights)):
            check_date = query_date - datetime.timedelta(days=i)
            mask = self.df["Timestamp"] == check_date
            day_rows = self.df.loc[mask]
            
            if len(day_rows)==0:
                continue
            
            day_headlines = day_rows["Headline"].tolist()
            model_batch = " .".join(day_headlines) #we do this because the model splits by "." to figure out the seperate headlines
            
            results_df = predict(model_batch, self.model, use_gpu=self.use_gpu)
            day_sentiment_scores = results_df['sentiment_score'].values
            day_avg_ss = sum(day_sentiment_scores ** 2) / len(day_sentiment_scores) #average squared sentiment score for a fixed day
            vol += day_weights[i] * day_avg_ss
            
        return vol

#Example usage:
# news_pipe = NewsPipeline()
# dates = [datetime.datetime.strptime(date_string, "%Y-%m-%d").date() for date_string in ['2021-01-04', '2021-01-06', '2021-02-07']]
# news_pipe.get_latest_data(dates)
