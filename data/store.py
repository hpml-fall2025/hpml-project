import pandas as pd
from datetime import timedelta
import numpy as np

class DataStore:
    def __init__(self, initial_data: pd.DataFrame = None):
        """
        Initialize the DataStore.
        If initial_data is provided, it should be a DataFrame with a 'timestamp' column.
        """
        if initial_data is not None:
            self.df = initial_data
        else:
            self.df = pd.DataFrame(columns=["timestamp", "news_rv", "har_rv", "actual_rv", "combined_rv"])

    def append_data(self, new_data: dict):
        """
        Appends a new row of data.
        new_data should be a dict like {'timestamp': ..., 'news_rv': ..., ...}
        """
        # Ensure timestamp is present
        if "timestamp" not in new_data:
            new_data["timestamp"] = pd.Timestamp.now(tz="UTC")
        
        # Create a DataFrame for the new row
        new_row = pd.DataFrame([new_data])
        
        # Concatenate
        if self.df.empty:
            self.df = new_row
        else:
            self.df = pd.concat([self.df, new_row], ignore_index=True)

    def get_data(self, timeframe: str = "30 minutes") -> pd.DataFrame:
        """
        Returns data filtered by the specified timeframe.
        Options: "5 minutes", "30 minutes", "2 hours", "1 day", "All"
        """
        if self.df.empty:
            return self.df

        end = self.df["timestamp"].iloc[-1]
        
        mapping = {
            "5 minutes": timedelta(minutes=5),
            "30 minutes": timedelta(minutes=30),
            "2 hours": timedelta(hours=2),
            "1 day": timedelta(days=1),
        }

        if timeframe == "All":
            return self.df
            
        window = mapping.get(timeframe, timedelta(minutes=30))
        start = end - window
        
        return self.df[self.df["timestamp"] >= start]
