import pandas as pd
import numpy as np 
import warnings
import statsmodels.api as sm
import datetime as dt
import yfinance as yf
import os
from .base import Pipeline

class VolatilityPipeline(Pipeline):
    def __init__(self):
        # Locate CSV relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, "SPY1min_clean.csv")
        
        # 1. Load and Resample Historical Data
        try:
            # Load only necessary columns to save memory/time if possible, but 130MB is manageable.
            model_data = pd.read_csv(csv_path)
            model_data['date'] = pd.to_datetime(model_data['date'])
            model_data.set_index('date', inplace=True)
            
            # Resample 1min -> 5min
            agg_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum' # Sum volume
            }
            # Drop NaN rows that might result from empty bins (though unlikely for SPY trading hours)
            model_data_5m = model_data.resample('5min').agg(agg_dict).dropna()
            
            # 2. Calculate RV features on Historical Data
            self.train_df = self.__rv_calculation(model_data_5m)
            
            # 3. Determine Scaling Parameters (Min/Max from Training Set)
            self.train_min = self.train_df.min()
            self.train_max = self.train_df.max()
            
            # 4. Prepare Training Data
            # Scale
            train_scaled = (self.train_df - self.train_min) / (self.train_max - self.train_min)
            
            # Target: Next day's RV (RV_daily shifted back by 1 day)
            # Logic: We want to predict t+1 using information up to t.
            train_scaled["Target"] = train_scaled["RV_daily"].shift(-1)
            train_scaled = train_scaled.dropna()
            
            # Features
            X = train_scaled[["RV_daily", "RV_weekly", "RV_monthly"]]
            X = sm.add_constant(X)
            y = train_scaled["Target"]
            
            # 5. Train Model
            self.model = sm.OLS(y, X).fit()
            print("VolatilityPipeline: Model trained successfully.")
            
        except Exception as e:
            print(f"Error initializing VolatilityPipeline: {e}")
            self.model = None

    def _get_latest_data(self) -> pd.DataFrame:
        """Fetch live 5-minute data from yfinance for SPY."""
        # 59 days is the max for 5m interval in yfinance
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=59)
        
        try:
            # Retrieve data
            df = yf.download("SPY", start=start_date, end=end_date, interval="5m", progress=False)
            
            if df.empty:
                return pd.DataFrame()

            # Handle MultiIndex columns if present (common in newer yfinance)
            if isinstance(df.columns, pd.MultiIndex):
                # If Ticker is the second level, drop it
                if df.columns.nlevels > 1:
                    df.columns = df.columns.droplevel(1)

            # Standardize columns
            df = df.rename(columns={
                "Close": "close", 
                "Volume": "volume", 
                "Open": "open", 
                "High": "high", 
                "Low": "low"
            })
            
            # Remove zero volume bars (market closed or glitches)
            df = df[df['volume'] > 0]
            
            # Timezone handling
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_convert("America/New_York").tz_localize(None)
                
            df.index.name = "date"
            
            return df
            
        except Exception as e:
            print(f"Error downloading data: {e}")
            return pd.DataFrame()
    
    def __rv_calculation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Realized Volatility components (Daily, Weekly, Monthly)
        using the user's specific formula.
        """
        df = df.copy()
        df["D"] = df.index.date
        
        # Calculate periods per day (Per)
        # We transform count per group back to the original index
        df["Per"] = df.groupby("D")["close"].transform("count")
        
        # Calculate 'Returns' based on user formula: ((Delta Price) / Per)^2
        # Ensure we don't diff across days
        same_day_mask = df["D"] == df["D"].shift(1)
        price_diff = df["close"] - df["close"].shift(1)
        
        # Initialize Ret as NaN
        df["Ret"] = np.nan
        
        # Apply formula only where previous row is same day
        # Note: shift(1) is Previous. User code: df["close"] - df["close"].shift() which is diff(1)
        valid_indices = same_day_mask
        
        # Calculation: (Diff * (1/Per)) ** 2
        # Using .loc to safe assign
        term = price_diff * (1.0 / df["Per"])
        df.loc[valid_indices, "Ret"] = term ** 2
        
        # Aggregate to Daily RV
        # Sum of 'Ret' for each day
        rv = df.groupby("D")["Ret"].sum().to_frame("RV_daily")
        
        # Square Root as per user code implementation
        rv["RV_daily"] = np.sqrt(rv["RV_daily"])
        
        # Calculate Rolling Weekly (5 days) and Monthly (21 days) averages
        rv["RV_weekly"] = rv["RV_daily"].rolling(window=5).mean()
        rv["RV_monthly"] = rv["RV_daily"].rolling(window=21).mean()
        
        # Drop NaN values generated by rolling windows
        rv.dropna(inplace=True)
        
        return rv

    def get_latest_data(self) -> dict:
        """
        Fetch latest data, calculate RV, and predict next day's volatility.
        """
        if self.model is None:
            return {"error": "Model not initialized"}

        curr_data = self._get_latest_data()
        
        if curr_data.empty:
            return {"error": "No data retrieved"}

        # Calculate RV features for the live data
        curr_rv = self.__rv_calculation(curr_data)
        
        if curr_rv.empty:
             return {"error": "Not enough live data for rolling windows"}

        # Scale using TRAINING statistics
        # We must use the same min/max from training to make valid predictions
        curr_rv_scaled = (curr_rv - self.train_min) / (self.train_max - self.train_min)
        
        # Get the most recent day's features
        last_row = curr_rv_scaled.iloc[[-1]][["RV_daily", "RV_weekly", "RV_monthly"]]
        
        # Add constant for OLS
        last_row['const'] = 1.0
        
        # Reorder columns to match model params (order matters for matrix mult)
        last_row = last_row[self.model.params.index]
        
        # Predict
        prediction = self.model.predict(last_row)[0]
        
        return {"volatility_prediction": float(prediction)}
