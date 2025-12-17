import pandas as pd
import numpy as np 
import warnings
import statsmodels.api as sm
import datetime as dt
import yfinance as yf
import os
from typing import Tuple
from pipelines.base import Pipeline

class VolatilityPipeline(Pipeline):
    def __init__(self):
        # Locate CSV relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, "SPY1min_clean.csv")
        
        # 1. Load and Resample Historical Data
        try:
            # Load only necessary columns
            model_data = pd.read_csv(csv_path)
            model_data['date'] = pd.to_datetime(model_data['date'])
            model_data.set_index('date', inplace=True)
            
            # Resample 1min -> 5min
            agg_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum' 
            }
            # Drop NaN rows
            self.full_hist_data = model_data.resample('5min').agg(agg_dict).dropna()
            
            # Initial Training on ALL history (Default behavior)
            self._train_model(self.full_hist_data)
            
            # Demo Mode State
            self.demo_mode = False
            self.demo_current_idx = 0
            self.ground_truth_rv = pd.Series() # Init as empty series
            
        except Exception as e:
            print(f"Error initializing VolatilityPipeline: {e}")
            self.model = None

    def _train_model(self, data: pd.DataFrame):
        try:
            # 2. Calculate RV features
            self.train_df = self.__rv_calculation(data)
            
            # 3. Determine Scaling Parameters
            self.train_min = self.train_df.min()
            self.train_max = self.train_df.max()
            
            # 4. Prepare Training Data
            train_scaled = (self.train_df - self.train_min) / (self.train_max - self.train_min)
            train_scaled["Target"] = train_scaled["RV_daily"].shift(-1)
            train_scaled = train_scaled.dropna()
            
            X = train_scaled[["RV_daily", "RV_weekly", "RV_monthly"]]
            X = sm.add_constant(X)
            y = train_scaled["Target"]
            
            # 5. Train Model
            self.model = sm.OLS(y, X).fit()
            print("VolatilityPipeline: Model trained successfully.")
            print(f"nobs={int(self.model.nobs)}  df_model={int(self.model.df_model)}  df_resid={int(self.model.df_resid)}")
            print(f"R2={self.model.rsquared:.4f}  AdjR2={self.model.rsquared_adj:.4f}")
            print(f"MSE(resid)={self.model.mse_resid:.6f}  RMSE={np.sqrt(self.model.mse_resid):.6f}")
            print(f"MAE(resid)={np.mean(np.abs(self.model.resid)):.6f}")
            print(f"AIC={self.model.aic:.2f}  BIC={self.model.bic:.2f}")
            print()
        except Exception as e:
            print(f"Error training model: {e}")

    def setup_demo_mode(self, test_ratio=0.2):
        """
        Activates Demo Mode.
        Splits historical data into Train (80%) and Test (20%).
        Retrains model on Train set.
        Sets internal pointer to start of Test set for simulation.
        """
        self.demo_mode = True
        
        n = len(self.full_hist_data)
        split_idx = int(n * (1 - test_ratio))
        
        # Split Data
        train_data = self.full_hist_data.iloc[:split_idx]
        # We keep full data accessible for windowing, but demo "starts" at split_idx
        self.demo_start_idx = split_idx
        self.demo_current_idx = split_idx
        
        print(f"Demo Mode: Training on {len(train_data)} samples, Testing on {n - split_idx} samples.")
        
        # Retrain on only the training portion
        self._train_model(train_data)
        
        # Pre-calculate Ground Truth for the Test Set (Actual RVs)
        # We calculate RV on the FULL dataset so we have continuity across the split
        full_rv = self.__rv_calculation(self.full_hist_data)
        # Target for day D is RV of D+1. 
        # So for a prediction made at time T (Day D), we compare to RV_daily(D+1).
        # We store this for easy lookup.
        self.ground_truth_rv = full_rv["RV_daily"].shift(-1)

    def _get_latest_data(self) -> pd.DataFrame:
        """Fetch live 5-minute data from yfinance OR simulate from history."""
        
        if self.demo_mode:
            # Simulate Data Stream
            # We return data from [CurrentIdx - Window] to [CurrentIdx]
            # Window: 59 days (approx 5min bars) -> 59 * 78 bars/day approx? 
            # Actually, let's just grab a large enough chunk to ensure rolling windows work.
            # 2 months ~ 4000 bars (assuming ~78 bars/day for 6.5h)
            lookback = 5000 
            
            start_idx = max(0, self.demo_current_idx - lookback)
            
            # Slice the history up to the current simulation time
            df_slice = self.full_hist_data.iloc[start_idx : self.demo_current_idx + 1]
            
            return df_slice
            
        else:
            # LIVE MODE (yfinance)
            end_date = dt.datetime.now()
            start_date = end_date - dt.timedelta(days=59)
            
            try:
                df = yf.download("SPY", start=start_date, end=end_date, interval="5m", progress=False)
                
                if df.empty:
                    return pd.DataFrame()

                if isinstance(df.columns, pd.MultiIndex):
                    if df.columns.nlevels > 1:
                        df.columns = df.columns.droplevel(1)

                df = df.rename(columns={
                    "Close": "close", 
                    "Volume": "volume", 
                    "Open": "open", 
                    "High": "high", 
                    "Low": "low"
                })
                
                df = df[df['volume'] > 0]
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


    def predict_har_vol(self, date) -> Tuple[float, float]:
        """"
        input: date to predict volatility on
        output: realized volatility prediction based on trained model
        """

        if self.model is None:
            raise ValueError("Model not initialized")

        if isinstance(date, str):
            target_date = pd.to_datetime(date).date()
        elif isinstance(date, dt.datetime):
            target_date = date.date()
        else:
            target_date = date
        
        prev_date = target_date - dt.timedelta(days=1)
        prev_end = dt.datetime.combine(prev_date, dt.time(23, 59, 59, 999999))
        lookback_start = prev_end - dt.timedelta(days=59)

        curr_data = self.full_hist_data.loc[lookback_start:prev_end]
        if curr_data.empty:
            raise ValueError("No intraday data found for requested date window")
        
        curr_rv = self.__rv_calculation(curr_data)
        if curr_rv.empty:
            raise ValueError("Not enough data to compute RV features (need rolling windows)")
        if prev_date not in curr_rv.index:
            raise ValueError("No RV features found for previous date")


        curr_rv_scaled = (curr_rv - self.train_min) / (self.train_max - self.train_min)

        last_row = curr_rv_scaled.loc[[prev_date], ["RV_daily", "RV_weekly", "RV_monthly"]]
        last_row["const"] = 1.0
        last_row = last_row[self.model.params.index]

        prediction_for_target_date = float(self.model.predict(last_row).iloc[0])

        true_prev_rv = float(curr_rv.loc[prev_date, "RV_daily"])

        return prediction_for_target_date, true_prev_rv


    def get_latest_data(self) -> dict:
        """
        Fetch latest data, calculate RV, and predict next day's volatility.
        Demo Mode: Auto-increments time step and returns Ground Truth.
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
        
        result = {"volatility_prediction": float(prediction)}
        
        if self.demo_mode:
            # Add Ground Truth
            # The current simulation time is the last timestamp of curr_data
            current_sim_time = curr_data.index[-1]
            current_sim_date = current_sim_time.date()
            
            result["timestamp"] = str(current_sim_time)
            
            # Ground Truth Logic:
            # We predicted 'Target' which is RV of DAY+1.
            # We check if we have the actual RV for this date in our ground_truth map.
            # Note: The mapping calculates Target = Shift(-1).
            # So looking up 'current_sim_date' in ground_truth_rv gives the TRUE target (RV of tomorrow).
            
            if self.ground_truth_rv is not None and current_sim_date in self.ground_truth_rv.index:
                actual = self.ground_truth_rv.loc[current_sim_date]
                result["actual_next_day_rv"] = float(actual)
            else:
                result["actual_next_day_rv"] = None
                
            # Advance Pointer
            # We assume 1 tick per call
            self.demo_current_idx += 1
            if self.demo_current_idx >= len(self.full_hist_data):
                result["demo_ended"] = True
                
        return result
