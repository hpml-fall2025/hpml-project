# Dashboard: Realized Variance Monitoring

The **dashboard** folder contains a Streamlit-based real-time monitoring application that visualizes realized variance predictions using two complementary pipelines: FinBERT sentiment analysis and HAR (Heterogeneous Autoregressive) volatility modeling.

## Overview

The dashboard combines:
- **News Pipeline**: Uses FinBERT to analyze financial news headlines and convert sentiment scores into volatility estimates
- **Volatility Pipeline**: Uses HAR-RV statistical models to predict realized variance from historical price data
- **Combined Model**: Dynamically weights both signals to produce a unified volatility prediction

The application displays real-time metrics, interactive charts, and allows you to adjust model parameters on the fly.

## Prerequisites

Before running the dashboard, ensure you have:

1. **Python 3.8+** installed
2. **All project dependencies** installed (see main `requirements.txt`)
3. **FinBERT model** trained and available at `pipelines/finBERT/models/sentiment/`
4. **News headlines data** available at `data/headlines.csv`
5. **Historical price data** available at `SPY1min_clean.csv` (root directory)

## Installation

1. **Install Dependencies**

   From the project root directory, in your venv:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Model Files**

   Ensure the FinBERT sentiment model exists:
   ```bash
   ls pipelines/finBERT/models/sentiment/
   # Should contain: config.json, pytorch_model.bin
   ```

3. **Verify Data Files**

   Ensure required data files exist:
   ```bash
   ls data/headlines.csv
   ls SPY1min_clean.csv
   ```

## Running the Dashboard

1. **Navigate to Project Root**

   ```bash
   cd /path/to/hpml-project
   ```

2. **Launch Streamlit**

   ```bash
   streamlit run dashboard/app.py
   ```

   The dashboard will automatically:
   - Initialize the News Pipeline (loads FinBERT model)
   - Initialize the Volatility Pipeline (loads and processes SPY data)
   - Start fetching and displaying data

3. **Access the Dashboard**

   The application will open in your default web browser at `http://localhost:8501`

## Demo Mode (Currently the one we use)

When enabled, Demo Mode:
- Splits historical data into 80% training / 20% testing
- Retrains the volatility model on the training set
- Simulates predictions on the test set with ground truth comparison
- Displays hypothetical news headlines for FinBERT inference synchronized with simulation time
