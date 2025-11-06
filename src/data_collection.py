"""
Data collection module for FinBERT sentiment analysis pipeline.
Handles downloading Financial PhraseBank, calculating realized volatility, 
collecting financial headlines, and aligning datasets.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datasets import load_dataset
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
import config


def download_financial_phrasebank():
    """
    Download Financial PhraseBank dataset from HuggingFace.
    Uses 'sentences_allagree' split for higher quality labels.
    """
    print("Downloading Financial PhraseBank dataset...")
    
    try:
        # Load dataset
        dataset = load_dataset("financial_phrasebank", "sentences_allagree")
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(dataset['train'])
        
        # Map labels to continuous sentiment scores
        df['sentiment_score'] = df['label'].map(config.LABEL_TO_SENTIMENT)
        
        # Save to CSV
        output_path = config.RAW_DATA_DIR / "financial_phrasebank.csv"
        df.to_csv(output_path, index=False)
        
        print(f"✓ Financial PhraseBank downloaded: {len(df)} sentences")
        print(f"  Saved to: {output_path}")
        print(f"  Label distribution:")
        print(df['label'].value_counts().sort_index())
        
        return df
    
    except Exception as e:
        print(f"✗ Error downloading Financial PhraseBank: {e}")
        raise


def calculate_realized_volatility():
    """
    Fetch SPY price data and calculate realized volatility metrics.
    Returns daily, weekly, and monthly RV for HAR model compatibility.
    """
    print(f"\nCalculating realized volatility for {config.TICKER}...")
    
    try:
        # Add buffer for rolling window calculations
        start_date = datetime.strptime(config.START_DATE, "%Y-%m-%d") - timedelta(days=30)
        end_date = datetime.strptime(config.END_DATE, "%Y-%m-%d") + timedelta(days=1)
        
        print(f"  Fetching data from {start_date.date()} to {end_date.date()}...")
        
        # Download data
        ticker_obj = yf.Ticker(config.TICKER)
        data = ticker_obj.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="1d"
        )
        
        if data.empty:
            raise ValueError(f"No data returned for {config.TICKER}")
        
        # Calculate log returns
        data['returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Calculate realized volatility metrics
        # Daily RV (22-day rolling window, annualized)
        data['RV_daily'] = data['returns'].rolling(
            window=config.RV_WINDOW_DAILY
        ).std() * np.sqrt(252)
        
        # Weekly RV (5-day rolling window, annualized)
        data['RV_weekly'] = data['returns'].rolling(
            window=config.RV_WINDOW_WEEKLY
        ).std() * np.sqrt(252)
        
        # Monthly RV (22-day rolling window, annualized)
        data['RV_monthly'] = data['RV_daily']  # Same as daily for 22-day window
        
        # Keep only necessary columns
        rv_data = data[['RV_daily', 'RV_weekly', 'RV_monthly']].copy()
        rv_data = rv_data.dropna()
        
        # Filter to actual date range
        rv_data = rv_data[
            (rv_data.index >= config.START_DATE) & 
            (rv_data.index <= config.END_DATE)
        ]
        
        # Save to CSV
        output_path = config.RAW_DATA_DIR / "realized_volatility.csv"
        rv_data.to_csv(output_path)
        
        print(f"✓ Realized volatility calculated: {len(rv_data)} trading days")
        print(f"  Saved to: {output_path}")
        print(f"  Date range: {rv_data.index.min().date()} to {rv_data.index.max().date()}")
        print(f"  RV_daily stats: mean={rv_data['RV_daily'].mean():.4f}, std={rv_data['RV_daily'].std():.4f}")
        
        return rv_data
    
    except Exception as e:
        print(f"✗ Error calculating realized volatility: {e}")
        raise


def collect_financial_headlines():
    """
    Collect financial headlines using NewsAPI or fall back to sample data.
    Note: NewsAPI free tier has limitations (100 requests/day, 1 month history).
    """
    print("\nCollecting financial headlines...")
    
    # Check if NewsAPI key is configured
    if config.NEWS_API_KEY is None:
        print("  ⚠ No NewsAPI key configured. Generating sample headlines for testing...")
        return generate_sample_headlines()
    
    try:
        from newsapi import NewsApiClient
        
        newsapi = NewsApiClient(api_key=config.NEWS_API_KEY)
        headlines_data = []
        
        # NewsAPI free tier: limited to last 30 days
        # For full historical data, would need premium tier or alternative sources
        start = datetime.strptime(config.START_DATE, '%Y-%m-%d')
        end = datetime.strptime(config.END_DATE, '%Y-%m-%d')
        
        # Collect in weekly batches to manage API limits
        current = start
        while current <= end:
            week_end = min(current + timedelta(days=7), end)
            
            try:
                articles = newsapi.get_everything(
                    q='(stock market OR S&P 500 OR earnings OR financial)',
                    sources='bloomberg,reuters,cnbc,financial-times',
                    from_param=current.strftime('%Y-%m-%d'),
                    to=week_end.strftime('%Y-%m-%d'),
                    language='en',
                    sort_by='relevancy',
                    page_size=100
                )
                
                for article in articles.get('articles', []):
                    published = article['publishedAt'][:10]  # YYYY-MM-DD
                    headlines_data.append({
                        'date': published,
                        'headline': article['title'],
                        'source': article['source']['name']
                    })
                
                print(f"  Collected {len(articles.get('articles', []))} articles for {current.date()}")
                
            except Exception as e:
                print(f"  Warning: Error for {current.date()}: {e}")
            
            current = week_end + timedelta(days=1)
        
        # Convert to DataFrame and aggregate by date
        if not headlines_data:
            print("  ⚠ No headlines collected. Generating sample data...")
            return generate_sample_headlines()
        
        df = pd.DataFrame(headlines_data)
        df['date'] = pd.to_datetime(df['date'])
        
        # Aggregate headlines by date
        headlines_grouped = df.groupby('date').agg({
            'headline': lambda x: list(x),
            'source': 'count'
        }).rename(columns={'source': 'num_headlines'})
        
        output_path = config.RAW_DATA_DIR / "financial_headlines.csv"
        headlines_grouped.to_csv(output_path)
        
        print(f"✓ Headlines collected: {len(headlines_grouped)} days")
        print(f"  Saved to: {output_path}")
        print(f"  Total headlines: {df.shape[0]}")
        
        return headlines_grouped
    
    except ImportError:
        print("  ⚠ newsapi-python not installed. Install with: pip install newsapi-python")
        print("  Generating sample headlines for testing...")
        return generate_sample_headlines()
    
    except Exception as e:
        print(f"  ⚠ Error collecting headlines: {e}")
        print("  Generating sample headlines for testing...")
        return generate_sample_headlines()


def generate_sample_headlines():
    """
    Generate sample financial headlines for testing when NewsAPI is unavailable.
    """
    print("  Generating sample headlines...")
    
    # Sample headline templates with varied sentiment
    positive_headlines = [
        "Stock market rallies on strong earnings reports",
        "S&P 500 reaches new all-time high",
        "Tech sector leads gains as investor confidence grows",
        "Federal Reserve signals continued economic support",
        "Corporate earnings exceed analyst expectations"
    ]
    
    negative_headlines = [
        "Markets decline amid inflation concerns",
        "S&P 500 falls on weak economic data",
        "Investors grow cautious as volatility increases",
        "Tech stocks tumble on regulatory fears",
        "Economic uncertainty weighs on market sentiment"
    ]
    
    neutral_headlines = [
        "S&P 500 closes mixed in quiet trading session",
        "Markets await Federal Reserve policy decision",
        "Trading volumes remain steady in financial sector",
        "Analysts maintain cautious outlook on earnings",
        "Stock market shows little movement ahead of data release"
    ]
    
    all_templates = positive_headlines + negative_headlines + neutral_headlines
    
    # Generate headlines for date range
    start = pd.to_datetime(config.START_DATE)
    end = pd.to_datetime(config.END_DATE)
    date_range = pd.date_range(start, end, freq='D')
    
    headlines_data = []
    np.random.seed(config.RANDOM_SEED)
    
    for date in date_range:
        # Skip weekends (approximately)
        if date.weekday() >= 5:
            continue
        
        # Generate 3-7 headlines per day
        num_headlines = np.random.randint(3, 8)
        day_headlines = np.random.choice(all_templates, size=num_headlines, replace=True)
        
        for headline in day_headlines:
            headlines_data.append({
                'date': date,
                'headline': headline,
                'source': 'Sample'
            })
    
    df = pd.DataFrame(headlines_data)
    
    # Aggregate by date
    headlines_grouped = df.groupby('date').agg({
        'headline': lambda x: list(x),
        'source': 'count'
    }).rename(columns={'source': 'num_headlines'})
    
    output_path = config.RAW_DATA_DIR / "financial_headlines.csv"
    headlines_grouped.to_csv(output_path)
    
    print(f"✓ Sample headlines generated: {len(headlines_grouped)} days")
    print(f"  Saved to: {output_path}")
    
    return headlines_grouped


def align_datasets():
    """
    Align headlines and realized volatility by date.
    Handles missing data (weekends/holidays) appropriately.
    """
    print("\nAligning datasets...")
    
    try:
        # Load datasets
        rv_data = pd.read_csv(
            config.RAW_DATA_DIR / "realized_volatility.csv",
            index_col=0,
            parse_dates=True
        )
        
        headlines_data = pd.read_csv(
            config.RAW_DATA_DIR / "financial_headlines.csv",
            index_col=0,
            parse_dates=True
        )
        
        # Merge on date (inner join - only trading days with both data)
        aligned = rv_data.join(headlines_data, how='inner')
        
        # Ensure headlines column is properly formatted
        # (CSV storage may have converted lists to strings)
        import ast
        aligned['headline'] = aligned['headline'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        
        # Save aligned dataset
        output_path = config.PROCESSED_DATA_DIR / "aligned_dataset.csv"
        aligned.to_csv(output_path)
        
        print(f"✓ Datasets aligned: {len(aligned)} days")
        print(f"  Saved to: {output_path}")
        print(f"  Date range: {aligned.index.min().date()} to {aligned.index.max().date()}")
        print(f"  Average headlines per day: {aligned['num_headlines'].mean():.1f}")
        
        return aligned
    
    except Exception as e:
        print(f"✗ Error aligning datasets: {e}")
        raise


def create_train_val_test_splits():
    """
    Create time-based train/validation/test splits.
    Critical for financial data to prevent look-ahead bias.
    """
    print("\nCreating train/val/test splits...")
    
    try:
        # Load aligned dataset
        data = pd.read_csv(
            config.PROCESSED_DATA_DIR / "aligned_dataset.csv",
            index_col=0,
            parse_dates=True
        )
        
        # Create splits based on date ranges
        train_data = data[
            (data.index >= config.TRAIN_START) & 
            (data.index <= config.TRAIN_END)
        ]
        
        val_data = data[
            (data.index >= config.VAL_START) & 
            (data.index <= config.VAL_END)
        ]
        
        test_data = data[
            (data.index >= config.TEST_START) & 
            (data.index <= config.TEST_END)
        ]
        
        # Save splits
        train_data.to_csv(config.SPLITS_DIR / "train.csv")
        val_data.to_csv(config.SPLITS_DIR / "val.csv")
        test_data.to_csv(config.SPLITS_DIR / "test.csv")
        
        print(f"✓ Splits created:")
        print(f"  Train: {len(train_data)} days ({config.TRAIN_START} to {config.TRAIN_END})")
        print(f"  Val:   {len(val_data)} days ({config.VAL_START} to {config.VAL_END})")
        print(f"  Test:  {len(test_data)} days ({config.TEST_START} to {config.TEST_END})")
        
        return train_data, val_data, test_data
    
    except Exception as e:
        print(f"✗ Error creating splits: {e}")
        raise


def main():
    """
    Run complete data collection pipeline.
    """
    print("=" * 60)
    print("FinBERT Data Collection Pipeline")
    print("=" * 60)
    
    try:
        # Create directories if they don't exist
        config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Download Financial PhraseBank
        download_financial_phrasebank()
        
        # Step 2: Calculate realized volatility
        calculate_realized_volatility()
        
        # Step 3: Collect headlines
        collect_financial_headlines()
        
        # Step 4: Align datasets
        align_datasets()
        
        # Step 5: Create splits
        create_train_val_test_splits()
        
        print("\n" + "=" * 60)
        print("✓ Data collection complete!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Data collection failed: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()

