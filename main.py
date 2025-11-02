"""
Main execution script for 5-minute Realized Variance Prediction System.
"""

import sys
from config import TICKER, TARGET_DATE, LOOKBACK_DAYS
import pull
import calc
import produce


def main() -> None:
    """
    Execute the HAR-RV prediction pipeline.
    """
    try:
        print("=" * 80)
        print("5-Minute Realized Variance Prediction System")
        print("HAR-RV Model with Simulated Streaming Architecture")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Ticker:       {TICKER}")
        print(f"  Target Date:  {TARGET_DATE}")
        print(f"  Lookback:     {LOOKBACK_DAYS} days")
        print()
        
        print(f"Step 1: Fetching historical data for {TICKER}...")
        data = pull.get_historical_data(TICKER, TARGET_DATE, LOOKBACK_DAYS)
        print(f"✓ Data fetch complete\n")
        
        print(f"Step 2: Simulating streaming predictions for {TARGET_DATE}...")
        predictions = calc.simulate_streaming(data, TARGET_DATE)
        print(f"✓ Predictions complete\n")
        
        print("Step 3: Displaying results...")
        produce.output_results(predictions)
        print("✓ Pipeline complete")
        
    except ValueError as e:
        print(f"\n❌ Validation Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Unexpected Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

