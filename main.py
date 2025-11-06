"""
Main orchestration script for FinBERT sentiment analysis pipeline.
Runs complete workflow: data collection → training → inference
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

import config
from src import data_collection, train_finbert, sentiment_inference


def run_data_collection():
    """
    Run data collection pipeline.
    """
    print("\n" + "=" * 70)
    print("STEP 1: DATA COLLECTION")
    print("=" * 70)
    
    data_collection.main()


def run_training():
    """
    Run FinBERT training pipeline.
    """
    print("\n" + "=" * 70)
    print("STEP 2: FINBERT TRAINING")
    print("=" * 70)
    
    train_finbert.main()


def run_inference():
    """
    Run sentiment inference pipeline.
    """
    print("\n" + "=" * 70)
    print("STEP 3: SENTIMENT INFERENCE")
    print("=" * 70)
    
    sentiment_inference.main()


def run_full_pipeline():
    """
    Run complete pipeline end-to-end.
    """
    print("\n" + "=" * 70)
    print("FINBERT SENTIMENT PIPELINE - FULL EXECUTION")
    print("=" * 70)
    
    try:
        # Step 1: Data Collection
        run_data_collection()
        
        # Step 2: Training
        run_training()
        
        # Step 3: Inference
        run_inference()
        
        print("\n" + "=" * 70)
        print("✓ PIPELINE COMPLETE!")
        print("=" * 70)
        
        print("\nGenerated Files:")
        print(f"  - Raw data: {config.RAW_DATA_DIR}")
        print(f"  - Processed data: {config.PROCESSED_DATA_DIR}")
        print(f"  - Train/Val/Test splits: {config.SPLITS_DIR}")
        print(f"  - FinBERT model: {config.FINBERT_MODEL_DIR}")
        print(f"  - Sentiment features: {config.PROCESSED_DATA_DIR / 'sentiment_features.csv'}")
        
        print("\nNext Steps:")
        print("  1. Review sentiment features in data/processed/")
        print("  2. Implement alpha weighting network (Phase 2)")
        print("  3. Integrate with teammates' HAR-RV model")
        print("  4. Train combined volatility forecasting model")
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        raise


def main():
    """
    Main entry point with CLI argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="FinBERT Sentiment Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python main.py --all
  
  # Run individual steps
  python main.py --data          # Data collection only
  python main.py --train         # Training only
  python main.py --inference     # Inference only
  
  # Run multiple steps
  python main.py --data --train
        """
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run complete pipeline (data collection → training → inference)'
    )
    parser.add_argument(
        '--data',
        action='store_true',
        help='Run data collection step'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Run training step'
    )
    parser.add_argument(
        '--inference',
        action='store_true',
        help='Run inference step'
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not (args.all or args.data or args.train or args.inference):
        parser.print_help()
        return
    
    try:
        # Run full pipeline
        if args.all:
            run_full_pipeline()
        
        # Or run individual steps
        else:
            if args.data:
                run_data_collection()
            
            if args.train:
                run_training()
            
            if args.inference:
                run_inference()
        
        print("\n✓ Execution complete!")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

