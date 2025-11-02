"""
Main script for FinBERT-CNN volatility forecasting
Recreating the paper's approach with FinBERT embeddings

Usage:
    python main.py --mode train
    python main.py --mode eval --checkpoint checkpoints/best_model.pt
"""
import argparse
import torch
import numpy as np
import random
from pathlib import Path

from config import Config
from data_loader import create_dataloaders
from model import create_model
from train import Trainer, load_checkpoint
from evaluate import evaluate_model, print_metrics, plot_predictions, plot_training_curves


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_mode(config):
    """Training mode"""
    print("="*60)
    print("TRAINING MODE")
    print("="*60)
    
    # Set seed
    set_seed(config.random_seed)
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Create model
    print("\nCreating model...")
    model = create_model(config)
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # Train
    train_losses, val_losses = trainer.train()
    
    # Plot training curves
    plot_training_curves(
        train_losses, 
        val_losses, 
        save_path="training_curves.png"
    )
    
    # Evaluate on test set with best model
    print("\nEvaluating on test set...")
    best_model_path = Path(config.checkpoint_dir) / "best_model.pt"
    model = load_checkpoint(model, best_model_path)
    
    test_metrics, test_preds, test_actuals = evaluate_model(model, test_loader, config)
    print_metrics(test_metrics, "Test")
    
    # Plot predictions
    plot_predictions(
        test_actuals, 
        test_preds, 
        save_path="test_predictions.png"
    )
    
    print("\nTraining complete!")


def eval_mode(config, checkpoint_path):
    """Evaluation mode"""
    print("="*60)
    print("EVALUATION MODE")
    print("="*60)
    
    # Set seed
    set_seed(config.random_seed)
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Create model
    print("\nCreating model...")
    model = create_model(config)
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    model = load_checkpoint(model, checkpoint_path)
    
    # Evaluate on all sets
    print("\nEvaluating on train set...")
    train_metrics, _, _ = evaluate_model(model, train_loader, config)
    print_metrics(train_metrics, "Train")
    
    print("\nEvaluating on validation set...")
    val_metrics, _, _ = evaluate_model(model, val_loader, config)
    print_metrics(val_metrics, "Validation")
    
    print("\nEvaluating on test set...")
    test_metrics, test_preds, test_actuals = evaluate_model(model, test_loader, config)
    print_metrics(test_metrics, "Test")
    
    # Plot predictions
    plot_predictions(
        test_actuals, 
        test_preds, 
        save_path="eval_predictions.png"
    )
    
    print("\nEvaluation complete!")


def inference_mode(config, checkpoint_path, headlines):
    """
    Inference mode for new headlines
    
    Args:
        config: Configuration object
        checkpoint_path: Path to model checkpoint
        headlines: List of headline strings
    """
    print("="*60)
    print("INFERENCE MODE")
    print("="*60)
    
    # Set seed
    set_seed(config.random_seed)
    
    # Create model
    print("\nCreating model...")
    model = create_model(config)
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    model = load_checkpoint(model, checkpoint_path)
    model.eval()
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.finbert_model)
    
    # Make predictions
    print(f"\nMaking predictions for {len(headlines)} headlines...")
    predictions = []
    
    with torch.no_grad():
        for headline in headlines:
            # Tokenize
            encoding = tokenizer(
                headline,
                max_length=config.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(config.device)
            attention_mask = encoding['attention_mask'].to(config.device)
            
            # Predict
            rv_pred = model(input_ids, attention_mask)
            predictions.append(rv_pred.item())
    
    # Print results
    print("\nResults:")
    print("-" * 60)
    for i, (headline, pred) in enumerate(zip(headlines, predictions), 1):
        print(f"\n{i}. {headline[:80]}...")
        print(f"   Predicted RV: {pred:.4f}")
    print("-" * 60)
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description="FinBERT-CNN Volatility Forecasting")
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['train', 'eval', 'inference'], 
        default='train',
        help='Mode: train, eval, or inference'
    )
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        default=None,
        help='Path to checkpoint for eval/inference mode'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default=None,
        help='Path to custom config file (optional)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = Config()
    
    # Override config if custom config provided
    if args.config:
        # Load custom config (implement as needed)
        pass
    
    # Check device availability
    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        config.device = "cpu"
    
    print(f"\nUsing device: {config.device}")
    
    # Run appropriate mode
    if args.mode == 'train':
        train_mode(config)
    
    elif args.mode == 'eval':
        if args.checkpoint is None:
            args.checkpoint = str(Path(config.checkpoint_dir) / "best_model.pt")
        eval_mode(config, args.checkpoint)
    
    elif args.mode == 'inference':
        if args.checkpoint is None:
            args.checkpoint = str(Path(config.checkpoint_dir) / "best_model.pt")
        
        # Example headlines
        example_headlines = [
            "Apple reports record Q4 earnings, beating analyst expectations",
            "Federal Reserve raises interest rates by 0.25%, markets react",
            "Tech stocks plunge amid recession fears and inflation concerns"
        ]
        
        inference_mode(config, args.checkpoint, example_headlines)


if __name__ == "__main__":
    main()
