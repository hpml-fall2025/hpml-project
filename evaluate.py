"""
Evaluation metrics for volatility forecasting
Following the paper's evaluation approach
"""
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def evaluate_model(model, data_loader, config):
    """
    Evaluate model on a dataset
    
    Args:
        model: Trained model
        data_loader: DataLoader for evaluation
        config: Configuration object
        
    Returns:
        metrics: Dictionary of evaluation metrics
        predictions: Array of predictions
        actuals: Array of actual values
    """
    model.eval()
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating")
        for batch in pbar:
            # Move to device
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            rv_true = batch['rv'].numpy()
            
            # Forward pass
            rv_pred = model(input_ids, attention_mask)
            rv_pred = rv_pred.cpu().numpy().flatten()
            
            predictions.extend(rv_pred)
            actuals.extend(rv_true)
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    
    # QLIKE (as used in the paper)
    # QLIKE = (RV_true / RV_pred) - log(RV_true / RV_pred) - 1
    # Avoid division by zero
    epsilon = 1e-8
    qlike = np.mean(
        (actuals / (predictions + epsilon)) - 
        np.log((actuals + epsilon) / (predictions + epsilon)) - 1
    )
    
    # R-squared
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'QLIKE': qlike,
        'R2': r2
    }
    
    return metrics, predictions, actuals


def print_metrics(metrics, dataset_name="Test"):
    """Pretty print evaluation metrics"""
    print(f"\n{dataset_name} Set Metrics:")
    print("-" * 40)
    for metric_name, value in metrics.items():
        print(f"  {metric_name:10s}: {value:.6f}")
    print("-" * 40)


def plot_predictions(actuals, predictions, save_path=None):
    """
    Plot actual vs predicted volatility
    
    Args:
        actuals: Array of actual RV values
        predictions: Array of predicted RV values
        save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Scatter plot
    axes[0, 0].scatter(actuals, predictions, alpha=0.5)
    axes[0, 0].plot([actuals.min(), actuals.max()], 
                     [actuals.min(), actuals.max()], 
                     'r--', lw=2)
    axes[0, 0].set_xlabel('Actual RV')
    axes[0, 0].set_ylabel('Predicted RV')
    axes[0, 0].set_title('Actual vs Predicted RV')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Time series (first 200 samples)
    n_plot = min(200, len(actuals))
    x = np.arange(n_plot)
    axes[0, 1].plot(x, actuals[:n_plot], label='Actual', alpha=0.7)
    axes[0, 1].plot(x, predictions[:n_plot], label='Predicted', alpha=0.7)
    axes[0, 1].set_xlabel('Sample')
    axes[0, 1].set_ylabel('RV')
    axes[0, 1].set_title(f'Time Series (first {n_plot} samples)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Residuals
    residuals = actuals - predictions
    axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Residual (Actual - Predicted)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residual Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_training_curves(train_losses, val_losses, save_path=None):
    """
    Plot training and validation loss curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Optional path to save plot
    """
    plt.figure(figsize=(10, 6))
    epochs = np.arange(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, label='Train Loss', marker='o', markersize=4)
    plt.plot(epochs, val_losses, label='Val Loss', marker='s', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()
