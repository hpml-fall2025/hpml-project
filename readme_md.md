# FinBERT-CNN Volatility Forecasting

Implementation of CNN-based volatility forecasting using FinBERT embeddings, based on the paper "Realised Volatility Forecasting: Machine Learning via Financial Word Embedding" (Rahimikia et al., 2024).

## Architecture

```
Headline Text → FinBERT Embeddings → CNN (1,2,3-grams) → Max Pool → FCNN → RV Prediction
```

**Key Features:**
- FinBERT for financial-domain embeddings (frozen)
- Multi-filter CNN capturing unigram, bigram, trigram patterns
- Global max pooling for feature extraction
- Fully connected layers for regression
- MSE loss with L2 regularization

## Project Structure

```
project/
├── config.py           # Hyperparameters and configuration
├── data_loader.py      # Data loading and preprocessing
├── model.py            # FinBERT-CNN architecture
├── train.py            # Training loop and checkpointing
├── evaluate.py         # Evaluation metrics and visualization
├── main.py             # Entry point
├── requirements.txt    # Python dependencies
└── README.md          # This file

data/
└── headlines_rv.csv   # Your data (see format below)

checkpoints/           # Model checkpoints (auto-created)
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Create `data/headlines_rv.csv` with columns:
- `date`: Date of the headline (YYYY-MM-DD)
- `headline`: News headline text
- `realized_volatility`: Next-day RV value

Example:
```csv
date,headline,realized_volatility
2020-01-01,Apple reports strong Q4 earnings,2.34
2020-01-02,Fed keeps interest rates unchanged,1.87
```

**Important:** Data should be sorted by date! The code preserves temporal order in train/val/test splits.

## Usage

### Training

Train a new model:
```bash
python main.py --mode train
```

This will:
- Load and split data (70/15/15 train/val/test)
- Train for 50 epochs (default)
- Save checkpoints to `checkpoints/`
- Evaluate on test set
- Generate plots: `training_curves.png`, `test_predictions.png`

### Evaluation

Evaluate a trained model:
```bash
python main.py --mode eval --checkpoint checkpoints/best_model.pt
```

### Inference

Predict RV for new headlines:
```bash
python main.py --mode inference --checkpoint checkpoints/best_model.pt
```

(Edit the example headlines in `main.py` or extend to read from file)

## Configuration

Edit `config.py` to customize:

```python
# Model
num_filters = 50              # Filters per CNN layer
filter_sizes = [1, 2, 3]      # Unigram, bigram, trigram
dropout_rate = 0.5

# Training
batch_size = 32
learning_rate = 1e-4
num_epochs = 50
weight_decay = 3.0            # L2 regularization

# Data
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
```

## Model Details

### FinBERT Embeddings
- **Model**: `ProsusAI/finbert` (pre-trained on financial text)
- **Output**: 768-dimensional embeddings per token
- **Max Length**: 512 tokens (default)
- **Frozen**: Embeddings are not fine-tuned (faster training)

### CNN Architecture
- **Filters**: 50 filters each for 1-gram, 2-gram, 3-gram
- **Activation**: ReLU
- **Pooling**: Global max pooling
- **Regularization**: Dropout (0.5) + L2 weight decay (3.0)

### Output
- **Task**: Regression (next-day realized volatility)
- **Loss**: Mean Squared Error (MSE)
- **Metrics**: MSE, RMSE, MAE, QLIKE, R²

## Evaluation Metrics

Following the paper, we report:
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error  
- **MAE**: Mean Absolute Error
- **QLIKE**: Quasi-likelihood (volatility-specific metric)
- **R²**: Coefficient of determination

## Results

Example output after training:
```
Test Set Metrics:
----------------------------------------
  MSE       : 0.123456
  RMSE      : 0.351362
  MAE       : 0.234567
  QLIKE     : 0.012345
  R2        : 0.789012
----------------------------------------
```

## Extending the Code

### Adding Custom Loss Functions
Edit `train.py`:
```python
# Example: Add QLIKE loss
def qlike_loss(y_pred, y_true):
    epsilon = 1e-8
    return torch.mean(
        (y_true / (y_pred + epsilon)) - 
        torch.log((y_true + epsilon) / (y_pred + epsilon)) - 1
    )
```

### Changing Filter Sizes
Edit `config.py`:
```python
filter_sizes = [1, 2, 3, 4]  # Add 4-grams
num_filters = 100             # More filters
```

### Fine-tuning FinBERT
Edit `model.py`:
```python
# In FinBERTCNN.__init__()
# Comment out the freezing:
# for param in self.finbert.parameters():
#     param.requires_grad = False

# Now FinBERT will be fine-tuned (slower but may improve)
```

## Next Steps (For Your Project)

This implementation provides the baseline NLP pipeline. For your project goals:

1. **Alpha Weighting**: 
   - Add confidence estimation (e.g., prediction variance)
   - Implement ensemble with HAR model: `final_pred = alpha * HAR + (1-alpha) * NLP`
   - Learn alpha based on uncertainty or validation performance

2. **Optimization Techniques**:
   - Model quantization (INT8)
   - Knowledge distillation
   - Batch processing for throughput
   - ONNX export for deployment

3. **Performance Metrics**:
   - Measure inference latency
   - Track throughput (predictions/sec)
   - Compare vs HAR baseline

## Troubleshooting

**CUDA Out of Memory:**
- Reduce `batch_size` in `config.py`
- Use CPU: `device = "cpu"`

**Poor Performance:**
- Check data quality (missing values, outliers)
- Verify temporal ordering is preserved
- Try more epochs or different learning rates
- Unfreeze FinBERT for fine-tuning (slower)

**Import Errors:**
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Check Python version (3.8+)

## References

- Paper: "Realised Volatility Forecasting: Machine Learning via Financial Word Embedding" (Rahimikia et al., 2024)
- FinBERT: https://huggingface.co/ProsusAI/finbert
- Original paper implementation: Uses custom Word2Vec embeddings

## License

MIT License (or specify your license)
