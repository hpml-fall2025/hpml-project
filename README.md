# FinBERT Sentiment-Driven Volatility Forecasting

High-performance ML pipeline combining sentiment analysis of financial news with realized volatility forecasting.

## Overview

This project implements a sentiment-driven volatility forecasting system with two main components:

1. **FinBERT**: Fine-tuned BERT model for financial sentiment analysis (continuous scores)
2. **Alpha Network** (Phase 2): Learned weighting mechanism to combine sentiment signals with HAR-RV volatility forecasts

## Project Structure

```
hpml-project/
├── data/
│   ├── raw/                    # Raw datasets
│   ├── processed/              # Processed features
│   └── splits/                 # Train/val/test splits
├── models/
│   ├── finbert/                # FinBERT checkpoints
│   └── alpha_net/              # Alpha network (Phase 2)
├── src/
│   ├── data_collection.py      # Data fetching
│   ├── data_loader.py          # PyTorch datasets
│   ├── finbert_model.py        # Model architecture
│   ├── train_finbert.py        # Training loop
│   └── sentiment_inference.py  # Inference pipeline
├── config.py                   # Configuration
├── main.py                     # Main orchestration script
└── requirements.txt            # Dependencies
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Configure NewsAPI

Edit `config.py` and set your NewsAPI key:

```python
NEWS_API_KEY = "your_api_key_here"
```

If not configured, sample headlines will be generated for testing.

## Usage

### Quick Start - Run Full Pipeline

```bash
python main.py --all
```

This runs all three steps:
1. Data collection
2. FinBERT training
3. Sentiment inference

### Run Individual Steps

```bash
# Data collection only
python main.py --data

# Training only
python main.py --train

# Inference only
python main.py --inference
```

### Alternative: Run Modules Directly

```bash
# Data collection
python src/data_collection.py

# Training
python src/train_finbert.py

# Inference
python src/sentiment_inference.py
```

## Configuration

Edit `config.py` to customize:

- **Data**: Date ranges, ticker symbol (SPY)
- **Model**: BERT variant, dropout rates, layer freezing
- **Training**: Batch size, learning rate, epochs, gradient accumulation
- **Device**: Automatically selects MPS (Mac) / CUDA / CPU

### Key Hyperparameters

```python
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 32
NUM_EPOCHS = 4
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
USE_AMP = True  # Mixed precision training
```

## Pipeline Details

### Phase 1: Data Collection

**Inputs:**
- Financial PhraseBank (HuggingFace)
- SPY price data (yfinance)
- Financial headlines (NewsAPI or sample data)

**Outputs:**
- `data/raw/financial_phrasebank.csv`
- `data/raw/realized_volatility.csv`
- `data/raw/financial_headlines.csv`
- `data/processed/aligned_dataset.csv`
- `data/splits/{train,val,test}.csv`

### Phase 2: FinBERT Training

**Architecture:**
- Base: `bert-base-uncased` (110M parameters)
- Head: Linear regression layer (continuous sentiment output)
- Loss: MSE
- Optimization: AdamW + linear warmup + gradient accumulation

**Training Features:**
- Mixed precision (MPS/CUDA)
- Gradient accumulation
- Learning rate warmup
- Best model checkpointing

**Outputs:**
- `models/finbert/best_model/`
- `models/finbert/checkpoint_epoch_N/`
- `models/finbert/training_history.json`

### Phase 3: Sentiment Inference

**Process:**
1. Load trained FinBERT model
2. Batch inference on all headlines
3. Aggregate daily sentiment features:
   - `mean_square_sentiment`: E[s²]
   - `sentiment_variance`: Var(s)
   - `headline_count`: Number of headlines

**Outputs:**
- `data/processed/sentiment_features.csv`
- `data/processed/sentiment_features_{train,val,test}.csv`

## Model Performance

After training, review:

```bash
# Training history
cat models/finbert/training_history.json

# Validation metrics
# - MSE: Mean squared error on sentiment prediction
# - Correlation: Correlation between predicted and actual sentiment
```

## Next Steps (Phase 2)

1. **Implement Alpha Network**
   - Input: Sentiment features + market regime
   - Output: α ∈ [0,1] weight
   - Architecture: Simple MLP

2. **Integrate with HAR-RV**
   - Get HAR-RV forecasts from teammates
   - Combined forecast: `RV = α * RV_sentiment + (1-α) * RV_HAR`

3. **Training**
   - Loss: MSE on realized volatility
   - Optimize end-to-end

## Optimization Experiments

To explore FinBERT training optimizations, modify `config.py`:

```python
# Batch size tuning
BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 1

# Learning rate scheduling
WARMUP_RATIO = 0.06  # Try: 0.06, 0.1, 0.15

# Dropout tuning
HIDDEN_DROPOUT_PROB = 0.2  # Try: 0.1, 0.2, 0.3

# Layer freezing
FREEZE_ENCODER_LAYERS = 6  # Freeze first 6 layers
```

## Hardware Requirements

- **Minimum**: CPU (slow)
- **Recommended**: 
  - Mac M1/M2 with MPS acceleration
  - NVIDIA GPU with CUDA
- **Memory**: 8GB+ RAM
- **Storage**: ~2GB for data + models

## Troubleshooting

### Issue: MPS not available
**Solution**: Falls back to CPU automatically

### Issue: Out of memory
**Solution**: Reduce batch size or increase gradient accumulation steps

```python
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
```

### Issue: Data collection fails
**Solution**: Check NewsAPI key or use sample data (automatic fallback)

## Development

### Test Individual Components

```bash
# Test model
python src/finbert_model.py

# Test data loader
python src/data_loader.py
```

## Citation

Based on FinBERT paper:
```
@article{araci2019finbert,
  title={FinBERT: Financial Sentiment Analysis with Pre-trained Language Models},
  author={Araci, Dogu},
  journal={arXiv preprint arXiv:1908.10063},
  year={2019}
}
```

## License

MIT

