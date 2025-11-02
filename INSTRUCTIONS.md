# Quick Start Guide

Get up and running in 5 minutes!

## 1. Setup Environment

```bash
# Clone/download the code
cd your-project-directory

# Install dependencies
pip install -r requirements.txt
```

## 2. Generate Sample Data (Optional)

If you don't have real data yet:

```bash
# Generate 1000 samples
python generate_sample_data.py --n_samples 1000 --output data/headlines_rv.csv
```

This creates a CSV with synthetic financial headlines and RV values.

## 3. Train the Model

```bash
# Train with default settings
python main.py --mode train
```

This will:
- Load data from `data/headlines_rv.csv`
- Split into train (70%), val (15%), test (15%)
- Train for 50 epochs
- Save best model to `checkpoints/best_model.pt`
- Show training progress with progress bars
- Generate plots

**Expected output:**
```
Loading data...
Data splits:
  Train: 700 samples (2020-01-01 to 2021-11-17)
  Val:   150 samples (2021-11-18 to 2022-05-12)
  Test:  150 samples (2022-05-13 to 2022-10-06)

Creating model...
Model created:
  Total parameters: 92,456,789
  Trainable parameters: 123,456
  Frozen FinBERT parameters: 92,333,333

Starting training for 50 epochs...
Epoch 1/50 [Train]: 100%|██████████| 22/22 [00:15<00:00]
Epoch 1/50 [Val]:   100%|██████████| 5/5 [00:02<00:00]
Epoch 1/50 - Train Loss: 5.234567, Val Loss: 4.876543

...

Training complete! Best validation loss: 1.234567
```

## 4. Evaluate

```bash
# Evaluate the best model
python main.py --mode eval --checkpoint checkpoints/best_model.pt
```

**Expected output:**
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

## 5. Run Inference

```bash
# Predict on new headlines
python main.py --mode inference --checkpoint checkpoints/best_model.pt
```

This runs on example headlines hardcoded in `main.py`. To use custom headlines, edit `main.py`:

```python
# In main() function, replace example_headlines with your own:
example_headlines = [
    "Your custom headline here",
    "Another headline",
]
```

## Next Steps

### Customize Configuration

Edit `config.py` to tune hyperparameters:

```python
# Model architecture
num_filters = 100              # Increase model capacity
filter_sizes = [1, 2, 3, 4]   # Add 4-grams

# Training
batch_size = 64               # Larger batches (if GPU allows)
learning_rate = 5e-5          # Fine-tune learning rate
num_epochs = 100              # Train longer
```

### Use Your Real Data

Replace `data/headlines_rv.csv` with your data. Required format:

```csv
date,headline,realized_volatility
2020-01-01,Apple reports strong earnings,2.34
2020-01-02,Fed maintains rates,1.87
```

**Important:**
- Must have columns: `date`, `headline`, `realized_volatility`
- Dates should be sorted chronologically
- RV values should be positive

### Monitor Training

Watch the training curves in `training_curves.png`:
- Train loss should decrease smoothly
- Val loss should track train loss (if diverging, reduce model complexity or increase regularization)

Check predictions in `test_predictions.png`:
- Scatter plot: points should be close to diagonal
- Time series: predicted should track actual
- Residuals: should be roughly normal

### Debug Common Issues

**Model overfitting (val loss > train loss):**
- Increase `dropout_rate` (try 0.6-0.7)
- Increase `weight_decay` (try 5.0)
- Decrease `num_filters` (try 25)

**Model underfitting (both losses high):**
- Increase `num_filters` (try 100)
- Add more filter sizes (try [1,2,3,4,5])
- Train longer (`num_epochs = 100`)
- Decrease `weight_decay` (try 1.0)

**Training too slow:**
- Reduce `batch_size` if GPU memory issue
- Use CPU if no GPU: `device = "cpu"` in config
- Reduce `num_filters`

**Poor predictions:**
- Check data quality (outliers, missing values)
- Verify temporal ordering preserved
- Try unfreezing FinBERT (slower but may help)

## File Overview

```
├── config.py               # All hyperparameters
├── data_loader.py          # Data pipeline
├── model.py                # Model architecture
├── train.py                # Training loop
├── evaluate.py             # Metrics & plots
├── main.py                 # Run training/eval/inference
├── generate_sample_data.py # Create synthetic data
└── requirements.txt        # Dependencies
```

## Tips

1. **Start small**: Use 1000 samples to verify everything works
2. **Monitor GPU usage**: `nvidia-smi` to check memory
3. **Save checkpoints**: Default saves every 5 epochs
4. **Use validation set**: Don't evaluate on test until final run
5. **Experiment systematically**: Change one hyperparameter at a time

## Getting Help

Check `README.md` for detailed documentation.

Common issues:
- **ImportError**: Run `pip install -r requirements.txt`
- **FileNotFoundError**: Generate data with `generate_sample_data.py`
- **CUDA OOM**: Reduce `batch_size` or use CPU
- **Poor results**: Check data quality and try different hyperparameters
