
import sys
import os
import shutil
import time
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, RandomSampler
import pandas as pd
import numpy as np

# Add parent directory to path to import finbert
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# NLTK setup to avoid errors
import nltk
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt')
        nltk.download('punkt_tab')
    except:
        pass

from finbert.finbert import Config, FinBert
from transformers import AutoModelForSequenceClassification

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_steps(model, dataloader, device, optimizer, num_steps=10):
    model.train()
    model.to(device)
    
    start_time = time.time()
    step_times = []
    
    iterator = iter(dataloader)
    for i in range(num_steps):
        step_start = time.time()
        try:
            batch = next(iterator)
        except StopIteration:
            break
            
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, token_type_ids, label_ids, agree_ids = batch
        
        # Forward
        logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        
        # Loss (CrossEntropy for classification)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 3), label_ids.view(-1))
        
        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        step_end = time.time()
        step_times.append(step_end - step_start)
        print(f"  Step {i+1}/{num_steps} - Loss: {loss.item():.4f} - Time: {step_times[-1]*1000:.2f}ms")

    total_time = time.time() - start_time
    avg_time = sum(step_times) / len(step_times) if step_times else 0
    return avg_time, total_time

def run_comparison(data_path, output_dir):
    print(f"Starting FinBERT LoRA Verification using data at: {data_path}")
    
    # Ensure data exists
    train_file = os.path.join(data_path, 'train.csv')
    if not os.path.exists(train_file):
        print(f"ERROR: 'train.csv' not found in {data_path}")
        print("Please ensure the directory contains the training data as expected by FinBERT.")
        return

    # Setup output paths
    base_output = Path(output_dir)
    if os.path.exists(base_output):
        shutil.rmtree(base_output)
    os.makedirs(base_output, exist_ok=True)
    
    results = {}

    # --- 1. Baseline Model ---
    print("\n" + "="*40)
    print("Testing Baseline Model (Full Finetuning)")
    print("="*40)
    
    bertmodel_base = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    config_base = Config(
        data_dir=str(data_path),
        bert_model=bertmodel_base,
        model_dir=str(base_output / "baseline"),
        use_lora=False,
        discriminate=True, 
        gradual_unfreeze=True,
        train_batch_size=16  # Smaller batch to fit memory if needed
    )
    finbert_base = FinBert(config_base)
    finbert_base.prepare_model(label_list=['positive','negative','neutral'])
    
    # Get Data
    print("Loading Baseline Data...")
    train_data_base = finbert_base.get_data('train')
    train_loader_base = finbert_base.get_loader(train_data_base, 'train')
    
    model_base = finbert_base.create_the_model()
    params_base = count_parameters(model_base)
    print(f"Baseline Trainable Parameters: {params_base:,}")
    
    print("Running Baseline Training Steps...")
    avg_base, total_base = train_steps(model_base, train_loader_base, finbert_base.device, finbert_base.optimizer)
    results['baseline'] = {'params': params_base, 'avg_time': avg_base}

    # Clean up to free memory
    del model_base, finbert_base, bertmodel_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- 2. LoRA Model ---
    print("\n" + "="*40)
    print("Testing LoRA Model (Adapter Finetuning)")
    print("="*40)
    
    bertmodel_lora = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    config_lora = Config(
        data_dir=str(data_path),
        bert_model=bertmodel_lora,
        model_dir=str(base_output / "lora"),
        use_lora=True,
        lora_r=8,
        discriminate=True, # Should be disabled
        gradual_unfreeze=True, # Should be disabled
        train_batch_size=16
    )
    finbert_lora = FinBert(config_lora)
    finbert_lora.prepare_model(label_list=['positive','negative','neutral'])
    
    # Get Data
    print("Loading LoRA Data...")
    train_data_lora = finbert_lora.get_data('train')
    train_loader_lora = finbert_lora.get_loader(train_data_lora, 'train')
    
    model_lora = finbert_lora.create_the_model()
    params_lora = count_parameters(model_lora)
    print(f"LoRA Trainable Parameters: {params_lora:,}")
    
    # Verify flags
    print(f"LoRA Config Check - Discriminate: {finbert_lora.config.discriminate} (Expected: False)")
    print(f"LoRA Config Check - Gradual Unfreeze: {finbert_lora.config.gradual_unfreeze} (Expected: False)")
    
    print("Running LoRA Training Steps...")
    avg_lora, total_lora = train_steps(model_lora, train_loader_lora, finbert_lora.device, finbert_lora.optimizer)
    results['lora'] = {'params': params_lora, 'avg_time': avg_lora}

    # --- Comparison Report ---
    print("\n" + "="*40)
    print("COMPARISON RESULTS")
    print("="*40)
    
    p_base = results['baseline']['params']
    p_lora = results['lora']['params']
    t_base = results['baseline']['avg_time']
    t_lora = results['lora']['avg_time']
    
    print(f"{'Metric':<20} | {'Baseline':<15} | {'LoRA':<15} | {'Reduction/Speedup':<15}")
    print("-" * 75)
    print(f"{'Trainable Params':<20} | {p_base:<15,} | {p_lora:<15,} | {p_lora/p_base*100:.2f}% (Size)")
    print(f"{'Avg Step Time (s)':<20} | {t_base:<15.4f} | {t_lora:<15.4f} | {t_base/t_lora:.2f}x (Speed)")
    
    if p_lora < p_base * 0.05:
        print("\nSUCCESS: LoRA parameters are significantly lower (<5%) than baseline.")
    else:
        print("\nWARNING: LoRA parameter reduction is less than expected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify FinBERT LoRA Integration")
    parser.add_argument("--data_dir", type=str, default="pipelines/finBERT/data/sentiment_data",
                        help="Path to directory containing train.csv")
    parser.add_argument("--output_dir", type=str, default="pipelines/finBERT/models/test_output",
                        help="Directory to save test outputs")
    
    args = parser.parse_args()
    
    run_comparison(args.data_dir, args.output_dir)
