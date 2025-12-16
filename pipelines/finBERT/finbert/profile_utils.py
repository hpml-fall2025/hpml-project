import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import torch.ao.quantization

from contextlib import nullcontext
from typing import Any

from .finbert import FinBert

import pandas as pd
import time

import torch.nn.functional as F
from sklearn.metrics import f1_score


def timed_eval(
    *, finbert: FinBert, model: torch.nn.Module, examples, use_amp: bool
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluation loop with optional CUDA autocast + timing (kept small for notebook use).
    
    Helper function for profiling evaluation performance.
    """
    loader = finbert.get_loader(examples, phase="eval")
    device = finbert.device

    model.eval()
    preds: list[np.ndarray] = []
    labels: list[int] = []

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, token_type_ids, label_ids, _agree_ids = batch

            logits = model(input_ids=input_ids, attention_mask=attention_mask)[0]

            preds.extend(logits.detach().cpu().numpy())
            labels.extend(label_ids.detach().cpu().numpy().tolist())

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    wall_s = time.perf_counter() - start
    n = len(labels)

    results_df = pd.DataFrame({"predictions": preds, "labels": labels})

    timing = {
        "eval_wall_s": float(wall_s),
        "eval_num_samples": int(n),
        "eval_samples_per_s": float(n / wall_s) if wall_s > 0 else float("inf"),
    }
    return results_df, timing

def get_model_size_mb(model):
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def get_profiler_activities(device):
    """Get appropriate profiler activities based on device"""
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    return activities


def print_profiler_results(prof, device, title="Profiling Results"):
    """Pretty print profiler results"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")
    
    print("By CPU Time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    
    if device.type == "cuda":
        print("\nBy CUDA Time:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    print(f"\n{'='*80}\n")


def setup_nltk_data():
    """Download necessary NLTK data"""
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        pass


def print_device_info(device):
    """Print device information"""
    print(f"\n{'='*80}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(device)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    elif device.type == "mps":
        print("Note: MPS profiling shows CPU time only. Actual GPU execution time not separately tracked.")
    print(f"{'='*80}\n")


def quantize_int8_model(model, calibration_loader=None, device='cuda'):
    """
    Apply INT8 post-training quantization using TorchAO for GPU inference.
    
    Args:
        model: The model to quantize
        calibration_loader: Optional data loader for calibration (not used for dynamic quantization)
        device: Device to run quantization on
    
    Returns:
        Quantized model ready for GPU inference
    """
    try:
        from torchao.quantization import quantize_, int8_dynamic_activation_int4_weight
        
        # Move model to GPU for quantization
        model = model.to(device)
        model.eval()
        
        # Apply dynamic INT8 activation with INT4 weight quantization
        # This provides good compression with minimal accuracy loss
        quantize_(model, int8_dynamic_activation_int4_weight())
        
        print("✓ Applied TorchAO INT8 dynamic quantization")
        return model
        
    except ImportError:
        print("⚠ TorchAO not available, falling back to torch.ao.quantization")
        # Fallback to standard PyTorch dynamic quantization (CPU-optimized)
        quantized_model = torch.ao.quantization.quantize_dynamic(
            model.cpu(),
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        print("✓ Applied torch.ao dynamic quantization (CPU-optimized)")
        return quantized_model.to(device)


print("✓ Helper utilities loaded")
