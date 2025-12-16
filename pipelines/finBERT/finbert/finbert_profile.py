from __future__ import absolute_import, division, print_function

import random

import pandas as pd
from torch.nn import MSELoss, CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
    TensorDataset)
from torch.optim import AdamW
# PyTorch
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torch.ao.quantization


from tqdm import tqdm
from tqdm import trange
from nltk.tokenize import sent_tokenize
from finbert.utils import *
import numpy as np
import logging

from transformers.optimization import get_linear_schedule_with_warmup
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
from .profile_utils import get_model_size_mb, get_profiler_activities, print_profiler_results, setup_nltk_data, print_device_info
from finbert.finbert import FinBert

import time

import copy

def _snapshot_requires_grad(model):
    return {name: p.requires_grad for name, p in model.named_parameters()}

def _restore_requires_grad(model, req):
    for name, p in model.named_parameters():
        if name in req:
            p.requires_grad = req[name]

def _snapshot_training_state(model, optimizer, scheduler, global_step, i, device):
    snap = {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "global_step": global_step,
        "i": i,
        # RNG states for determinism
        "py_random": random.getstate(),
        "np_random": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "requires_grad": _snapshot_requires_grad(model)
    }
    if device.type == "cuda":
        snap["torch_cuda"] = torch.cuda.get_rng_state_all()
    return snap

def _restore_training_state(model, optimizer, scheduler, snap, device):
    model.load_state_dict(snap["model"])
    _restore_requires_grad(model, snap["requires_grad"])
    optimizer.load_state_dict(snap["optimizer"])
    scheduler.load_state_dict(snap["scheduler"])
    random.setstate(snap["py_random"])
    np.random.set_state(snap["np_random"])
    torch.random.set_rng_state(snap["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(snap["torch_cuda"])
    return snap["global_step"], snap["i"]

def _prof_event_time_ms(key_averages, key: str, prefer_cuda: bool) -> float:
    """Return total time (ms) for a profiler event key from `prof.key_averages()`."""
    for e in key_averages:
        if getattr(e, "key", None) == key:
            cuda_total = getattr(e, "cuda_time_total", 0) or 0
            cpu_total = getattr(e, "cpu_time_total", 0) or 0
            cuda_ms = cuda_total / 1000.0 if cuda_total else 0.0
            cpu_ms = cpu_total / 1000.0 if cpu_total else 0.0
            if prefer_cuda and cuda_ms:
                return cuda_ms
            return cpu_ms
    return 0.0


def summarize_key_averages_ms(key_averages, keys, prefer_cuda: bool, prefix: str = ""):
    """Summarize selected profiler keys into a flat dict of `{prefix}{key}_ms: float`."""
    return {f"{prefix}{k}_ms": _prof_event_time_ms(key_averages, k, prefer_cuda) for k in keys}


def log_profiler_table(key_averages, name="profiler_results"):
    """Log profiler key averages as a W&B Table."""
    try:
        import wandb
        if wandb.run is None:
            return
        
        rows = []
        # key_averages is iterable of FunctionEventAvg
        for e in key_averages:
            rows.append({
                "key": e.key,
                "cpu_time_total_ms": e.cpu_time_total / 1000.0,
                "cuda_time_total_ms": e.cuda_time_total / 1000.0,
                "self_cpu_time_total_ms": e.self_cpu_time_total / 1000.0,
                "self_cuda_time_total_ms": e.self_cuda_time_total / 1000.0,
                "cpu_memory_usage": e.cpu_memory_usage,
                "cuda_memory_usage": e.cuda_memory_usage,
                "count": e.count,
            })
        
        # Sort by total time (cuda if available, else cpu)
        # But wandb table allows sorting in UI.
        
        table = wandb.Table(data=pd.DataFrame(rows))
        wandb.log({name: table})
        # print(f"✓ Logged {name} table to wandb")
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠ Failed to log profiler table: {e}")


class ProfiledFinBert(FinBert):
    """Extended FinBert class with profiling instrumentation.
    
    Note: GPU-specific profiling (ProfilerActivity.CUDA) only works with NVIDIA CUDA devices.
    For MPS (Apple Silicon), only CPU profiling is available, though actual computation runs on GPU.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.profile_results = {}
    
    def train(self, train_examples, model):
        """
        Trains the model with profiling instrumentation.
        """
        validation_examples = self.get_data('validation')
        global_step = 0
        self.validation_losses = []
        
        # Training
        train_dataloader = self.get_loader(train_examples, 'train')
        
        model.train()
        
        step_number = len(train_dataloader)
        
        # Setup profiler - CUDA profiling only works with NVIDIA GPUs, not MPS
        activities = [ProfilerActivity.CPU]
        if self.device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        
        print("\n" + "="*80)
        print("Starting Profiled Training")
        print(f"Device: {self.device}")
        print(f"Profiling activities: {activities}")
        if self.device.type == "mps":
            print("Note: MPS profiling shows CPU time only. Actual GPU execution time not separately tracked.")
        print("="*80 + "\n")
        
        i = 0

        profile_steps = int(getattr(self.config, "profile_train_steps", 20) or 20)
        # If someone passes a weird value, fall back to 20.
        if profile_steps < 1:
            profile_steps = 20

        scaler = torch.amp.GradScaler('cuda', enabled=self.config.use_amp)
        
        
        snap = _snapshot_training_state(
            model=model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            global_step=global_step,
            i=i,
            device=self.device,
        )
        
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False
        ) as prof:
            
            for epoch in trange(int(self.config.num_train_epochs), desc="Epoch"):
                model.train()
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                
                for step, batch in enumerate(tqdm(train_dataloader, desc='Iteration')):
                    
                    # Gradual unfreezing logic
                    if (self.config.gradual_unfreeze and i == 0):
                        for param in model.bert.parameters():
                            param.requires_grad = False
                    
                    if (step % (step_number // 3)) == 0:
                        i += 1
                    
                    if (self.config.gradual_unfreeze and i > 1 and i < self.config.encoder_no):
                        for k in range(i - 1):
                            try:
                                for param in model.bert.encoder.layer[self.config.encoder_no - 1 - k].parameters():
                                    param.requires_grad = True
                            except:
                                pass
                    
                    if (self.config.gradual_unfreeze and i > self.config.encoder_no + 1):
                        for param in model.bert.embeddings.parameters():
                            param.requires_grad = True
                    
                    # Data loading profiling
                    with record_function("data_transfer"):
                        batch = tuple(t.to(self.device) for t in batch)
                        input_ids, attention_mask, token_type_ids, label_ids, agree_ids = batch
                    
                    # Forward pass profiling
                    with record_function("forward_pass"):
                        with torch.amp.autocast('cuda', enabled=self.config.use_amp):
                            logits = model(input_ids, attention_mask, token_type_ids)[0]
                    
                    # Loss calculation profiling
                    with record_function("loss_calculation"):
                        with torch.amp.autocast('cuda', enabled=self.config.use_amp):
                            weights = self.class_weights.to(self.device)
                            if self.config.output_mode == "classification":
                                loss_fct = CrossEntropyLoss(weight=weights)
                                loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
                            elif self.config.output_mode == "regression":
                                loss_fct = MSELoss()
                                loss = loss_fct(logits.view(-1), label_ids.view(-1))
                            
                            if self.config.gradient_accumulation_steps > 1:
                                loss = loss / self.config.gradient_accumulation_steps
                    
                    # Backward pass profiling
                    with record_function("backward_pass"):
                        if self.config.use_amp:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                    
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
                    
                    # Optimizer step profiling
                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        with record_function("optimizer_step"):
                            if self.config.use_amp:
                                scaler.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                                scaler.step(self.optimizer)
                                scaler.update()
                            else:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                                self.optimizer.step()
                                
                            self.scheduler.step()
                            self.optimizer.zero_grad()
                            global_step += 1
                    
                    # Only profile first epoch to save time
                    if epoch == 0 and (step + 1) >= profile_steps:
                        break
                
                # Break after first epoch for profiling
                if epoch == 0:
                    print("\n" + "="*80)
                    print(f"Profiling complete for first epoch ({profile_steps} steps)")
                    print("Continuing full training without profiling...")
                    print("="*80 + "\n")
                    break
        
        # Print profiler results
        print("\n" + "="*80)
        print("PROFILING RESULTS - Training")
        print("="*80 + "\n")
        
        print("\nBy CPU Time:")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        
        if self.device.type == "cuda":
            print("\nBy CUDA Time:")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        
        print("\n" + "="*80 + "\n")
        
        # Store results
        ka = prof.key_averages()
        self.profile_results['training'] = ka
        self.profile_results['training_summary'] = summarize_key_averages_ms(
            ka,
            keys=["data_transfer", "forward_pass", "loss_calculation", "backward_pass", "optimizer_step"],
            prefer_cuda=(self.device.type == "cuda"),
            prefix="train_",
        )

        # Log full profiler table to W&B
        log_profiler_table(ka, "training_profile_table")
        
        global_step, i = _restore_training_state(
            model=model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            snap=snap,
            device=self.device,
        )


        self.optimizer.zero_grad(set_to_none=True)
        for epoch in trange(int(self.config.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(train_dataloader, desc='Iteration')):
                
                if (self.config.gradual_unfreeze and i == 0):
                    for param in model.bert.parameters():
                        param.requires_grad = False
                
                if (step % (step_number // 3)) == 0:
                    i += 1
                
                if (self.config.gradual_unfreeze and i > 1 and i < self.config.encoder_no):
                    for k in range(i - 1):
                        try:
                            for param in model.bert.encoder.layer[self.config.encoder_no - 1 - k].parameters():
                                param.requires_grad = True
                        except:
                            pass
                
                if (self.config.gradual_unfreeze and i > self.config.encoder_no + 1):
                    for param in model.bert.embeddings.parameters():
                        param.requires_grad = True
                
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, token_type_ids, label_ids, agree_ids = batch
                
                with torch.amp.autocast('cuda', enabled=self.config.use_amp):
                    logits = model(input_ids, attention_mask, token_type_ids)[0]
                    weights = self.class_weights.to(self.device)
                    
                    if self.config.output_mode == "classification":
                        loss_fct = CrossEntropyLoss(weight=weights)
                        loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
                    elif self.config.output_mode == "regression":
                        loss_fct = MSELoss()
                        loss = loss_fct(logits.view(-1), label_ids.view(-1))
                    
                    if self.config.gradient_accumulation_steps > 1:
                        loss = loss / self.config.gradient_accumulation_steps
                
                if self.config.use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.use_amp:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    # Log training metrics to W&B if available
                    try:
                        import wandb
                        if wandb.run is not None and global_step % 10 == 0:
                            wandb.log({
                                'train_loss': tr_loss / nb_tr_steps,
                                'learning_rate': self.optimizer.param_groups[0]['lr'],
                                'epoch': epoch,
                                'step': global_step
                            })
                    except ImportError:
                        pass
            
            # Validation
            validation_loader = self.get_loader(validation_examples, phase='eval')
            model.eval()
            
            valid_loss, valid_accuracy = 0, 0
            nb_valid_steps, nb_valid_examples = 0, 0
            
            for input_ids, attention_mask, token_type_ids, label_ids, agree_ids in tqdm(validation_loader, desc="Validating"):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                label_ids = label_ids.to(self.device)
                agree_ids = agree_ids.to(self.device)
                
                with torch.no_grad():
                    logits = model(input_ids, attention_mask, token_type_ids)[0]
                    
                    if self.config.output_mode == "classification":
                        loss_fct = CrossEntropyLoss(weight=weights)
                        tmp_valid_loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
                    elif self.config.output_mode == "regression":
                        loss_fct = MSELoss()
                        tmp_valid_loss = loss_fct(logits.view(-1), label_ids.view(-1))
                    
                    valid_loss += tmp_valid_loss.mean().item()
                    nb_valid_steps += 1
            
            valid_loss = valid_loss / nb_valid_steps
            self.validation_losses.append(valid_loss)
            print("Validation losses: {}".format(self.validation_losses))
            
            # Log validation loss
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({'val_loss': valid_loss, 'epoch': epoch})
            except ImportError:
                pass
            
            if valid_loss == min(self.validation_losses):
                try:
                    os.remove(self.config.model_dir / ('temporary' + str(best_model)))
                except:
                    print('No best model found')
                torch.save({'epoch': str(i), 'state_dict': model.state_dict()},
                           self.config.model_dir / ('temporary' + str(i)))
                best_model = i
        
        # Save the trained model
        checkpoint = torch.load(self.config.model_dir / ('temporary' + str(best_model)))
        model.load_state_dict(checkpoint['state_dict'])
        model_to_save = model.module if hasattr(model, 'module') else model
        output_model_file = os.path.join(self.config.model_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(self.config.model_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())
        os.remove(self.config.model_dir / ('temporary' + str(best_model)))
        
        
        
        
        return model




def profile_inference(text, model, write_to_csv=False, path=None, variant_name="unknown", use_gpu=False, gpu_name='cuda:0', batch_size=5, use_amp=False):
    """
    Profile model inference performance.
    
    Args:
        model: Model to profile
        text: Text to analyze
        variant_name: Name of the model variant
        use_gpu: Whether to use GPU
        gpu_name: GPU device name
        batch_size: Batch size for inference
    
    Returns:
        results_df: DataFrame with predictions
        metrics: Dictionary with performance metrics
    """
    from nltk.tokenize import sent_tokenize
    from finbert.utils import InputExample, convert_examples_to_features, softmax, chunks, get_device
    
    # Setup NLTK
    setup_nltk_data()
    
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Device selection
    if use_gpu:
        device = get_device(no_cuda=False)
        if device.type == "cuda" and gpu_name.startswith("cuda:"):
            device = torch.device(gpu_name)
    else:
        device = torch.device("cpu")

    # AMP autocast is only supported on CUDA in this codepath (torch.cuda.amp.*)
    use_amp = bool(use_amp and device.type == "cuda")
    
    print_device_info(device)
    
    # Check if model is already on device (e.g., BitsAndBytes quantized models)
    is_quantized = hasattr(model, 'is_loaded_in_8bit') and model.is_loaded_in_8bit
    is_quantized = is_quantized or (hasattr(model, 'is_loaded_in_4bit') and model.is_loaded_in_4bit)
    
    # Only move model if it's not already quantized and placed
    if not is_quantized:
        model = model.to(device)
    else:
        print(f"✓ Model already quantized and placed on device (skipping .to() call)")
        # For quantized models, get the actual device from model
        if hasattr(model, 'device'):
            device = model.device
        elif hasattr(model, 'hf_device_map'):
            # BitsAndBytes models have device_map
            device = torch.device('cuda:0')  # Usually on cuda:0
    
    
    label_list = ['positive', 'negative', 'neutral']
    label_dict = {0: 'positive', 1: 'negative', 2: 'neutral'}
    result = pd.DataFrame(columns=['sentence', 'logit', 'prediction', 'sentiment_score'])
    
    # Setup profiler
    activities = get_profiler_activities(device)
    
    total_inference_time = 0
    
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    ) as prof:
        
        with record_function("sentence_tokenization"):
            sentences = sent_tokenize(text)
        
        for batch in chunks(sentences, batch_size):
            with record_function("create_examples"):
                examples = [InputExample(str(i), sentence) for i, sentence in enumerate(batch)]
            
            with record_function("convert_to_features"):
                features = convert_examples_to_features(examples, label_list, 64, tokenizer)
            
            with record_function("prepare_tensors"):
                all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
                all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long).to(device)
                all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long).to(device)
            
            with torch.no_grad():
                # Remove the model_to_device profiling section for quantized models
                if not is_quantized:
                    with record_function("model_to_device"):
                        model = model.to(device)
                
                with record_function("inference_forward"):
                    start_time = time.time()
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        logits = model(input_ids=all_input_ids, attention_mask=all_attention_mask, token_type_ids=all_token_type_ids)[0]
                    total_inference_time += time.time() - start_time
                
                with record_function("postprocess_results"):
                    logits = softmax(np.array(logits.cpu()))
                    sentiment_score = pd.Series(logits[:, 0] - logits[:, 1])
                    predictions = np.squeeze(np.argmax(logits, axis=1))
                    
                    batch_result = {
                        'sentence': batch,
                        'logit': list(logits),
                        'prediction': predictions,
                        'sentiment_score': sentiment_score
                    }
                    
                    batch_result = pd.DataFrame(batch_result)
                    result = pd.concat([result, batch_result], ignore_index=True)
    
    # Print profiler results
    print_profiler_results(prof, device, title=f"Inference Profiling - {variant_name}")
    
    result['prediction'] = result.prediction.apply(lambda x: label_dict[x])
    if write_to_csv:
        result.to_csv(path, sep=',', index=False)
    
    # Extract profiler timings (prefer CUDA times when available on CUDA devices)
    ka = prof.key_averages()
    prefer_cuda = (device.type == "cuda")
    tokenization_time_ms = _prof_event_time_ms(ka, "sentence_tokenization", prefer_cuda)
    inference_forward_time_ms = _prof_event_time_ms(ka, "inference_forward", prefer_cuda)
    model_to_device_time_ms = _prof_event_time_ms(ka, "model_to_device", prefer_cuda)

    # Log full profiler table to W&B
    log_profiler_table(ka, "inference_profile_table")

    metrics = {
        'variant': variant_name,
        'total_sentences': len(sentences),
        'inference_time_ms': total_inference_time * 1000,
        'time_per_sentence_ms': (total_inference_time * 1000) / len(sentences) if len(sentences) else 0.0,
        'device': str(device),
        'is_quantized': is_quantized,
        'tokenization_time_ms': tokenization_time_ms,
        'inference_forward_time_ms': inference_forward_time_ms,
        'model_to_device_time_ms': model_to_device_time_ms,
    }
    
    print(f"\nInference Summary:")
    print(f"  Total sentences: {metrics['total_sentences']}")
    print(f"  Total inference time: {metrics['inference_time_ms']:.2f} ms")
    print(f"  Time per sentence: {metrics['time_per_sentence_ms']:.2f} ms")
    if is_quantized:
        print(f"  ✓ Quantized model profiled successfully")
    
    return result, metrics
