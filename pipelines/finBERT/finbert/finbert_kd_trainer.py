from __future__ import absolute_import, division, print_function

import random

import pandas as pd
from torch.nn import MSELoss, CrossEntropyLoss, KLDivLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
    TensorDataset)
from torch.optim import AdamW
# PyTorch
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torch.ao.quantization
import torch.nn.functional as F


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
from finbert.finbert import Config

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


class KDFinBert():
    """Extended FinBert class with profiling instrumentation.
    
    Note: GPU-specific profiling (ProfilerActivity.CUDA) only works with NVIDIA CUDA devices.
    For MPS (Apple Silicon), only CPU profiling is available, though actual computation runs on GPU.
    """
    
    def __init__(self, config, teacher, student, alpha = 0.5, temperature = 2.0):
        self.profile_results = {}
        self.config = config
        self.teacher = teacher
        self.student = student
        self.alpha = alpha
        self.temperature = temperature
    
    def prepare_model(self, label_list):
        self.processors = {"finsent": FinSentProcessor}
        self.num_labels_task = {"finsent": 2}

        if self.config.local_rank == -1 or self.config.no_cuda:
            self.device = get_device(self.config.no_cuda)
            if self.device.type == "cuda":
                self.n_gpu = torch.cuda.device_count()
            else:
                self.n_gpu = 1 if self.device.type == "mps" else 0
        else:
            torch.cuda.set_device(self.config.local_rank)
            self.device = torch.device("cuda", self.config.local_rank)
            self.n_gpu = 1
            torch.distributed.init_process_group(backend="nccl")


        if self.config.gradient_accumulation_steps < 1:
            raise ValueError(
                f"Invalid gradient_accumulation_steps: {self.config.gradient_accumulation_steps}, should be >= 1"
            )

        self.config.train_batch_size = self.config.train_batch_size // self.config.gradient_accumulation_steps

        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if self.device.type == "cuda" and self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.config.seed)

        if os.path.exists(self.config.model_dir) and os.listdir(self.config.model_dir):
            raise ValueError(f"Output directory ({self.config.model_dir}) already exists and is not empty.")
        if not os.path.exists(self.config.model_dir):
            os.makedirs(self.config.model_dir)

        self.processor = self.processors["finsent"]()
        self.num_labels = len(label_list)
        self.label_list = label_list

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model, do_lower_case=self.config.do_lower_case
        )
        
    def get_data(self, phase):
        examples = self.processor.get_examples(self.config.data_dir, phase)
        self.num_train_optimization_steps = int(
            len(examples) / self.config.train_batch_size / self.config.gradient_accumulation_steps
        ) * self.config.num_train_epochs

        if phase == "train":
            train = pd.read_csv(os.path.join(self.config.data_dir, "train.csv"), sep="\t", index_col=False)
            labels = self.label_list
            class_weights = [train.shape[0] / train[train.label == label].shape[0] for label in labels]
            self.class_weights = torch.tensor(class_weights)

        return examples

    def create_the_model(self):
        """
        Sets up teacher/student on device, freezes teacher, and builds optimizer/scheduler for student.
        Keeps the FinBERT optimizer/scheduler pattern (no discriminate grouping here to keep it simple & stable).
        """
        teacher = self.teacher.to(self.device)
        student = self.student.to(self.device)

        # Teacher frozen
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        # Student trainable
        student.train()

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        
        #did non discriminative 
        param_optimizer = list(student.named_parameters())
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        self.num_warmup_steps = int(float(self.num_train_optimization_steps) * self.config.warm_up_proportion)

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_train_optimization_steps,
        )

        return teacher, student


    #logits: (batch_size, num_classes)
    #labels: (batch_size)
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        inputs is expected to include:
          input_ids, attention_mask, token_type_ids, labels
        """

        outputs_student = model(**inputs)

        if self.config.output_mode == "classification" and hasattr(self, "class_weights"):
            weights = self.class_weights.to(self.device)
            loss_fct = CrossEntropyLoss(weight=weights)
            student_loss = loss_fct(outputs_student.logits.view(-1, self.num_labels), inputs["labels"].view(-1))

        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)

        assert outputs_student.logits.size() == outputs_teacher.logits.size()

        # soften probabilities and compute distillation loss
        loss_function = KLDivLoss(reduction="batchmean")
        T = float(self.config.temperature)
        loss_logits = (
            loss_function(
                F.log_softmax(outputs_student.logits / T, dim=-1),
                F.softmax(outputs_teacher.logits / T, dim=-1),
            )
            * (T ** 2)
        )

        alpha = float(self.config.alpha)
        loss = alpha * student_loss + (1.0 - alpha) * loss_logits

        return (loss, outputs_student) if return_outputs else loss

    
    def train(self, train_examples, teacher, student):

        validation_examples = self.get_data("validation")
        global_step = 0
        self.validation_losses = []
        
        train_dataloader = self.get_loader(train_examples, "train")

        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        
        student.train()
        
        step_number = len(train_dataloader)

        activities = [ProfilerActivity.CPU]
        if self.device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        profile_steps = int(getattr(self.config, "profile_train_steps", 20) or 20)
        if profile_steps < 1:
            profile_steps = 20

        scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_amp)


        best_model = None

        # Snapshot: so profiling run doesn't change the "real" training trajectory
        i = 0
        snap = _snapshot_training_state(
            student=student,
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
            with_stack=False,
        ) as prof:

            for epoch in trange(int(self.config.num_train_epochs), desc="Epoch"):
                student.train()
                teacher.eval()

                for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                    with record_function("data_transfer"):
                        batch = tuple(t.to(self.device) for t in batch)
                        input_ids, attention_mask, token_type_ids, label_ids, agree_ids = batch

                        inputs = {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "token_type_ids": token_type_ids,
                            "labels": label_ids,
                        }

                    with record_function("forward_pass"):
                        # forward pass is inside compute_loss (student + teacher)
                        # but we keep this wrapper for consistent block timing
                        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                            loss = self.compute_loss(student, inputs, return_outputs=False)
                            if self.config.gradient_accumulation_steps > 1:
                                loss = loss / self.config.gradient_accumulation_steps

                    with record_function("backward_pass"):
                        if self.config.use_amp:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()

                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        with record_function("optimizer_step"):
                            if self.config.use_amp:
                                scaler.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                                scaler.step(self.optimizer)
                                scaler.update()
                            else:
                                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                                self.optimizer.step()

                            self.scheduler.step()
                            self.optimizer.zero_grad()
                            global_step += 1

                    # Stop after N optimizer steps (not micro-steps)
                    if epoch == 0 and global_step >= profile_steps:
                        break

                if epoch == 0 and global_step >= profile_steps:
                    break

        # Store profile summary like ProfiledFinBert
        ka = prof.key_averages()
        self.profile_results["training"] = ka
        self.profile_results["training_summary"] = summarize_key_averages_ms(
            ka,
            keys=["data_transfer", "forward_pass", "backward_pass", "optimizer_step"],
            prefer_cuda=(self.device.type == "cuda"),
            prefix="train_",
        )

        global_step, i = _restore_training_state(
            student=student,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            snap=snap,
            device=self.device,
        )

        self.optimizer.zero_grad(set_to_none=True)


        for epoch in trange(int(self.config.num_train_epochs), desc="Epoch"):
            student.train()
            teacher.eval()

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, token_type_ids, label_ids, agree_ids = batch

                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                    "labels": label_ids,
                }

                with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                    loss = self.compute_loss(student, inputs, return_outputs=False)
                    if self.config.gradient_accumulation_steps > 1:
                        loss = loss / self.config.gradient_accumulation_steps

                if self.config.use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.use_amp:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

            validation_loader = self.get_loader(validation_examples, phase="eval")
            student.eval()

            valid_loss = 0.0
            nb_valid_steps = 0

            weights = self.class_weights.to(self.device) if hasattr(self, "class_weights") else None

            for batch in tqdm(validation_loader, desc="Validating"):
                input_ids, attention_mask, token_type_ids, label_ids, agree_ids = (t.to(self.device) for t in batch)
                with torch.no_grad():
                    outputs = student(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    )
                    logits = outputs.logits

                    if self.config.output_mode == "classification":
                        if weights is not None:
                            loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
                        else:
                            loss_fct = torch.nn.CrossEntropyLoss()
                        tmp_valid_loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
                    else:
                        loss_fct = torch.nn.MSELoss()
                        tmp_valid_loss = loss_fct(logits.view(-1), label_ids.view(-1))

                    valid_loss += float(tmp_valid_loss.mean().item())
                    nb_valid_steps += 1

            valid_loss = valid_loss / max(1, nb_valid_steps)
            self.validation_losses.append(valid_loss)
            print("Validation losses:", self.validation_losses)

            if valid_loss == min(self.validation_losses):
                # remove previous best
                if best_model is not None:
                    try:
                        os.remove(os.path.join(self.config.model_dir, f"temporary{best_model}"))
                    except Exception:
                        pass

                torch.save(
                    {"epoch": str(epoch), "state_dict": student.state_dict()},
                    os.path.join(self.config.model_dir, f"temporary{epoch}"),
                )
                best_model = epoch

        ckpt = torch.load(os.path.join(self.config.model_dir, f"temporary{best_model}"), map_location="cpu")
        student.load_state_dict(ckpt["state_dict"])

        student_to_save = student.module if hasattr(student, "module") else student
        torch.save(student_to_save.state_dict(), os.path.join(self.config.model_dir, WEIGHTS_NAME))
        with open(os.path.join(self.config.model_dir, CONFIG_NAME), "w") as f:
            f.write(student_to_save.config.to_json_string())

        os.remove(os.path.join(self.config.model_dir, f"temporary{best_model}"))
        return student



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
                    with torch.cuda.amp.autocast(enabled=use_amp):
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


