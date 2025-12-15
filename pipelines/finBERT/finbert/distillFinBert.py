from __future__ import absolute_import, division, print_function

import random

import pandas as pd
from torch.nn import MSELoss, CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
    TensorDataset)
from torch.optim import AdamW
from tqdm import tqdm
from tqdm import trange
from nltk.tokenize import sent_tokenize
from finbert.utils import *
import numpy as np
import logging

import torch
from torch.profiler import profile, record_function, ProfilerActivity

from transformers.optimization import get_linear_schedule_with_warmup
from transformers import AutoTokenizer

from .profile_utils import get_model_size_mb, get_profiler_activities, print_profiler_results, setup_nltk_data, print_device_info

import time

import copy


from finbert.finbert import Config


logger = logging.getLogger(__name__)

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

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

class DistillFinBert(object):
    
    def __init__(self,
                 config):
        self.config = config
        self.profile_results = {}
    
    def _classifier_modules(self, model):
        mods = []
        if hasattr(model, "pre_classifier"):
            mods.append(model.pre_classifier)
        if hasattr(model, "classifier"):
            mods.append(model.classifier)
        return mods

    def _iter_named_params(self, module):
        return [] if module is None else list(module.named_parameters())

    def _forward_logits(self, model, input_ids, attention_mask, token_type_ids):
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits if hasattr(out, "logits") else out[0]

    def prepare_model(self, label_list):
        """
        Sets some of the components of the model: Dataset processor, number of labels, usage of gpu and distributed
        training, gradient accumulation steps and tokenizer.
        Parameters
        ----------
        label_list: list
            The list of labels values in the dataset. For example: ['positive','negative','neutral']
        """

        self.processors = {
            "finsent": FinSentProcessor
        }

        self.num_labels_task = {
            'finsent': 2
        }

        if self.config.local_rank == -1 or self.config.no_cuda:
            self.device = get_device(self.config.no_cuda)
            # Only count GPUs for CUDA (MPS doesn't support multi-GPU in the same way)
            if self.device.type == "cuda":
                self.n_gpu = torch.cuda.device_count()
            else:
                self.n_gpu = 1 if self.device.type == "mps" else 0
        else:
            # Distributed training only supported with CUDA
            torch.cuda.set_device(self.config.local_rank)
            self.device = torch.device("cuda", self.config.local_rank)
            self.n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl')
        logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            self.device, self.n_gpu, bool(self.config.local_rank != -1), self.config.fp16))

        if self.config.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                self.config.gradient_accumulation_steps))

        self.config.train_batch_size = self.config.train_batch_size // self.config.gradient_accumulation_steps

        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if self.device.type == "cuda" and self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.config.seed)


        # Set seed for CUDA devices (MPS doesn't have manual_seed_all)
        if self.device.type == "cuda" and self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.config.seed)

        if os.path.exists(self.config.model_dir) and os.listdir(self.config.model_dir):
            raise ValueError("Output directory ({}) already exists and is not empty.".format(self.config.model_dir))
        if not os.path.exists(self.config.model_dir):
            os.makedirs(self.config.model_dir)

        self.processor = self.processors['finsent']()
        self.num_labels = len(label_list)
        self.label_list = label_list

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model, do_lower_case=self.config.do_lower_case)

    def get_data(self, phase):
        """
        Gets the data for training or evaluation. It returns the data in the format that pytorch will process. In the
        data directory, there should be a .csv file with the name <phase>.csv
        Parameters
        ----------
        phase: str
            Name of the dataset that will be used in that phase. For example if there is a 'train.csv' in the data
            folder, it should be set to 'train'.
        Returns
        -------
        examples: list
            A list of InputExample's. Each InputExample is an object that includes the information for each example;
            text, id, label...
        """

        self.num_train_optimization_steps = None
        examples = None
        examples = self.processor.get_examples(self.config.data_dir, phase)
        self.num_train_optimization_steps = int(
            len(
                examples) / self.config.train_batch_size / self.config.gradient_accumulation_steps) * self.config.num_train_epochs

        if phase == 'train':
            train = pd.read_csv(os.path.join(self.config.data_dir, 'train.csv'), sep='\t', index_col=False)
            weights = list()
            labels = self.label_list

            class_weights = [train.shape[0] / train[train.label == label].shape[0] for label in labels]
            self.class_weights = torch.tensor(class_weights)

        return examples

    def create_the_model(self):
        """
        Creates the model. Sets the model to be trained and the optimizer.
        """

        model = self.config.bert_model

        model.to(self.device)

        # Prepare optimizer
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        lr = self.config.learning_rate
        dft_rate = 1.2
        
        enc_layers = list(model.distilbert.transformer.layer)
        L = len(enc_layers)

        if self.config.discriminate:
            # apply the discriminative fine-tuning. discrimination rate is governed by dft_rate.

            encoder_params = []
            for i in range(L):
                layer_i = enc_layers[i]
                encoder_decay = {
                    "params": [p for n, p in self._iter_named_params(layer_i) if not any(nd in n for nd in no_decay)],
                    'weight_decay': 0.01,
                    'lr': lr / (dft_rate ** (L - i))}
                encoder_nodecay = {
                    "params": [p for n, p in self._iter_named_params(layer_i) if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": lr / (dft_rate ** (L - i)),
                }
                encoder_params.extend([encoder_decay, encoder_nodecay])
            
            emb = model.distilbert.embeddings
            head_mods = self._classifier_modules(model)
            
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self._iter_named_params(emb) if not any(nd in n for nd in no_decay)],
                    "weight_decay": 0.01,
                    "lr": lr / (dft_rate ** (L + 1)),
                },
                {
                    "params": [p for n, p in self._iter_named_params(emb) if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": lr / (dft_rate ** (L + 1)),
                },
            ]
            
            for hm in head_mods:
                optimizer_grouped_parameters.extend(
                    [
                        {
                            "params": [p for n, p in self._iter_named_params(hm) if not any(nd in n for nd in no_decay)],
                            "weight_decay": 0.01,
                            "lr": lr,
                        },
                        {
                            "params": [p for n, p in self._iter_named_params(hm) if any(nd in n for nd in no_decay)],
                            "weight_decay": 0.0,
                            "lr": lr,
                        },
                    ]
                )

            optimizer_grouped_parameters.extend(encoder_params)
        else:
            param_optimizer = list(model.named_parameters())
            optimizer_grouped_parameters = [
                {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
                {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            ]

        schedule = "warmup_linear"

        self.num_warmup_steps = int(float(self.num_train_optimization_steps) * self.config.warm_up_proportion)

        self.optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.config.learning_rate)

        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=self.num_warmup_steps,
                                                    num_training_steps=self.num_train_optimization_steps)

        return model

    def get_loader(self, examples, phase):
        """
        Creates a data loader object for a dataset.
        Parameters
        ----------
        examples: list
            The list of InputExample's.
        phase: 'train' or 'eval'
            Determines whether to use random sampling or sequential sampling depending on the phase.
        Returns
        -------
        dataloader: DataLoader
            The data loader object.
        """

        features = convert_examples_to_features(examples, self.label_list,
                                                self.config.max_seq_length,
                                                self.tokenizer,
                                                self.config.output_mode)

        # Log the necessasry information
        logger.info("***** Loading data *****")
        logger.info("  Num examples = %d", len(examples))
        logger.info("  Batch size = %d", self.config.train_batch_size)
        logger.info("  Num steps = %d", self.num_train_optimization_steps)

        # Load the data, make it into TensorDataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

        if self.config.output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        elif self.config.output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

        try:
            all_agree_ids = torch.tensor([f.agree for f in features], dtype=torch.long)
        except:
            all_agree_ids = torch.tensor([0.0 for f in features], dtype=torch.long)

        data = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids, all_agree_ids)

        # Distributed, if necessary
        if phase == 'train':
            my_sampler = RandomSampler(data)
        elif phase == 'eval':
            my_sampler = SequentialSampler(data)

        dataloader = DataLoader(data, sampler=my_sampler, batch_size=self.config.train_batch_size)
        return dataloader

    def train(self, train_examples, model):
        validation_examples = self.get_data("validation")
        global_step = 0
        self.validation_losses = []

        train_dataloader = self.get_loader(train_examples, "train")
        model.train()

        step_number = len(train_dataloader)

        # profiler activities (mirrors your ProfiledFinBert: CUDA only when available)
        activities = [ProfilerActivity.CPU]
        if self.device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        profile_steps = int(getattr(self.config, "profile_train_steps", 20) or 20)
        if profile_steps < 1:
            profile_steps = 20

        scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_amp)

        # Snapshot so profiled warmup doesn't change the real training
        i = 0
        snap = _snapshot_training_state(
            model=model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            global_step=global_step,
            i=i,
            device=self.device,
        )

        # Distil has 6 layers; BERT has 12; clamp encoder_no to existing L
        enc_layers = list(model.distilbert.transformer.layer)
        L = len(enc_layers)
        encoder_no = int(getattr(self.config, "encoder_no", L) or L)
        encoder_no = max(1, min(encoder_no, L))

        # ----------------------------
        # Profiling run (first epoch, first N steps)
        # ----------------------------
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:

            for epoch in trange(int(self.config.num_train_epochs), desc="Epoch"):
                model.train()

                for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                    # --- gradual unfreeze (same logic, but correct backbone) ---
                    if self.config.gradual_unfreeze and i == 0:
                        for param in model.distilbert.parameters():
                            param.requires_grad = False
                       
                    if (step_number // 3) > 0 and (step % (step_number // 3)) == 0:
                        i += 1

                    if self.config.gradual_unfreeze and i > 1 and i < encoder_no:
                        for k in range(i - 1):
                            layer_idx = encoder_no - 1 - k
                            if 0 <= layer_idx < L:
                                for param in enc_layers[layer_idx].parameters():
                                    param.requires_grad = True

                    if self.config.gradual_unfreeze and i > encoder_no + 1:
                        for param in model.distilbert.embeddings.parameters():
                            param.requires_grad = True

                    with record_function("data_transfer"):
                        batch = tuple(t.to(self.device) for t in batch)
                        input_ids, attention_mask, token_type_ids, label_ids, _agree_ids = batch

                    with record_function("forward_pass"):
                        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                            logits = self._forward_logits(model, input_ids, attention_mask, token_type_ids)

                    with record_function("loss_calculation"):
                        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                            weights = self.class_weights.to(self.device)
                            if self.config.output_mode == "classification":
                                loss_fct = CrossEntropyLoss(weight=weights)
                                loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
                            else:
                                loss_fct = MSELoss()
                                loss = loss_fct(logits.view(-1), label_ids.view(-1))

                            if self.config.gradient_accumulation_steps > 1:
                                loss = loss / self.config.gradient_accumulation_steps

                    with record_function("backward_pass"):
                        if self.config.use_amp:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()

                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        with record_function("optimizer_step"):
                            if self.config.fp16:
                                lr_this_step = self.config.learning_rate * warmup_linear(
                                    global_step / self.num_train_optimization_steps, self.config.warm_up_proportion
                                )
                                for param_group in self.optimizer.param_groups:
                                    param_group["lr"] = lr_this_step

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

                    if epoch == 0 and (step + 1) >= profile_steps:
                        break

                if epoch == 0:
                    break

        # Store + summarize profile
        ka = prof.key_averages()
        self.profile_results["training"] = ka
        self.profile_results["training_summary"] = summarize_key_averages_ms(
            ka,
            keys=["data_transfer", "forward_pass", "loss_calculation", "backward_pass", "optimizer_step"],
            prefer_cuda=(self.device.type == "cuda"),
            prefix="train_",
        )

        # Restore state so real training starts identically to non-profiled baseline
        global_step, i = _restore_training_state(
            model=model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            snap=snap,
            device=self.device,
        )

        # ----------------------------
        # Real training (same as base FinDistilBert, but kept here so it matches your ProfiledFinBert flow)
        # ----------------------------
        self.optimizer.zero_grad(set_to_none=True)

        best_model = None

        for _epoch in trange(int(self.config.num_train_epochs), desc="Epoch"):
            model.train()

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                if self.config.gradual_unfreeze and i == 0:
                    for param in model.distilbert.parameters():
                        param.requires_grad = False
                        
                if (step_number // 3) > 0 and (step % (step_number // 3)) == 0:
                    i += 1

                if self.config.gradual_unfreeze and i > 1 and i < encoder_no:
                    for k in range(i - 1):
                        layer_idx = encoder_no - 1 - k
                        if 0 <= layer_idx < L:
                            for param in enc_layers[layer_idx].parameters():
                                param.requires_grad = True

                if self.config.gradual_unfreeze and i > encoder_no + 1:
                    for param in model.distilbert.embeddings.parameters():
                        param.requires_grad = True

                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, token_type_ids, label_ids, _agree_ids = batch

                with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                    logits = self._forward_logits(model, input_ids, attention_mask, token_type_ids)
                    weights = self.class_weights.to(self.device)

                    if self.config.output_mode == "classification":
                        loss_fct = CrossEntropyLoss(weight=weights)
                        loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
                    else:
                        loss_fct = MSELoss()
                        loss = loss_fct(logits.view(-1), label_ids.view(-1))

                    if self.config.gradient_accumulation_steps > 1:
                        loss = loss / self.config.gradient_accumulation_steps

                if self.config.use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.fp16:
                        lr_this_step = self.config.learning_rate * warmup_linear(
                            global_step / self.num_train_optimization_steps, self.config.warm_up_proportion
                        )
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = lr_this_step

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

            # --- Validation (same as your original) ---
            validation_loader = self.get_loader(validation_examples, phase="eval")
            model.eval()

            valid_loss = 0.0
            nb_valid_steps = 0

            for batch in tqdm(validation_loader, desc="Validating"):
                input_ids, attention_mask, token_type_ids, label_ids, _agree_ids = (t.to(self.device) for t in batch)
                with torch.no_grad():
                    logits = self._forward_logits(model, input_ids, attention_mask, token_type_ids)

                    if self.config.output_mode == "classification":
                        loss_fct = CrossEntropyLoss(weight=weights)
                        tmp_valid_loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
                    else:
                        loss_fct = MSELoss()
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
                    {"epoch": str(i), "state_dict": model.state_dict()},
                    os.path.join(self.config.model_dir, f"temporary{i}"),
                )
                best_model = i

        # Save the trained model
        checkpoint = torch.load(os.path.join(self.config.model_dir, f"temporary{best_model}"), map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])

        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(model_to_save.state_dict(), os.path.join(self.config.model_dir, WEIGHTS_NAME))
        with open(os.path.join(self.config.model_dir, CONFIG_NAME), "w") as f:
            f.write(model_to_save.config.to_json_string())

        os.remove(os.path.join(self.config.model_dir, f"temporary{best_model}"))
        return model

    def evaluate(self, model, examples):
        """
        Evaluate the model.
        Parameters
        ----------
        model: BertModel
            The model to be evaluated.
        examples: list
            Evaluation data as a list of InputExample's/
        Returns
        -------
        evaluation_df: pd.DataFrame
            A dataframe that includes for each example predicted probability and labels.
        """

        eval_loader = self.get_loader(examples, phase='eval')

        logger.info("***** Running evaluation ***** ")
        logger.info("  Num examples = %d", len(examples))
        logger.info("  Batch size = %d", self.config.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        predictions = []
        labels = []
        agree_levels = []
        text_ids = []

        for input_ids, attention_mask, token_type_ids, label_ids, agree_ids in tqdm(eval_loader, desc="Testing"):
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            label_ids = label_ids.to(self.device)
            agree_ids = agree_ids.to(self.device)

            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)[0]

                if self.config.output_mode == "classification":
                    loss_fct = CrossEntropyLoss()
                    tmp_eval_loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
                elif self.config.output_mode == "regression":
                    loss_fct = MSELoss()
                    tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

                np_logits = logits.cpu().numpy()

                if self.config.output_mode == 'classification':
                    prediction = np.array(np_logits)
                elif self.config.output_mode == "regression":
                    prediction = np.array(np_logits)

                for agree_id in agree_ids:
                    agree_levels.append(agree_id.item())

                for label_id in label_ids:
                    labels.append(label_id.item())

                for pred in prediction:
                    predictions.append(pred)

                text_ids.append(input_ids)

                # tmp_eval_loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
                # tmp_eval_loss = model(input_ids, token_type_ids, attention_mask, label_ids)

                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1

            # logits = logits.detach().cpu().numpy()
            # label_ids = label_ids.to('cpu').numpy()
            # tmp_eval_accuracy = accuracy(logits, label_ids)

            # eval_loss += tmp_eval_loss.mean().item()
            # eval_accuracy += tmp_eval_accuracy

        evaluation_df = pd.DataFrame({'predictions': predictions, 'labels': labels, "agree_levels": agree_levels})

        return evaluation_df


