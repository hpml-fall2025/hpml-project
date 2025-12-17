# FinBERT Pipeline: Quick Start Guide

## 1. Installation
Set up a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Downloading Pretrained Models
- Download the [FinBERT Sentiment Model](https://prosus-public.s3-eu-west-1.amazonaws.com/finbert/finbert-sentiment/pytorch_model.bin).
- Place it in a suitable directory, e.g., `models/sentiment/pytorch_model.bin`.
- Copy a compatible `config.json` into the same directory.
- Edit `config.json` to specify:
    ```json
    {
      "model_type": "bert",
      "num_labels": 3  // optional but recommended
    }
    ```
- Load the model in code with `.from_pretrained(<model_directory>)`.

## 3. NLTK Setup
Install and setup the NLTK tokenizer:
```bash
pip install nltk
python -m nltk.downloader punkt
```

## 4. Data Preparation
- Download the [Financial PhraseBank dataset (Malo et al., 2014)](https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10/data/0c96051eee4fb1d56e000000/FinancialPhraseBank-v10.zip?origin=publication_list).
- Extract and note the path to `Sentences_50Agree.txt`.
- Create `data/sentiment_data/` and populate it with `train.csv`, `validation.csv`, and `test.csv` (splits).
- Use our dataset script for conversion:
    ```bash
    python scripts/datasets.py --data_path <path_to_Sentences_50Agree.txt>
    ```

## 5. Training
- Main training is handled in `finbert_training.ipynb` (other variants in `notebooks/`).
- Models are saved to `models/`.
- Example config:
    ```python
    config = Config(
        data_dir=cl_data_path,
        bert_model=bertmodel,
        num_train_epochs=4.0,
        model_dir=cl_path,
        max_seq_length=64,
        train_batch_size=32,
        learning_rate=2e-5,
        output_mode='classification',
        warm_up_proportion=0.2,
        local_rank=-1,
        discriminate=True,           # mitigates catastrophic forgetting
        gradual_unfreeze=True        # enables gradual layer unfreezing
    )
    ```

## 6. Profiling
Profile training performance with `baseline_profiling.ipynb` (adds PyTorch profiling to the baseline notebook).

## 7. Available Model Variants
Notebooks in `notebooks/` explain and implement each of these:
- **Baseline:** Standard FinBERT.
- **Distilled BERT** (Base, Medium, Mini): Smaller, faster variants using knowledge distillation.
- **LoRA** (@shobini): Low-Rank Adaptation for parameter-efficient training.
- **AMP:** Automatic Mixed Precision for faster training and lower memory usage.
- **FP16 Weights:** Store weights as FP16 for reduced memory/compute.
- **Master Train:** Combines Distillation, AMP, and FP16 weights for optimal tradeoffs.

### Details on Model Variants

#### Distilled Models
- DistilledBERT variants are trained using the baseline FinBERT as teacher.
- Relevant notebooks: `distilled_finbert.ipynb`, `medium_finbert.ipynb`, `mini_finbert.ipynb`.

#### LoRA (@shobini)
- Implements efficient fine-tuning via Low-Rank Adaptation.

#### AMP
- Trains using PyTorch native mixed-precision (`torch.cuda.amp`), enabling FP16 ops for speed/memory with FP32-equivalent accuracy.

#### FP16 Weights
- Runs training in FP32, then converts weights to FP16 for inference and deployment efficiency.

#### Master Train
- Unified workflow for best accuracy-throughput-memory efficiency by leveraging distillation, AMP, and FP16.

---

For further details, see the individual notebook files and code comments.
