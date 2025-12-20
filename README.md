# HPML Project: Optimizing Inference Pipelines for Real-Time Financial Trading Systems

## Team Information
- **Team Name**: Optimizing Inference Pipelines for Real-Time Financial Trading Systems
- **Members**:
  - Shobini Iyer (si2449)
  - Mihir Joshi (mnj2122)
  - Shriya Mahakala (srm2245)
  - Ashley Seo (ajs2459)
  - Taimur Shaikh (tfs2123)

---

## 1. Problem Statement
Financial markets are driven by both quantitative metrics and qualitative sentiment, but traditional volatility models (like HAR-RV) often overlook the immediate impact of news, while Large Language Models (LLMs) capable of sentiment analysis are too computationally intensive for real-time trading applications. The primary challenge is the computational cost associated with fine-tuning and deploying transformer models like FinBERT in a latency-sensitive financial environment. This project aims to build a high-performance, end-to-end pipeline that fuses FinBERT-derived sentiment with an adjusted HAR-RV volatility module to produce timely, robust forecasts under real-time constraints.

---

## 2. Model Description
We implemented a unified service integrating a quantitative volatility forecast with qualitative sentiment signals.

**Sentiment Analysis (FinBERT):**
- **Framework:** PyTorch.
- **Architecture:** We utilize knowledge distillation, training a smaller student backbone (e.g., `prajjwall/bert-mini` or `medium`) against the original FinBERT teacher.
- **Optimizations:** The pipeline incorporates Automatic Mixed Precision (AMP), FP16 weight compression, and Low-Rank Adaptation (LoRA) to reduce memory footprint and increase throughput.

**Volatility Forecasting (HAR-RV):**
- **Architecture:** An adjusted Heterogeneous Autoregressive model of Realized Volatility (HAR-RV) tailored for hourly prediction horizons rather than daily.
- **Integration:** The system computes a weighted sum of the HAR-RV forecast and normalized sentiment scores.

---

## 3. Final Results Summary

The following results compare our baseline FinBERT (FP32) against our fully optimized integrated pipeline (Student + AMP + FP16 weights + LoRA) on NVIDIA T4 GPUs.

| Metric               | Value (Optimized) | Value (Baseline) |
|----------------------|-------------------|------------------|
| Test Accuracy        | 77.22%| 72.37%|
| Macro-F1             | 0.7555| 0.6852|
| Inference Throughput | 872.4 samples/s | 154.9 samples/s |
| Model Size           | 127.7 MB | 417.7 MB |
| HAR-RV RMSE (Hourly) | 0.0089 | N/A |
| Device               | NVIDIA T4 (Google Cloud) | NVIDIA T4 |

**Key Achievement:** We achieved a 5.63x inference throughput speedup and a 3.27x reduction in model size while improving accuracy.

---

## 4. Reproducibility Instructions

### A. Requirements

Install dependencies and ensure you have a venv set up:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

also unzip the SPY1min_clean.csv.zip file to get access to the 1 min intraday data (as it is too large to be committed to the repository)


To see more about how to set up the FinBERT pipeline if you'd like to run locally, read our fork of the FinBERT repository here: https://github.com/taimurshaikh/finBERT/tree/master


---

### B. Wandb Dashboard

View training and evaluation metrics here: https://wandb.ai/si2449-columbia-university/Project-Runs/overview

---

### C. Specify for Training or For Inference or if Both

To train the optimized student model using distillation and LoRA (note: this will take a while):

```bash
python train.py --config configs/distillation_lora.yaml
```

To run the HAR-RV volatility dashboard:

```bash
streamlit run dashboard/app.py
```

---

### D. Evaluation

To evaluate the trained model's throughput and accuracy:

```bash
python eval.py --weights checkpoints/best_student_model.pth --quantization fp16

```

---

### E. Quickstart: Minimum Reproducible Result

To reproduce our minimum reported result (77.22% accuracy, ~872 samples/s throughput), run:

```bash
# Step 1: Set up environment
pip install -r requirements.txt

# Step 2: Download dataset (Financial PhraseBank / SPY 1-min data)
bash scripts/download_dataset.sh

# Step 3: Run training (Distilled Student with AMP)
python train.py --config configs/optimized_finbert.yaml

# Step 4: Evaluate with FP16 weights enabled
python eval.py --weights checkpoints/best_model.pth --fp16_weights

```

---

## 5. Notes

* All scripts for profiling, training, and integration are located in `scripts/`, `train.py`, `eval.py`, and `configs/`.
* Trained Models are saved in `models/`.
* **Contact**: si2449@columbia.edu, mnj2122@columbia.edu, srm2245@columbia.edu, ajs2459@columbia.edu, tfs2123@columbia.edu.
* See our Medium article for a summary of results: "Speeding up sentiment: Building a real-time financial trading pipeline with FinBERT".
```