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

### C. Specify Inference

Because of the way our project is structured, section C of our readme is the same as the quickstart:

Follow the instructions in pipelines/finbert to download the FinBERT model and place it in pipelines/finbert/models/sentiment/pytorch_model.bin.
Run predict.py to predict volatility for a time.

To run the HAR-RV volatility dashboard:

```bash
streamlit run dashboard/app.py
```

---

### D. Evaluation

Evaluation can be found through the following file structures:

dashboard contains all the code/instructions for instantiating the dashboard. This dashboard is used to visualize the FinBERT predicitions, HAR-RV predictions, and the weighted final predictions for volatility.

pipelines contains the code/instructions for the FinBERT and HAR-RV pipelines.

data contains a data management script and a sample csv we test with.

HAR-RV_forecast.ipynb contains exploratory code for visualizing the stock data and experimenting with the HAR-RV models.

---

### E. Quickstart: Minimum Reproducible Result

Follow the instructions in pipelines/finbert to download the FinBERT model and place it in pipelines/finbert/models/sentiment/pytorch_model.bin.
Run predict.py to predict volatility for a time.

To run the HAR-RV volatility dashboard:

```bash
streamlit run dashboard/app.py
```

---

## 5. Notes

* All scripts for profiling, training, and integration are located in `scripts/`, `train.py`, `eval.py`, and `configs/`.
* Trained Models are saved in `pipelines/finbert/models/`.
* **Contact**: si2449@columbia.edu, mnj2122@columbia.edu, srm2245@columbia.edu, ajs2459@columbia.edu, tfs2123@columbia.edu.
* See our Medium article for a summary of results: "Speeding up sentiment: Building a real-time financial trading pipeline with FinBERT".
```