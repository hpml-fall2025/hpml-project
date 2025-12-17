## Installation
Set up a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## End to End Pipeline

<img width="1546" height="444" alt="image" src="https://github.com/user-attachments/assets/0ffb2da1-f51c-4b15-85ef-3ad6c5c066e0" />

## QuickStart

1. Follow the instructions in pipelines/finbert to download the FinBERT model and place it in pipelines/finbert/models/sentiment/pytorch_model.bin.
2. Run predict.py to predict volatility for a time. 

## Folder/file descriptions
**dashboard** contains all the code/instructions for instantiating the dashboard. This dashboard is used to visualize the FinBERT predicitions, HAR-RV predictions, and the weighted final predictions for volatility. 

**pipelines** contains the code/instructions for the FinBERT and HAR-RV pipelines. 

**data** contains a data management script and a sample csv we test with.

**HAR-RV_forecast.ipynb** contains exploratory code for visualizing the stock data and experimenting with the HAR-RV models. 
