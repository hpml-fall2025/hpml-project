# Pipelines Folder

The **finBERT** folder contains all the code for training and inference of the sentimentent analysis portion of our project.
Specifcally, the folder contains code for setting up, training, and predicting with the FinBERT model which takes one news 
headline and outputs a sentiment score.

**news.py** contains the NewsPipeline class which we use to predict volatility using FinBERT. We feed FinBERT multiple news headlines, 
get sentiment scores for each headline, and convert this into a volatility estimate. 

**volatility.py** trains/sets up the HAR-RV model. Contains the get_latest_data function which calculates the volatility prediction for the current time. 

**base.py** sets template for news.py and volatility.py.

The dashboard calls get_latest_data from News.py and Volatility.py to stream predictions.
