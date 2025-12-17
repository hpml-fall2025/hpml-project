# Pipelines Folder

The **finBERT** folder contains all the code for training and inference of the sentimentent analysis portion of our project.
Specifcally, the folder contains code for setting up, training, and predicting with the FinBERT model which takes one news 
headline and outputs a sentiment score.

**news.py** contains the NewsPipeline class which we use to predict volatility using FinBERT. We feed FinBERT multiple news headlines, 
get sentiment scores for each headline, and convert this into a volatility estimate. 
