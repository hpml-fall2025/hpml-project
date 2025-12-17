import pipelines.weighting 
import datetime 


query_date = input("Enter the time (YYYY-MM-DD HH) to query volatility prediction: ")


try:
    time_obj = datetime.datetime.strptime(query_date, "%Y-%m-%d %H")
except ValueError:
    print("Invalid date format.")
    exit(1)

print("Predicted Volatility for", date_obj, "is:", pipelines.weighting.predict_weighted_vol(time_obj))
