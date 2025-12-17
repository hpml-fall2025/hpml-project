import pipelines.weighting 
import datetime 


query_date = input("Enter the date (YYYY-MM-DD) to query volatility prediction: ")


try:
    date_obj = datetime.datetime.strptime(query_date, "%Y-%m-%d").date()
    date_obj = date_obj.date()
except ValueError:
    print("Invalid date format.")
    exit(1)

print("Predicted Volatility for", date_obj, "is:", pipelines.weighting.predict_weighted_vol(date_obj))
