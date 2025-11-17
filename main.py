"""
Main execution script for 5-minute Realized Variance Prediction System.
"""

import sys
from config import *
import pull
import utils
import train
import evaluate
from itertools import combinations


def main() -> None:
	try:		
		print(f"Fetching historical data for {TICKER}...")
		training_data = pull.get_historical_data(TICKER, TRAINING_INTERVAL, TIME_INTERVAL)
		validation_data = pull.get_historical_data(TICKER, VALIDATION_INTERVAL, TIME_INTERVAL)
		test_data = pull.get_historical_data(TICKER, TEST_INTERVAL, TIME_INTERVAL)
		
		test_dates = utils.get_date_range(TEST_INTERVAL)
		validation_dates = utils.get_date_range(VALIDATION_INTERVAL)
		training_dates = utils.get_date_range(TRAINING_INTERVAL)
		print(f"Fetch complete\n")

		# Specific Model Analysis

		# print(f"\nConfiguration:")
		# print(f"\tTicker:                       {TICKER}")
		# print(f"\tHAR_RV Model Features:        {MODEL_CONFIGS['features']}")
		# print(f"\tRegression Type:              {MODEL_CONFIGS['reg_type']}")
		# print(f"\tRidge Alpha (if applicable):  {MODEL_CONFIGS.get('ridge_alpha', '0.0')}")
		# print()
		# if MODEL_CONFIGS.get('ridge_alpha') is None:
		# 	MODEL_CONFIGS['ridge_alpha'] = 0.0

		# print(f"Training HAR-RV model using {training_dates[0]} through {training_dates[-1]}")
		
		# intercept, coeffs = train.train(MODEL_CONFIGS, training_data, training_dates)
		# print(f"Intercept: {intercept}")
		# print(f"Coefficients: {coeffs}")

		# print(f"Testing on Validation Interval: {VALIDATION_INTERVAL[0]} through {VALIDATION_INTERVAL[1]}")
		# predictions, metrics = evaluate.evaluate(MODEL_CONFIGS, intercept, coeffs, validation_data, validation_dates)
		# print(f"MSE:   {metrics['mse']:.6f}")
		# print(f"RMSE:  {metrics['rmse']:.6f}")
		# print(f"MAE:   {metrics['mae']:.6f}")
		# print(f"SMAPE: {metrics['smape']:.6f}")
		# print(f"R2:    {metrics['r2']:.6f}")
		# print(f"\nPredictions:")
		# print(predictions)

		
		
		# Model Validation Loop
		results = []

		all_configs = utils.generate_model_configs(FEATURE_CONFIGS, ALPHA_CONFIGS)

		for cfg in all_configs:
			print(f"\nConfiguration:")
			print(f"\tHAR_RV Model Features:        {cfg['features']}")
			print(f"\tRegression Type:              {cfg['reg_type']}")
			print(f"\tRidge Alpha (if applicable):  {cfg.get('ridge_alpha', '0.0')}")
			print()
			if cfg.get('ridge_alpha') is None:
				cfg['ridge_alpha'] = 0.0

			intercept, coeffs = train.train(cfg, training_data, training_dates)

			predictions, metrics = evaluate.evaluate(cfg, intercept, coeffs, validation_data, validation_dates)
			print(f"MSE:   {metrics['mse']:.6f}")
			print(f"RMSE:  {metrics['rmse']:.6f}")
			print(f"MAE:   {metrics['mae']:.6f}")
			print(f"SMAPE: {metrics['smape']:.6f}")
			print(f"MAPE: {metrics['mape']:.6f}")
			print(f"MPE: {metrics['mpe']:.6f}")
			print(f"R2:    {metrics['r2']:.6f}")

			results.append({
				"config": cfg,
				"metrics": metrics,
			})

		if results:
			print("\n====================")
			print("Best models by metric")
			print("====================")

			# lower is better
			for metric_name in ["mse", "rmse", "mae", "smape", "mape", "mpe"]:
				best = min(results, key=lambda r: r["metrics"][metric_name])
				cfg = best["config"]
				val = best["metrics"][metric_name]
				print(
					f"{metric_name.upper():5s} -> "
					f"features={cfg['features']}, "
					f"reg_type={cfg['reg_type']}, "
					f"ridge_alpha={cfg['ridge_alpha']}, "
					f"value={val:.6f}"
				)

			#  higher is better
			for metric_name in ["r2"]:
				best = max(results, key=lambda r: r["metrics"][metric_name])
				cfg = best["config"]
				val = best["metrics"][metric_name]
				print(
					f"{metric_name.upper():5s} -> "
					f"features={cfg['features']}, "
					f"reg_type={cfg['reg_type']}, "
					f"ridge_alpha={cfg['ridge_alpha']}, "
					f"value={val:.6f}"
				)

		return

	except ValueError as e:
		print(f"\nValidation Error: {str(e)}", file=sys.stderr)
		sys.exit(1)
		
	except Exception as e:
		print(f"\nUnexpected Error: {str(e)}", file=sys.stderr)
		sys.exit(1)


if __name__ == "__main__":
	main()
