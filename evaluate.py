# evaluate.py

from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from train import calculate_log_returns, _precompute_feature_series  # reuse helpers


def evaluate(
	model: Dict,
	intercept: float,
	coefs: List[float],
	data: pd.DataFrame,
	dates: List[datetime.date],
) -> Tuple[pd.DataFrame, Dict[str, float]]:

	features = model["features"]
	if len(features) != len(coefs):
		raise ValueError(
			f"Length of coefs ({len(coefs)}) does not match number of features ({len(features)})"
		)

	# Squared log-returns
	r = calculate_log_returns(data["Close"])
	r2 = r ** 2

	# Precompute all feature series (1b, 12b, 78b, 390b, 1D, 1W)
	feature_series = _precompute_feature_series(data.index, r2)

	# Build evaluation design matrix and labels (and timestamps)
	X, y, ts = _build_eval_matrix(
		index=data.index,
		r2=r2,
		feature_series=feature_series,
		tokens=features,
		dates=dates,
	)

	if X.size == 0:
		raise ValueError("No evaluation samples could be created for the given dates.")

	# Predictions: y_hat = intercept + X @ coefs
	coefs_vec = np.asarray(coefs, dtype=float)
	y_hat = intercept + X @ coefs_vec

	# Build predictions DataFrame
	pred_df = pd.DataFrame(
		{
			"timestamp": ts,
			"actual_rv": y,
			"predicted_rv": y_hat,
		}
	)

	# Percent error
	with np.errstate(divide="ignore", invalid="ignore"):
		percent_err = (
			(pred_df["predicted_rv"] - pred_df["actual_rv"])
			/ pred_df["actual_rv"]
			* 100.0
		)
		percent_err = percent_err.replace([np.inf, -np.inf], np.nan)

	pred_df["percent_error"] = percent_err

	residuals = y_hat - y
	mse = float(np.mean(residuals ** 2))
	rmse = float(np.sqrt(mse))
	mae = float(np.mean(np.abs(residuals)))


	# Symmetric MAPE
	with np.errstate(divide="ignore", invalid="ignore"):
		smape = np.nanmean(200 * np.abs(y_hat - y) / (np.abs(y_hat) + np.abs(y)))

	# R^2
	ss_res = np.sum(residuals ** 2)
	ss_tot = np.sum((y - np.mean(y)) ** 2)
	r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

	metrics = {
		"mse": mse,
		"rmse": rmse,
		"mae": mae,
		"smape": smape,
		"r2": r2,
	}

	return pred_df, metrics



def _build_eval_matrix(
	index: pd.DatetimeIndex,
	r2: pd.Series,
	feature_series: Dict[str, pd.Series],
	tokens: List[str],
	dates: List[datetime.date],
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
	"""
	Construct X, y, timestamps for evaluation.

	For each bar i where:
		- index[i].date() is in the inclusive range [min(dates), max(dates)]
		- all requested features at i are finite
		- next-bar r2[i+1] is finite

	we create one sample:
		X_i = [feature_series[token].iloc[i] for token in tokens]
		y_i = r2.iloc[i+1]
		ts_i = index[i+1]  (label timestamp)
	"""

	X_rows: List[List[float]] = []
	y_rows: List[float] = []
	ts_rows: List[pd.Timestamp] = []

	n = len(index)
	for i in range(n - 1):  # need i+1 for label
		if index[i].date() not in dates:
			continue

		y_val = r2.iloc[i + 1]
		if not np.isfinite(y_val):
			continue

		feats: List[float] = []
		good = True
		for t in tokens:
			val = feature_series[t].iloc[i]
			if not np.isfinite(val):
				good = False
				break
			feats.append(float(val))
		if not good:
			continue

		X_rows.append(feats)
		y_rows.append(float(y_val))
		ts_rows.append(index[i + 1])

	X = np.asarray(X_rows, dtype=float)
	y = np.asarray(y_rows, dtype=float)
	return X, y, ts_rows
