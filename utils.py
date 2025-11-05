from datetime import datetime, timedelta
from itertools import combinations
from typing import List, Dict, Any

def get_date_range(dates) -> List[datetime.date]:
	"""
	Returns a list of date strings (YYYY-MM-DD) from start_date to end_date inclusive.
	"""
	start = datetime.strptime(dates[0], "%Y-%m-%d").date()
	end = datetime.strptime(dates[1], "%Y-%m-%d").date()

	days = []
	current = start
	while current <= end:
		days.append(current)
		current += timedelta(days=1)
	return days


def generate_model_configs(feature_tokens: List[str], alpha_list: List[float]) -> List[Dict[str, Any]]:
	model_configs: List[Dict[str, Any]] = []

	# Generate all non-empty feature subsets
	all_feature_subsets: List[List[str]] = []
	for r in range(1, len(feature_tokens) + 1):
		for combo in combinations(feature_tokens, r):
			subset = list(combo)

			# Skip subsets that contain both 78b and 1D
			if "78b" in subset and "1D" in subset:
				continue

			# Skip subsets that contain both 390b and 1W
			if "390b" in subset and "1W" in subset:
				continue

			all_feature_subsets.append(subset)

	# cross each subset with each alpha
	for feats in all_feature_subsets:
		for alpha in alpha_list:
			cfg = {
				"features": feats,
				"reg_type": "ridge",
				"ridge_alpha": alpha,
			}
			model_configs.append(cfg)

	return model_configs