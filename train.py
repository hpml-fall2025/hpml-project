# train.py

from typing import List, Dict, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge


def train(model: Dict, data: pd.DataFrame, dates: List[datetime.date]) -> List[float]:
    features = model["features"]
    reg_type = model["reg_type"]

    ridge_alpha = model.get("ridge_alpha", 0.0)

    r = calculate_log_returns(data["Close"])
    r2 = r ** 2

    feature_series = _precompute_feature_series(data.index, r2)

    X, y = _build_training_matrix(
        index=data.index,
        r2=r2,
        feature_series=feature_series,
        tokens=features,
        dates=dates,
    )

    # Fit regression
    if reg_type == "ols":
        reg = LinearRegression()
    elif reg_type == "ridge":
        reg = Ridge(alpha=ridge_alpha)
    else:
        raise ValueError(f"Unknown reg_type '{reg_type}'. Expected 'ols' or 'ridge'.")

    reg.fit(X, y)

    return reg.intercept_, list(map(float, reg.coef_))


# Now that we're on daily data, tokens are:
#   1b  = current day's RV (rolling 1)
#   1D  = previous day's RV
#   1W  = mean RV over previous 5 days
#   1M  = mean RV over previous 22 days
VALID_TOKENS = {"1b", "1D", "1W", "1M"}


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns log(P_t / P_{t-1}) with same index as prices."""
    return np.log(prices / prices.shift(1))


def _precompute_feature_series(
    index: pd.DatetimeIndex,
    r2: pd.Series,
) -> Dict[str, pd.Series]:
    """
    Precompute all feature series for (now) daily data:

        '1b' : RV at current index (rolling 1-day mean of r2)
        '1D' : previous day's realized variance
        '1W' : mean RV over previous 5 days
        '1M' : mean RV over previous 22 days

    Returns:
        dict: token -> Series aligned with `index`.
    """
    # 1b: rolling 1 (effectively just r2 itself, but keeps interface consistent)
    f_1b = r2.rolling(1, min_periods=1).mean()

    # Daily realized variance per date (here each index is already one day,
    # but grouping keeps it robust if there are oddities)
    dates = pd.Series(index.date, index=index)
    daily_rv = r2.groupby(dates).sum()

    unique_days = np.array(daily_rv.index)
    day_to_1D: Dict = {}
    day_to_1W: Dict = {}
    day_to_1M: Dict = {}

    for i, d in enumerate(unique_days):
        # 1D: previous day's RV
        if i >= 1:
            day_to_1D[d] = float(daily_rv[unique_days[i - 1]])
        else:
            day_to_1D[d] = np.nan

        # 1W: mean of previous 5 days' RV
        if i >= 5:
            prev_days = unique_days[i - 5 : i]
            vals = [daily_rv[pd_] for pd_ in prev_days]
            day_to_1W[d] = float(np.mean(vals))
        else:
            day_to_1W[d] = np.nan

        # 1M: mean of previous 22 days' RV
        if i >= 22:
            prev_days_m = unique_days[i - 22 : i]
            vals_m = [daily_rv[pd_] for pd_ in prev_days_m]
            day_to_1M[d] = float(np.mean(vals_m))
        else:
            day_to_1M[d] = np.nan

    # Align 1D/1W/1M back to bar-level index
    day_array = dates.values
    f_1D = pd.Series([day_to_1D[d] for d in day_array], index=index)
    f_1W = pd.Series([day_to_1W[d] for d in day_array], index=index)
    f_1M = pd.Series([day_to_1M[d] for d in day_array], index=index)

    return {
        "1b": f_1b,
        "1D": f_1D,
        "1W": f_1W,
        "1M": f_1M,
    }


def _build_training_matrix(
    index: pd.DatetimeIndex,
    r2: pd.Series,
    feature_series: Dict[str, pd.Series],
    tokens: List[str],
    dates: List[datetime.date],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct the design matrix X and target y.

    For each bar/day i where:
        - index[i].date() is in training `dates`
        - all requested features at i are finite
        - next-day r2[i+1] is finite

    we create one sample:
        X_i = [feature_series[token].iloc[i] for token in tokens]
        y_i = r2.iloc[i+1]
    """
    for t in tokens:
        if t not in VALID_TOKENS:
            raise ValueError(
                f"Unsupported feature token '{t}'. Valid: {sorted(VALID_TOKENS)}"
            )

    X_rows: List[List[float]] = []
    y_rows: List[float] = []

    n = len(index)
    for i in range(n - 1):
        day_i = index[i].date()
        if day_i not in dates:
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

    if not X_rows:
        raise ValueError(
            "No training samples could be created for the given dates and model config."
        )

    X = np.asarray(X_rows, dtype=float)
    y = np.asarray(y_rows, dtype=float)
    return X, y