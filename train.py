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


VALID_TOKENS = {"1b", "12b", "78b", "390b", "1D", "1W"}


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns log(P_t / P_{t-1}) with same index as prices."""
    return np.log(prices / prices.shift(1))



def _precompute_feature_series(
    index: pd.DatetimeIndex,
    r2: pd.Series,
) -> Dict[str, pd.Series]:
    """
    Precompute all feature series:

        '1b'   : mean of last 1 bar (essentially r2 itself)
        '12b'  : mean of last 12 bars
        '78b'  : mean of last 78 bars
        '390b' : mean of last 390 bars
        '1D'   : previous day's realized variance (sum of r2 over that day)
        '1W'   : mean RV over previous 5 days

    Returns:
        dict: token -> Series aligned with `index`.
    """
    f_1b = r2.rolling(1, min_periods=1).mean()
    f_12b = r2.rolling(12, min_periods=12).mean()
    f_78b = r2.rolling(78, min_periods=78).mean()
    f_390b = r2.rolling(390, min_periods=390).mean()

    # Daily realized variance: sum of r2 for each calendar date
    # Treat each unique index.date as one "session"
    dates = pd.Series(index.date, index=index)
    daily_rv = r2.groupby(dates).sum()

    unique_days = np.array(daily_rv.index)
    day_to_1D: Dict = {}
    day_to_1W: Dict = {}

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

    # Align 1D/1W back to bar-level index
    day_array = dates.values
    f_1D = pd.Series([day_to_1D[d] for d in day_array], index=index)
    f_1W = pd.Series([day_to_1W[d] for d in day_array], index=index)

    return {
        "1b": f_1b,
        "12b": f_12b,
        "78b": f_78b,
        "390b": f_390b,
        "1D": f_1D,
        "1W": f_1W,
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

    For each bar i where:
        - index[i].date() is in training `dates`
        - all requested features at i are finite
        - next-bar r2[i+1] is finite

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
