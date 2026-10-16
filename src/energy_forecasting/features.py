"""Time and historical-consumption feature engineering."""

from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "day_of_year",
    "month",
    "is_weekend",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
]


def make_features(history: pd.Series, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Build one row per timestamp using only consumption available beforehand."""
    values = history.to_numpy(dtype=float)
    if len(values) < 168:
        raise ValueError("At least 168 historical hourly readings are required.")

    rows = []
    for timestamp in timestamps:
        rows.append(
            {
                "hour": timestamp.hour,
                "day_of_week": timestamp.dayofweek,
                "day_of_year": timestamp.dayofyear,
                "month": timestamp.month,
                "is_weekend": int(timestamp.dayofweek >= 5),
                "lag_1": values[-1],
                "lag_24": values[-24],
                "lag_168": values[-168],
                "rolling_mean_24": float(np_mean(values[-24:])),
            }
        )
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS, index=timestamps)


def training_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create supervised features, discarding rows without a full lag history."""
    target = data["consumption_kwh"].astype(float)
    timestamps = pd.DatetimeIndex(data["timestamp"])
    features = pd.DataFrame(index=data.index)
    features["hour"] = timestamps.hour
    features["day_of_week"] = timestamps.dayofweek
    features["day_of_year"] = timestamps.dayofyear
    features["month"] = timestamps.month
    features["is_weekend"] = (timestamps.dayofweek >= 5).astype(int)
    features["lag_1"] = target.shift(1)
    features["lag_24"] = target.shift(24)
    features["lag_168"] = target.shift(168)
    features["rolling_mean_24"] = target.shift(1).rolling(window=24).mean()
    valid = features.notna().all(axis=1)
    return features.loc[valid, FEATURE_COLUMNS], target.loc[valid]


def np_mean(values) -> float:
    """Return the mean for the small rolling window without another dependency."""
    return float(sum(values) / len(values))
