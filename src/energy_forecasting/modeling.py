"""Time-ordered model training, evaluation, and recursive forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

from .features import FEATURE_COLUMNS, make_features, training_features


@dataclass
class ForecastResult:
    model: HistGradientBoostingRegressor
    metrics: dict[str, float]
    evaluation: pd.DataFrame
    forecast: pd.DataFrame
    split_timestamp: pd.Timestamp


def _recursive_predict(
    model: HistGradientBoostingRegressor,
    history: pd.Series,
    timestamps: pd.DatetimeIndex,
) -> np.ndarray:
    observed = list(history.astype(float).to_numpy())
    predictions = []
    for timestamp in timestamps:
        features = make_features(pd.Series(observed), pd.DatetimeIndex([timestamp]))
        prediction = max(0.0, float(model.predict(features[FEATURE_COLUMNS])[0]))
        predictions.append(prediction)
        observed.append(prediction)
    return np.asarray(predictions)


def train_evaluate_forecast(
    data: pd.DataFrame,
    horizon_hours: int = 168,
    test_fraction: float = 0.2,
) -> ForecastResult:
    """Fit a model, evaluate with walk-forward predictions, then forecast ahead."""
    if not 0.05 <= test_fraction <= 0.4:
        raise ValueError("test_fraction must be between 0.05 and 0.40.")
    if horizon_hours < 1:
        raise ValueError("horizon_hours must be at least 1.")

    features, target = training_features(data)
    split_index = int(len(features) * (1 - test_fraction))
    if split_index < 168 or len(features) - split_index < 24:
        raise ValueError("Dataset is too short for the requested train/test split.")

    model = HistGradientBoostingRegressor(
        max_iter=180,
        learning_rate=0.08,
        max_leaf_nodes=20,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(features.iloc[:split_index], target.iloc[:split_index])

    split_timestamp = pd.Timestamp(data["timestamp"].iloc[split_index + 167])
    test_timestamps = pd.DatetimeIndex(data["timestamp"].iloc[split_index + 168 :])
    test_actual = data["consumption_kwh"].iloc[split_index + 168 :].to_numpy(
        dtype=float
    )
    evaluation_history = data["consumption_kwh"].iloc[: split_index + 168]
    test_predictions = _recursive_predict(model, evaluation_history, test_timestamps)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(test_actual, test_predictions))),
        "r2": float(r2_score(test_actual, test_predictions)),
        "mae": float(np.mean(np.abs(test_actual - test_predictions))),
    }
    evaluation = pd.DataFrame(
        {
            "timestamp": test_timestamps,
            "actual_kwh": test_actual,
            "predicted_kwh": test_predictions,
        }
    )

    model.fit(features, target)
    future_timestamps = pd.date_range(
        start=pd.Timestamp(data["timestamp"].iloc[-1]) + pd.Timedelta(1, unit="h"),
        periods=horizon_hours,
        freq="h",
    )
    future_values = _recursive_predict(
        model, data["consumption_kwh"], future_timestamps
    )
    forecast = pd.DataFrame(
        {"timestamp": future_timestamps, "predicted_kwh": future_values}
    )
    return ForecastResult(
        model=model,
        metrics=metrics,
        evaluation=evaluation,
        forecast=forecast,
        split_timestamp=split_timestamp,
    )
