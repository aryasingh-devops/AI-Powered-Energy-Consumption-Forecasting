"""Dataset loading and deterministic virtual energy-consumption simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd


TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "consumption_kwh"


def simulate_energy_data(
    periods: int = 24 * (365 * 2 + 1),
    seed: int = 42,
    start: str = "2023-01-01",
) -> pd.DataFrame:
    """Create realistic-looking hourly building energy data for a demo.

    The generated signal combines daily operating hours, weekly schedules,
    annual seasonality, a gradual trend, and seeded measurement noise.
    """
    if periods < 24 * 14:
        raise ValueError("The simulation needs at least 14 days of hourly data.")

    timestamps = pd.date_range(start=start, periods=periods, freq="h")
    rng = np.random.default_rng(seed)
    hour = timestamps.hour.to_numpy()
    day_of_week = timestamps.dayofweek.to_numpy()
    day_of_year = timestamps.dayofyear.to_numpy()
    weekend = day_of_week >= 5

    occupancy = np.where((hour >= 7) & (hour <= 19), 1.0, 0.38)
    occupancy = occupancy * np.where(weekend, 0.64, 1.0)
    daily_cycle = 8.5 * occupancy
    cooling_load = 4.0 + 3.5 * np.maximum(
        0, np.sin(2 * np.pi * (hour - 8) / 24)
    )
    seasonal_load = 2.5 * np.cos(2 * np.pi * (day_of_year - 200) / 365.25)
    long_term_trend = np.linspace(0, 1.8, periods)
    noise = rng.normal(loc=0, scale=0.9, size=periods)

    consumption = np.maximum(
        2.0, 14.0 + daily_cycle + cooling_load + seasonal_load + long_term_trend + noise
    )
    return pd.DataFrame(
        {TIMESTAMP_COLUMN: timestamps, TARGET_COLUMN: consumption.round(3)}
    )


def prepare_energy_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean timestamped energy observations."""
    required = {TIMESTAMP_COLUMN, TARGET_COLUMN}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            "Dataset must contain columns named 'timestamp' and "
            f"'consumption_kwh'; missing: {', '.join(sorted(missing))}."
        )

    cleaned = data.loc[:, [TIMESTAMP_COLUMN, TARGET_COLUMN]].copy()
    cleaned[TIMESTAMP_COLUMN] = pd.to_datetime(
        cleaned[TIMESTAMP_COLUMN], errors="coerce"
    )
    cleaned[TARGET_COLUMN] = pd.to_numeric(cleaned[TARGET_COLUMN], errors="coerce")
    cleaned = cleaned.dropna(subset=[TIMESTAMP_COLUMN, TARGET_COLUMN])
    cleaned = cleaned.sort_values(TIMESTAMP_COLUMN)
    cleaned = cleaned.drop_duplicates(subset=TIMESTAMP_COLUMN, keep="last")
    cleaned = cleaned.loc[cleaned[TARGET_COLUMN] >= 0].reset_index(drop=True)

    differences = cleaned[TIMESTAMP_COLUMN].diff().dropna()
    hourly_ns = pd.Timedelta(1, unit="h").value
    if not differences.empty and not (
        differences.astype("timedelta64[ns]").astype("int64") == hourly_ns
    ).all():
        raise ValueError(
            "Readings must be hourly and continuous. Resample or fill missing hours "
            "before training."
        )
    if len(cleaned) < 24 * 14:
        raise ValueError("At least 14 days of valid hourly readings are required.")
    return cleaned
