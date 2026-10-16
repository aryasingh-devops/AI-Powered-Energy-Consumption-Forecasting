from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energy_forecasting.data import prepare_energy_data, simulate_energy_data
from energy_forecasting.features import make_features, training_features
from energy_forecasting.modeling import train_evaluate_forecast


class EnergyForecastingTests(unittest.TestCase):
    def test_simulation_is_reproducible_and_hourly(self):
        first = simulate_energy_data(periods=24 * 21)
        second = simulate_energy_data(periods=24 * 21)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(
            first["timestamp"].diff().dropna().unique().tolist(),
            [pd.Timedelta(1, unit="h")],
        )

    def test_cleaning_sorts_removes_duplicate_and_invalid_rows(self):
        data = simulate_energy_data(periods=24 * 14)
        mixed = pd.concat(
            [
                data,
                data.iloc[[4]].assign(consumption_kwh=-1),
                data.iloc[[4]],
                data.iloc[[3]].assign(consumption_kwh=999),
                data.iloc[[0]],
                pd.DataFrame(
                    {"timestamp": ["not-a-date"], "consumption_kwh": [12]}
                ),
            ],
            ignore_index=True,
        )
        cleaned = prepare_energy_data(mixed)
        self.assertTrue(cleaned["timestamp"].is_monotonic_increasing)
        self.assertFalse(cleaned["timestamp"].duplicated().any())
        self.assertTrue((cleaned["consumption_kwh"] >= 0).all())
        self.assertEqual(len(cleaned), len(data))
        self.assertEqual(
            cleaned.loc[cleaned["timestamp"] == data["timestamp"].iloc[3],
                       "consumption_kwh"].iloc[0],
            999,
        )

    def test_features_use_prior_values_and_expected_columns(self):
        data = simulate_energy_data(periods=24 * 21)
        features, target = training_features(data)
        self.assertEqual(len(features), len(target))
        self.assertEqual(features.columns[-1], "rolling_mean_24")
        generated = make_features(
            data["consumption_kwh"], pd.DatetimeIndex([data["timestamp"].iloc[-1]])
        )
        self.assertAlmostEqual(
            generated["lag_1"].iloc[0], data["consumption_kwh"].iloc[-1]
        )

    def test_training_evaluation_and_forecast_shapes(self):
        data = prepare_energy_data(simulate_energy_data(periods=24 * 90))
        result = train_evaluate_forecast(data, horizon_hours=48)
        self.assertEqual(len(result.forecast), 48)
        self.assertEqual(
            result.forecast["timestamp"].iloc[0],
            data["timestamp"].iloc[-1] + pd.Timedelta(1, unit="h"),
        )
        self.assertTrue(set(result.metrics) == {"rmse", "r2", "mae"})
        self.assertGreaterEqual(result.forecast["predicted_kwh"].min(), 0)

    def test_upload_schema_and_continuity_are_validated(self):
        with self.assertRaisesRegex(ValueError, "must contain columns"):
            prepare_energy_data(pd.DataFrame({"time": [1], "usage": [2]}))
        data = simulate_energy_data(periods=24 * 14)
        data.loc[200, "timestamp"] += pd.Timedelta(2, unit="h")
        with self.assertRaisesRegex(ValueError, "hourly and continuous"):
            prepare_energy_data(data)


if __name__ == "__main__":
    unittest.main()
