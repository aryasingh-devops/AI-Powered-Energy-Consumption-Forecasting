"""Streamlit dashboard for the AI-Powered Energy Forecasting project."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from energy_forecasting.data import (  # noqa: E402
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    prepare_energy_data,
    simulate_energy_data,
)
from energy_forecasting.modeling import train_evaluate_forecast  # noqa: E402

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

st.set_page_config(
    page_title="Energy Forecasting Studio",
    page_icon="⚡",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #f4f7fb; }
    [data-testid="stSidebar"] { background: #101b2d; }
    [data-testid="stSidebar"] * { color: #eef5ff; }
    .hero { padding: 1.4rem 1.7rem; border-radius: 18px; color: white;
            background: linear-gradient(115deg,#10233f,#176b69); margin-bottom: 1rem; }
    .hero p { color: #d3e9eb; margin-bottom: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>⚡ Energy Forecasting Studio</h1>
      <p>Operational planning from hourly meter data — powered by a reproducible
      machine-learning simulation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Forecast controls")
    source = st.radio("Data source", ["Virtual building simulation", "Upload CSV"])
    horizon_days = st.slider("Forecast horizon (days)", 1, 14, 7)
    st.caption("The model predicts recursively using the latest known readings.")
    uploaded_file = None
    if source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Choose hourly readings", type=["csv"], help=(
                "CSV must contain timestamp and consumption_kwh columns."
            )
        )
    st.divider()
    st.caption("Demo data only — not a utility dispatch or safety-critical system.")

try:
    if source == "Upload CSV":
        if uploaded_file is None:
            st.info(
                "Upload a CSV with `timestamp` and `consumption_kwh` columns "
                "to run the forecasting workflow."
            )
            st.stop()
        raw_data = pd.read_csv(uploaded_file)
        dataset_label = uploaded_file.name
    else:
        raw_data = simulate_energy_data()
        dataset_label = "Virtual building simulation"

    data = prepare_energy_data(raw_data)
    result = train_evaluate_forecast(data, horizon_hours=horizon_days * 24)
except (ValueError, pd.errors.ParserError) as error:
    st.error(f"Unable to prepare this dataset: {error}")
    st.stop()

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

left, middle, right, extra = st.columns(4)
left.metric("Readings", f"{len(data):,}", help="Hourly observations after cleaning")
middle.metric("Test RMSE", f"{result.metrics['rmse']:.2f} kWh")
right.metric("Test R²", f"{result.metrics['r2']:.3f}")
extra.metric(
    "Forecast energy",
    f"{result.forecast['predicted_kwh'].sum():,.0f} kWh",
    help=f"Total predicted energy over {horizon_days} days",
)

st.caption(
    f"Dataset: **{dataset_label}** · "
    f"{data[TIMESTAMP_COLUMN].min():%Y-%m-%d %H:%M} to "
    f"{data[TIMESTAMP_COLUMN].max():%Y-%m-%d %H:%M} · "
    f"Evaluation begins after {result.split_timestamp:%Y-%m-%d %H:%M}"
)

overview, evaluation_tab, forecast_tab, data_tab = st.tabs(
    ["Overview", "Model evaluation", "Forecast", "Data & methodology"]
)

with overview:
    st.subheader("Recent energy profile")
    recent = data.tail(24 * 14)
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(recent[TIMESTAMP_COLUMN], recent[TARGET_COLUMN], color="#17817d")
    axis.set_ylabel("Consumption (kWh per hour)")
    axis.set_xlabel("")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)
    st.markdown(
        "**Business interpretation:** use the forecast to schedule flexible loads, "
        "staffing, battery charging, or demand-response participation. Validate "
        "with site operators before acting on predictions."
    )

with evaluation_tab:
    st.subheader("Walk-forward holdout performance")
    st.write(
        "The final 20% of the time-ordered data is held out. Predictions for each "
        "test hour use earlier observations and prior model predictions—not future "
        "actual meter values."
    )
    evaluation_view = result.evaluation.tail(24 * 7)
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(
        evaluation_view["timestamp"], evaluation_view["actual_kwh"],
        label="Actual", color="#193b5a",
    )
    axis.plot(
        evaluation_view["timestamp"], evaluation_view["predicted_kwh"],
        label="Predicted", color="#e18b39",
    )
    axis.set_ylabel("Consumption (kWh per hour)")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)
    st.write(
        f"**MAE:** {result.metrics['mae']:.2f} kWh · "
        f"**RMSE:** {result.metrics['rmse']:.2f} kWh · "
        f"**R²:** {result.metrics['r2']:.3f}"
    )

with forecast_tab:
    st.subheader(f"Next {horizon_days} days")
    figure, axis = plt.subplots(figsize=(12, 4))
    recent = data.tail(24 * 3)
    axis.plot(recent[TIMESTAMP_COLUMN], recent[TARGET_COLUMN], label="History")
    axis.plot(
        result.forecast["timestamp"], result.forecast["predicted_kwh"],
        label="Forecast", color="#e18b39",
    )
    axis.set_ylabel("Consumption (kWh per hour)")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)
    st.dataframe(result.forecast.head(24), use_container_width=True)

with data_tab:
    st.subheader("Cleaned input preview")
    st.dataframe(data.tail(20), use_container_width=True)
    st.markdown(
        """
        **Model:** HistGradientBoostingRegressor with calendar features, 1-hour,
        24-hour, and 168-hour lags, and a lagged 24-hour rolling mean.

        **Limitations:** This is a synthetic proof-of-work demo. It does not model
        tariffs, weather feeds, holidays, outages, or equipment constraints.
        """
    )

if source == "Virtual building simulation":
    data.to_csv(DATA_DIR / "energy_consumption.csv", index=False)
else:
    data.to_csv(OUTPUT_DIR / "cleaned_energy_data.csv", index=False)
result.evaluation.to_csv(OUTPUT_DIR / "evaluation_predictions.csv", index=False)
result.forecast.to_csv(OUTPUT_DIR / "energy_forecast.csv", index=False)

st.success(
    "Run complete. Forecast and evaluation CSV files are saved in the outputs folder."
)
