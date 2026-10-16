# AI-Powered Energy Consumption Forecasting System

A beginner-friendly, industry-inspired time-series forecasting project. It generates
a reproducible virtual building meter dataset, cleans hourly readings, engineers
calendar and lag features, trains a machine-learning model, evaluates it on a
time-ordered holdout, and forecasts future energy consumption in an interactive
Streamlit dashboard.

> **Scope:** The included data is simulated for learning and portfolio demonstration.
> This is not a utility dispatch tool and should not be used for operational decisions
> without validation against real site data and domain constraints.

## Problem statement

Buildings and industrial facilities need to estimate near-term electricity demand.
Forecasts help energy teams plan flexible equipment, battery charging, demand response,
staffing, and procurement. Poor forecasts can contribute to avoidable peak demand
charges, poor scheduling, and unnecessary reserve capacity.

In simple terms, the system learns the site's past usage patterns and estimates what
the next hours may look like. Technically, it is a supervised time-series regression
pipeline: hourly consumption and time-derived/lagged features feed a
`HistGradientBoostingRegressor`. A chronological holdout and recursive walk-forward
evaluation avoid shuffling time or feeding future actual readings into test predictions.

## Industry relevance

| Use case | How a forecast can help |
| --- | --- |
| Smart cities / utilities | Anticipate load patterns and plan demand response |
| Manufacturing | Schedule energy-intensive production and maintenance windows |
| Data centers | Plan cooling and flexible compute against expected demand |
| Renewable energy | Coordinate storage and flexible loads with expected demand |
| Electricity boards | Improve short-term demand planning and reserve estimates |

Forecasting supports, rather than replaces, tariff analysis, grid constraints, weather,
operator judgement, and safety procedures.

## Tech stack and options

| Option | Tools and models | Difficulty | GPU | Expected output |
| --- | --- | --- | --- | --- |
| A — easiest | Python, Pandas, Matplotlib, scikit-learn; Linear Regression or Random Forest | Beginner | No | CSV forecast and basic plots |
| **B — selected** | **Python, Pandas, NumPy, scikit-learn HistGradientBoosting, Streamlit, Matplotlib** | **Beginner/intermediate** | **No** | **Interactive browser dashboard, evaluation, downloadable CSVs** |
| C — advanced | PyTorch/TensorFlow LSTM/Temporal CNN, weather and tariff feeds, experiment tracking | Advanced | Helpful, not mandatory | Multi-feature deep-learning service and monitoring |

Option B balances fast local execution, explainability, and a polished GitHub
demonstration. This project uses no paid API or GPU.

## Architecture and data flow

```text
Virtual hourly simulator OR uploaded CSV
                  |
                  v
 Validate schema -> sort -> remove bad rows/duplicates -> continuity check
                  |
                  v
 Calendar + lag(1h, 24h, 168h) + previous 24h rolling mean
                  |
                  v
 Chronological train/test split -> HistGradientBoostingRegressor
                  |
                  +----> recursive walk-forward test -> RMSE / MAE / R²
                  |
                  +----> recursive future forecast -> forecast CSV
                  |
                  v
 Streamlit tabs: overview, evaluation, forecast, data/methodology
```

The simulated data represents one building meter in kWh per hour. It contains daily
operating schedules, weekday/weekend differences, annual seasonality, a gentle trend,
and seeded noise. Upload CSVs with exactly these column names:

```csv
timestamp,consumption_kwh
2024-01-01 00:00:00,18.4
2024-01-01 01:00:00,17.9
```

Hourly records must be continuous, non-negative, and include at least 14 days. Missing
or irregular meter hours are rejected rather than silently interpolated.

## Folder structure

```text
AI-Powered Energy Consumption Forecasting/
├── data/                 # Generated input CSV; ignored by Git
├── docs/                 # Extended implementation and portfolio guide
├── images/               # Add dashboard screenshots here
├── models/               # Reserved for serialized model artifacts
├── notebooks/             # Optional experimentation notebooks
├── outputs/               # Evaluation and forecast CSVs; ignored by Git
├── src/energy_forecasting/ # Simulation, features, model/evaluation modules
├── tests/                 # Focused pipeline tests
├── main.py                # Streamlit browser application
├── requirements.txt
└── README.md
```

## Installation

Python 3.10–3.12 is recommended. No GPU is required.

### Windows PowerShell

```powershell
cd "C:\path\to\AI-Powered Energy Consumption Forecasting"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` in
that terminal and activate again. Alternatively use `.venv\Scripts\python.exe` directly.

### macOS / Linux

```bash
cd "/path/to/AI-Powered Energy Consumption Forecasting"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the project

From the project root with the environment activated:

```bash
streamlit run main.py
```

Streamlit opens a local browser tab (usually `http://localhost:8501`). Select the
virtual simulation or upload a CSV, set a forecast horizon of 1–14 days, and inspect
the Overview, Model evaluation, Forecast, and Data & methodology tabs.

Run the automated checks:

```bash
python -m unittest discover -s tests -v
```

The dashboard writes:

- `data/energy_consumption.csv` for the default simulated input.
- `outputs/evaluation_predictions.csv` for time-ordered test predictions.
- `outputs/energy_forecast.csv` for the selected future horizon.
- `outputs/cleaned_energy_data.csv` when using an uploaded dataset.

Generated data and outputs are intentionally git-ignored. The dashboard also shows
sample forecasts and interactive result tables. Numerical values depend on the data
and should not be presented as measured real-world performance.

## Model and evaluation

- **Model:** `HistGradientBoostingRegressor` (scikit-learn), no GPU.
- **Features:** hour, weekday, day of year, month, weekend flag; 1/24/168-hour lags;
  previous 24-hour mean.
- **Evaluation:** chronological final 20% holdout. Each test prediction is recursive:
  after the first predicted hour, later features use earlier model predictions, not
  future actual usage.
- **Forecast model:** after holdout metrics are calculated, the model is refit on all
  available observations before producing the future forecast.
- **Metrics:** RMSE (penalizes large errors), MAE (average absolute error), and R²
  (variance explained; can be negative when a model performs poorly).

No fixed metric is promised: performance changes with the dataset, forecast horizon,
and noise. Compare against a simple seasonal-naive baseline before claiming business
value on real meter data.

For the default seeded simulation, the browser dashboard reports live RMSE, MAE, R²,
and forecasted kWh alongside the charts. Exact values can vary with scikit-learn
version; use the metrics generated by your own run rather than treating a screenshot
as a guaranteed benchmark.

## Portfolio proof strategy

1. Commit source, tests, documentation, and a screenshot; never commit credentials or
   private meter data.
2. Run the app and capture Overview, evaluation, and forecast screens into `images/`.
3. Add an anonymized sample dataset only if its license permits redistribution; include
   its source and license.
4. In a project demo, explain the chronological split, recursive evaluation, RMSE/MAE,
   and limitations—not just the algorithm name.
5. Extend responsibly with a seasonal-naive baseline, weather/calendar inputs, drift
   monitoring, and a deployment workflow.

## Learning outcomes

- Data validation and reproducible synthetic time-series generation.
- Leakage-aware feature engineering and chronological validation.
- Recursive multi-step forecasts and regression metrics.
- A practical ML dashboard, file outputs, tests, and project documentation.

See [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md) for the phase-by-phase workflow,
simulation walkthrough, common mistakes, and GitHub publishing checklist.
