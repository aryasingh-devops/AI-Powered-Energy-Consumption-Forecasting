# Project guide: build, simulate, and publish

## What the project does

**Simple explanation:** energy forecasting estimates how much electricity a site may
use in upcoming hours, based on how it used electricity before.

**Technical explanation:** the pipeline turns an ordered hourly meter series into a
supervised regression problem. Calendar variables describe recurring cycles; lagged
consumption and a rolling mean capture recent and weekly patterns. A gradient-boosting
regressor learns the mapping from features to kWh. Future predictions are recursive,
so a prediction at hour *t* becomes an input feature for hour *t+1*.

Facilities can use forecasts as one input to reduce avoidable peak demand, schedule
flexible loads, plan storage charging, and prepare demand-response offers. Industry
deployment also needs weather, prices, holidays, equipment state, data-quality
monitoring, and a human approval process. This student project simulates the workflow,
not a live energy-management system.

## End-to-end workflow

1. **Collect:** use simulated readings or a permitted, documented hourly CSV dataset.
2. **Clean:** parse timestamps and numeric kWh, sort, drop invalid rows and duplicates,
   reject negative values and gaps.
3. **Engineer:** create calendar variables, 1/24/168-hour lags, and a lagged rolling mean.
4. **Train:** fit only on the earlier portion of a time-ordered dataset.
5. **Evaluate:** forecast the held-out tail recursively and calculate RMSE, MAE, R².
6. **Forecast:** generate the requested horizon from the final available reading.
7. **Visualize:** inspect history, holdout actual-vs-predicted values, and future values.
8. **Decide:** treat estimates as decision support; compare with business constraints.

## Phases: practical checklist

| Phase | What / why | Expected output and verification | Common mistake |
| --- | --- | --- | --- |
| 1 — Setup | Install Python and isolated dependencies | `python --version`, `pip list` | Installing globally or using the wrong interpreter |
| 2 — Dataset loading | Simulate or upload timestamp/consumption CSV | Preview rows and date range | Wrong column names or local time-zone inconsistencies |
| 3 — Data cleaning | Parse, sort, deduplicate, validate continuity | No null, duplicate, negative, or missing hourly rows | Filling gaps without documenting the imputation |
| 4 — Feature engineering | Add calendar and historical lag features | Lag features exist after 168 warm-up hours | Using future actual values in the features |
| 5 — Model building | Train gradient boosting on the earlier time range | Model fits without GPU | Randomly shuffling a time series |
| 6 — Evaluation | Walk-forward holdout prediction and metrics | Actual/predicted plot and RMSE/MAE/R² | Reporting training scores as test performance |
| 7 — Forecasting | Recursively predict future hourly usage | Forecast has exactly `days * 24` future timestamps | Assuming future lag values are known |
| 8 — Visualization | Inspect profile, test behavior, future estimates | Dashboard tabs and CSV outputs | Plotting only a metric with no units/time axis |
| 9 — GitHub publishing | Commit code/tests/docs/screenshots, exclude private/generated files | Reproducible clone/install/run instructions | Publishing credentials or unlicensed data |
| 10 — Final output | Explain limitations and potential business actions | A demo that runs end-to-end locally | Claiming simulated metrics as corporate savings |

## Virtual simulation walkthrough

1. Launch `streamlit run main.py` and leave **Virtual building simulation** selected.
2. The seeded generator creates two years of hourly readings with work schedules,
   weekend behavior, annual seasonality, a gradual trend, and random noise.
3. The dashboard validates every timestamp, learns on the first 80% and evaluates the
   last 20% in timestamp order.
4. The user selects 1–14 days; the app predicts each hour using only available history
   and earlier predictions.
5. Review the Overview graph, holdout graph/metrics, forecast graph/table, and data
   preview. The forecast can inform a hypothetical battery schedule or flexible-load
   plan, but this app does not optimize or dispatch equipment.
6. Capture screenshots (with no private upload data) and keep forecast/evaluation CSVs
   locally. The output CSVs are git-ignored by default.

Suggested evidence: dashboard overview, holdout actual-vs-predicted plot, forecast
plot and CSV header, passing test output, and a README with exact run steps.

## Source files

- `src/energy_forecasting/data.py`: deterministic simulation and input validation.
- `src/energy_forecasting/features.py`: supervised and recursive feature construction.
- `src/energy_forecasting/modeling.py`: model fit, walk-forward evaluation, metrics,
  future predictions.
- `main.py`: interactive Streamlit browser interface and result exports.
- `tests/test_pipeline.py`: simulation, cleaning, features, validation, and forecast tests.

## GitHub publishing steps

1. Create a public GitHub repository with a clear project description.
2. Create a local virtual environment and install `requirements.txt`.
3. Run `python -m unittest discover -s tests -v` and `streamlit run main.py`.
4. Add screenshots to `images/` and link them in the README if desired.
5. Check `git status`; confirm `.env`, private datasets, and generated outputs are not
   being tracked. Review dataset licenses before redistributing any real data.
6. Commit and push:

```bash
git init
git add README.md requirements.txt .gitignore main.py src tests docs images data models outputs notebooks
git commit -m "Build energy consumption forecasting dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/energy-consumption-forecasting.git
git push -u origin main
```

## Further industry-oriented enhancements

- Compare gradient boosting to seasonal-naive and linear-regression baselines.
- Add weather, holidays, electricity prices, and explicit meter metadata.
- Report metrics by hour/day/site and calculate peak-demand forecast errors.
- Add data freshness checks, drift tracking, model versioning, and retraining triggers.
- Include prediction intervals and a monitored operator approval workflow before actions.
