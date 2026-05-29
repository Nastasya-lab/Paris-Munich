# Weather Tmax Bot

Production-oriented MVP for leakage-safe probabilistic forecasts of airport-local daily maximum 2 m temperature.

First airport: `EDDM` Munich Airport. The forecast output is a calibrated probability distribution over integer Celsius bins, where `Tmax = 22C` means `21.5 <= Tmax < 22.5`.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Optional:

```bash
pip install -e ".[ml,mostlyright]"
```

## Core commands

```bash
python scripts/01_select_station.py --airport EDDM
python scripts/02_download_observations.py --airport EDDM --start 2020-01-01 --end 2020-12-31
python scripts/05_build_target.py --airport EDDM
python scripts/06_build_dataset.py --airport EDDM
python scripts/07_train_model.py --airport EDDM
python scripts/08_run_backtest.py --airport EDDM
python scripts/08b_run_rolling_backtest.py
python scripts/08c_compare_calibration.py
python scripts/13_run_leakage_audit.py
python scripts/14_build_monitoring_report.py
python scripts/15_retrain_and_promote.py --airport EDDM
python scripts/16_check_model_registry.py
python scripts/17_build_operational_monitoring.py
python scripts/18_check_data_freshness.py --no-fail-on-missing
python -m weather_tmax_bot.bot.cli predict --airport EDDM --target-date 2026-07-15 --issue-time now
python -m weather_tmax_bot.bot.cli status
python -m weather_tmax_bot.bot.cli analyze
python scripts/20_operational_smoke.py --airport EDDM --target-date 2026-07-15 --no-fail-on-stale
python scripts/21_prepare_operational_run.py --airport EDDM --target-date 2026-07-15
python scripts/22_predict_operational.py --airport EDDM --target-date 2026-07-15
python scripts/22_predict_operational.py --airport EDDM --target-date 2026-07-15 --require-ok
python scripts/27_run_operational_cycle.py --airport EDDM --target-date 2026-07-15 --issue-time now
python scripts/27_run_operational_cycle.py --airport EDDM --target-date 2026-07-15 --issue-time now --require-ok
python scripts/29_daily_operational_run.py --no-auto-refresh --no-require-ok
python scripts/30_daily_outcome_update.py
python scripts/31_scheduler_healthcheck.py --require-forward-ready
python scripts/32_send_telegram_test.py
python scripts/23_update_outcomes_and_reports.py
python scripts/24_refresh_pending_truth.py
python scripts/25_pending_truth_cron.py --fail-if-ready
python scripts/26_build_outcome_analysis.py
python scripts/28_check_launch_readiness.py --require-forward-ready
python scripts/10_start_api.py
```

`scripts/02_download_observations.py` discovers matching DWD 10-minute station ZIP files from the DWD CDC listing. You can also pass `--zip-path` to parse a pre-downloaded archive.

The MVP works without MostlyRight SDK. If the SDK is installed, `MostlyRightAdapter` can be used for supported source metadata/cache/METAR-ASOS style ingestion, while DWD truth, TAF, NWP archiving, leakage tests, and the probabilistic model remain internal.

## Data honesty

The project has two separate layers:

- Truth layer: final or quality-controlled DWD observations used only for labels and scoring.
- Knowledge layer: only records whose `knowledge_time_utc <= issue_time_utc`, used for features.

Every dataset build runs leakage checks before rows are accepted.

Forecasts are logged to `data/logs/forecast_log.jsonl` by default. Override with `WEATHER_TMAX_FORECAST_LOG_PATH`.

Retraining writes versioned artifacts to `data/models/`, records them in `data/models/model_registry.json`, and promotes a candidate only when leakage and basic probabilistic quality gates pass. CLI/API predictions use the active registry model when present and fall back to `quantile_mvp.joblib` otherwise.

Operational monitoring joins logged forecasts to completed DWD truth, then summarizes errors by model version, source mismatch, and source availability/missingness.
Runtime predictions and `status` also report freshness/staleness for METAR, TAF, and NWP inputs so stale operational archives are visible before they become model-quality issues.
Predictions include an extrapolation diagnostic when live feature values fall outside the training feature ranges, for example an off-grid issue hour outside the MVP schedule.
For day-to-day use, `scripts/27_run_operational_cycle.py` is the single operational entry point: it can refresh sources, predict, log, apply acceptance gates, write `latest_operational_prediction.json`, write `latest_operational_cycle.json`, and refresh monitoring summaries.

Operational API endpoints:

- `GET /predict?airport=EDDM&target_date=YYYY-MM-DD&issue_time=now`
- `GET /health`
- `GET /model-info` includes active model registry status and promotion metadata
- `GET /registry-health`
- `GET /data-freshness-health`
- `GET /monitoring-summary`
- `GET /operational-monitoring`
- `GET /first-analysis`
- `POST /prepare-operational-run`
- `POST /predict-operational`
- `GET /pending-truth`
- `POST /pending-truth-cron`

Operational archive commands:

```bash
python scripts/11_archive_operational_data.py --airport EDDM --target-date YYYY-MM-DD --provider open-meteo
python scripts/11b_archive_awc_live.py --airport EDDM
```

Railway deployment is documented in `docs/railway_deploy.md`. Mount a persistent Volume at `/app/data`.
