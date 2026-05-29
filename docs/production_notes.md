# Production notes

Run operational archiving before relying on live NWP features. Forecast logs are append-only JSONL in `data/logs/forecast_log.jsonl`.

For Railway deployment, see `docs/railway_deploy.md`. The key requirement is a shared persistent Volume mounted at `/app/data` for the API and all cron services.

Recommended schedule:

- archive METAR/TAF/NWP every cycle;
- update outcomes after DWD truth is available;
- retrain monthly or quarterly;
- recalibrate more frequently only if validation data is sufficient.

Current model artifacts:

- `data/models/quantile_mvp.joblib`
- `data/models/quantile_mvp.metadata.json`
- `data/models/quantile_mvp.calibrator.joblib`
- `data/models/quantile_mvp.calibrator.metadata.json`

Monitoring:

```bash
python scripts/12_update_outcomes.py
python scripts/14_build_monitoring_report.py
python scripts/17_build_operational_monitoring.py
python scripts/18_check_data_freshness.py --no-fail-on-missing
```

This joins immutable forecast logs with completed DWD truth targets and writes `data/reports/forecast_monitoring.parquet` when matching outcomes exist.
`scripts/17_build_operational_monitoring.py` adds by-model, source-mismatch, availability, and acceptance summaries.
Runtime freshness checks mark METAR stale after 2 hours, TAF after 12 hours, and NWP after 18 hours relative to the forecast issue time. Stale data does not block prediction, but it is surfaced in warnings, CLI status, and monitoring reports.
`scripts/18_check_data_freshness.py` can be used by cron/monitoring as a non-zero alerting gate. Use `--no-fail-on-missing` during forward-deployment phases where a source may not yet be archived.
Runtime extrapolation checks compare live numeric features against the active model's stored training ranges. Minor extrapolation in soft meteorological features is kept as a caution; severe extrapolation remains a hard quality reason. Issue times outside the configured MVP issue schedule are still treated as hard quality reasons.
`scripts/25_pending_truth_cron.py` is the recommended scheduled truth/outcome loop. Without `--fetch`, it reports whether completed target dates are ready for DWD truth refresh and can fail with `--fail-if-ready` for alerting. With `--fetch`, it refreshes DWD observations for ready dates, rebuilds daily targets, updates forecast outcomes, writes operational monitoring tables, and refreshes `docs/monitoring_report.md` plus `docs/first_analysis.md`.
`scripts/22_predict_operational.py` writes a single operational run report to `data/reports/latest_operational_prediction.json`. The report includes the probability distribution, lineage, freshness, extrapolation, source compatibility, quality status, acceptance gate result, and optional refresh summary. Forecast logs now persist the same quality and acceptance metadata in `raw_input_metadata` for later outcome analysis.
`scripts/27_run_operational_cycle.py` is the preferred all-in-one operational command. It runs optional source refresh, predicts, logs, writes `data/reports/latest_operational_prediction.json`, writes `data/reports/latest_operational_cycle.json`, rebuilds outcome status and operational inventory tables, and optionally refreshes monitoring/first-analysis docs. Use `--require-ok` when scheduler success should mean the forecast passed acceptance gates.
`scripts/28_check_launch_readiness.py` separates forward operational readiness from outcome-monitoring readiness. `--require-forward-ready` is suitable for deployment smoke checks; `--require-outcome-ready` should only pass after at least one logged forecast has completed DWD truth.

API operational endpoints:

- `GET /health`
- `GET /model-info`
- `GET /registry-health`
- `GET /data-freshness-health`
- `GET /monitoring-summary`
- `GET /operational-monitoring`
- `GET /first-analysis`
- `POST /prepare-operational-run`
- `POST /predict-operational`
- `GET /pending-truth`
- `POST /pending-truth-cron`

Retraining and model promotion:

```bash
python scripts/15_retrain_and_promote.py --airport EDDM
python scripts/16_check_model_registry.py
python scripts/18_check_data_freshness.py --no-fail-on-missing
```

The retraining script runs the leakage audit, evaluates a holdout quantile backtest, trains a timestamped candidate, writes `data/reports/retraining_report.json` and `docs/retraining_report.md`, then promotes the candidate in `data/models/model_registry.json` only if quality gates pass. Runtime prediction resolves the active registry model first and falls back to `data/models/quantile_mvp.joblib` when no promoted model exists.

`scripts/16_check_model_registry.py` writes `data/reports/model_registry_health.json` and exits non-zero if the active or fallback model cannot be resolved.

Suggested daily cycle:

1. Archive operational data:
   `python scripts/11_archive_operational_data.py --airport EDDM --target-date YYYY-MM-DD --provider open-meteo`
   `python scripts/11b_archive_awc_live.py --airport EDDM`
2. Predict and log:
   `python -m weather_tmax_bot.bot.cli predict --airport EDDM --target-date YYYY-MM-DD --issue-time now`
   `python scripts/27_run_operational_cycle.py --airport EDDM --target-date YYYY-MM-DD --issue-time now`
   `python scripts/27_run_operational_cycle.py --airport EDDM --target-date YYYY-MM-DD --issue-time now --require-ok`
3. After DWD truth is available, refresh observations/targets and outcomes:
   `python scripts/02_download_observations.py --airport EDDM --start YYYY-MM-DD --end YYYY-MM-DD`
   `python scripts/05_build_target.py --airport EDDM`
   `python scripts/12_update_outcomes.py`
4. Rebuild monitoring:
   `python scripts/13_run_leakage_audit.py`
   `python scripts/12_update_outcomes.py`
   `python scripts/17_build_operational_monitoring.py`
   `python scripts/18_check_data_freshness.py --no-fail-on-missing`
   `python scripts/14_build_monitoring_report.py`
   `python scripts/19_build_first_analysis.py`
   `python scripts/20_operational_smoke.py --airport EDDM --no-fail-on-stale`
   `python scripts/21_prepare_operational_run.py --airport EDDM --target-date YYYY-MM-DD`
   `python scripts/22_predict_operational.py --airport EDDM --target-date YYYY-MM-DD`
   `python scripts/22_predict_operational.py --airport EDDM --target-date YYYY-MM-DD --require-ok`
   `python scripts/27_run_operational_cycle.py --airport EDDM --target-date YYYY-MM-DD --issue-time now --no-update-reports`
   `python scripts/28_check_launch_readiness.py --require-forward-ready`
   `python scripts/23_update_outcomes_and_reports.py`
   `python scripts/24_refresh_pending_truth.py`
   `python scripts/24_refresh_pending_truth.py --fetch`
   `python scripts/25_pending_truth_cron.py --fail-if-ready`
   `python scripts/25_pending_truth_cron.py --fetch`
   `python scripts/25_pending_truth_cron.py --fetch --no-update-reports`
   `python scripts/26_build_outcome_analysis.py`
5. Monthly or quarterly retraining:
   `python scripts/15_retrain_and_promote.py --airport EDDM`
   `python scripts/16_check_model_registry.py`
