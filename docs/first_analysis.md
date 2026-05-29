# First analysis

This report summarizes whether the current MVP is ready for historical analysis and operational outcome analysis.

## Current state

Active model: `quantile_mvp_issue_features_20260529`
Registry health passed: `True`
Training rows: `15309`
Daily target rows: `2189`
Logged forecasts: `14`
Forecasts with outcomes: `0`
Forecast outcome status rows: `14`
Operational outcome stage: `pending`

## Readiness

- `historical_backtest_ready`: `True`
- `calibration_ready`: `True`
- `production_predict_ready`: `True`
- `leakage_audit_passed`: `True`
- `operational_outcome_analysis_ready`: `False`
- `operational_outcome_useful_sample`: `False`
- `operational_outcome_robust_sample`: `False`
- `operational_outcome_rows`: `0`
- `operational_outcome_stage`: `pending`
- `freshness_gate_passed`: `True`

## Selected calibration

Variant: `calibrated_spread`
Rows: `360`
NLL: `2.4413893831`
CRPS: `0.9323430385`
Coverage 50/80/90: `0.4972222222` / `0.8083333333` / `0.9416666667`

## Operational outcomes

Outcome rows: `0`
Minimum for first analysis: `1`
Minimum for useful sample: `10`
Minimum for robust monitoring: `30`

### Acceptance breakdown

No rows.

## Next actions

- Some forecasts are logged but still pending DWD truth; run outcome update after their target dates are available.
- Review `docs/backtest_results.md`, PIT/reliability plots, and rolling summaries for the first model-quality analysis.
- Continue accumulating forecast-as-issued NWP before drawing strong NWP backtest conclusions.
