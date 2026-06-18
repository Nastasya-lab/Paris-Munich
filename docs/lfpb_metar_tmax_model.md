# LFPB METAR Tmax model

This document describes the first Paris Le Bourget (`LFPB`) model that targets the daily maximum temperature reported by METAR.

## Target

The primary target for this branch is:

`METAR_Tmax_d = max(tmpc from LFPB METAR/SPECI reports during local day d in Europe/Paris)`

This differs from official station `TX`:

- `METAR_Tmax` is the value the airport weather reports showed operationally.
- official `TX` from Météo-France station `95088001 LE BOURGET` remains a control / comparison target.
- METAR has rounding and aviation-reporting behavior, but it matches the user's desired operational question.

## Data

The first historical build used:

- IEM METAR archive for `LFPB`;
- Météo-France daily `TX` for official comparison;
- Météo-France 6-minute precipitation `RR` as a rain-state feature;
- local day boundaries in `Europe/Paris`.

The target build produced:

- `111529` historical METAR rows;
- `2345` target days;
- `2310` usable target days;
- median `48` METAR reports per day;
- `35` low-coverage days.

Official `TX` and METAR Tmax are close but not identical:

- official minus METAR mean: `+0.143 °C`;
- official minus METAR median: `+0.20 °C`;
- rounded official TX equals rounded METAR Tmax in `77.1%` of paired days;
- official TX is within `1.0 °C` of METAR Tmax in `95.7%` of paired days.

## Model

The first model is an ordinal remaining-upside model:

1. At an issue time, compute the maximum METAR temperature observed so far.
2. Predict survival probabilities for remaining upside:
   `P(final METAR Tmax >= current_max + 1 °C)`,
   `P(final METAR Tmax >= current_max + 2 °C)`,
   and so on.
3. Enforce monotonicity so that higher-upside probabilities cannot exceed lower-upside probabilities.
4. Convert the survival curve into integer temperature-bin probabilities.
5. The output distribution never assigns probability below the already observed METAR maximum.

Features include:

- local issue hour;
- current METAR maximum;
- latest METAR temperature;
- drop from current maximum;
- METAR report counts;
- 1h/3h/6h temperature trends;
- recent rain/thunder from METAR;
- CAVOK flag;
- 6-minute precipitation totals since issue-relevant windows;
- month and day-of-year seasonality.

## Backtest

The first backtest uses yearly time-based folds:

- train on years before the test year;
- test on the next full year;
- no random split.

Overall historical comparison:

| model | MAE expected | RMSE expected | mean NLL | mean CRPS | 80% coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| METAR Tmax ML upside v1 | 1.059 | 1.644 | 2.051 | 0.0548 | 0.910 |
| METAR Tmax ML upside v1 calibrated | 1.089 | 1.717 | 1.908 | 0.0565 | 0.926 |
| hourly phase-prior baseline | 1.275 | 1.983 | 1.631 | 0.0846 | 0.912 |
| persistence | 2.440 | 3.877 | 16.301 | 0.5899 | 0.410 |

Interpretation:

- The ML model is materially better than persistence.
- The ML model improves point accuracy and CRPS versus the phase-prior baseline.
- Out-of-fold isotonic survival calibration improves NLL from `2.051` to `1.908` and raises 80% interval coverage from `91.0%` to `92.6%`.
- Calibration slightly worsens MAE from `1.059 °C` to `1.089 °C`; this is acceptable for probability honesty, but the phase-prior baseline still has better NLL.
- This model is a strong candidate for a LFPB shadow/live model, but probability calibration should continue to be monitored.

## Hybrid Retraining

Historical IEM TAF was requested for `LFPB` over `2020-01-01` to `2026-06-06`, but the IEM TAF endpoint returned `0` rows. Because of that, the next retraining step did not add TAF features. We kept the leakage-safe rule: no synthetic TAF, no future weather, no fake forecast-as-issued substitution.

Instead, a hybrid candidate was trained:

`hybrid = calibrated ML remaining-upside distribution + empirical hourly/seasonal phase-prior distribution`

The phase-prior blend weight is selected on the previous year and tested on the next year. The final artifact selected a small phase-prior weight:

- final phase-prior blend weight: `0.05`;
- fold weights: mostly `0.05`, with earlier folds using `0.15`.

Hybrid rolling comparison on evaluable calibrated folds:

| model | MAE expected | RMSE expected | mean NLL | mean CRPS | 80% coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| hybrid v1 | 1.110 | 1.744 | 1.572 | 0.0573 | 0.926 |
| calibrated ML | 1.098 | 1.728 | 1.911 | 0.0569 | 0.925 |
| phase-prior | 1.284 | 2.001 | 1.680 | 0.0866 | 0.908 |

Interpretation:

- Hybrid v1 is the best probability model by NLL.
- Calibrated ML remains slightly better by MAE and CRPS.
- Hybrid v1 is a better candidate if we value honest probability bins more than the last `0.01-0.02 °C` of point accuracy.
- The hybrid is not a new parallel forecast family for the user interface; it is a candidate replacement for the LFPB production model if accepted.

## ICON-D2 NWP Stress Test

We started the same Open-Meteo Single Runs ICON-D2 workflow that was used for Munich, but now with `LFPB` coordinates from `config/airports.yaml`.

Current partial archive:

- file: `data/forecasts/open_meteo_single_runs_icon_d2_LFPB.parquet`;
- source: `open_meteo.single_run.icon_d2`;
- raw archive rows: `5424`;
- unique model runs: `1120`;
- raw target span: `2025-07-24` to `2025-12-22`;
- usable as-of joined rows: `1152`;
- joined target days: `144`;
- joined period: `2025-07-27` to `2025-12-21`;
- all joined rows preserve `knowledge_time_utc <= issue_time_utc`.
- next backfill offset after this run: `1636`.

Preliminary stress-test split:

- train: `2025-07-27` to `2025-10-31`;
- test: `2025-11-01` to `2025-12-21`;
- train rows: `744`;
- test rows: `408`.

Results on the short as-of holdout:

| model | MAE expected | RMSE expected | mean NLL | mean CRPS | 80% coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| NWP-aware ML uncalibrated | 0.645 | 0.845 | 1.687 | 0.0299 | 0.924 |
| raw ICON-D2 residual distribution | 0.671 | 0.902 | 1.661 | 0.0811 | 0.909 |
| raw ICON-D2 point | 0.635 | 0.948 | 14.222 | 0.3554 | 0.485 |
| persistence current METAR max | 1.390 | 2.215 | 14.425 | 0.5221 | 0.478 |

Interpretation:

- ICON-D2 is very promising for `LFPB METAR Tmax`.
- The NWP-aware ML candidate now has the best MAE and CRPS on the expanded holdout, despite being uncalibrated.
- The residual-distribution baseline has the best NLL, so it remains the safer probability baseline.
- The raw ICON point forecast is point-accurate but has unusable probabilistic scores because it puts all mass on one integer bin.
- The period is too short for a production decision; complete the backfill before promoting an ICON-aware model.
- ICON-D2 may still be less geographically ideal for Paris than a broader-domain model such as ICON-EU; this should be audited next if ICON-D2 coverage remains patchy.

## Live forecast

The calibrated artifact is saved at:

- `data/models/lfpb_metar_tmax_upside_v1.joblib`
- `data/models/lfpb_metar_tmax_upside_v1.metadata.json`

The hybrid candidate is saved at:

- `data/models/lfpb_metar_tmax_hybrid_v1.joblib`
- `data/models/lfpb_metar_tmax_hybrid_v1.metadata.json`

Example live/as-of command:

```bash
python scripts/48_predict_lfpb_metar_tmax.py --airport LFPB --target-date 2025-08-01 --issue-time 2025-08-01T12:00:00Z --no-auto-refresh --no-notify
```

For live operation:

```bash
python scripts/48_predict_lfpb_metar_tmax.py --airport LFPB --issue-time now --auto-refresh --notify
```

For the hybrid candidate:

```bash
python scripts/48_predict_lfpb_metar_tmax.py --airport LFPB --issue-time now --auto-refresh --notify --model-path data/models/lfpb_metar_tmax_hybrid_v1.joblib --metadata-path data/models/lfpb_metar_tmax_hybrid_v1.metadata.json
```

The live feature row uses only METAR reports with `knowledge_time_utc <= issue_time_utc`. The output distribution is built as:

`current METAR max so far + calibrated remaining-upside distribution`

Therefore it cannot assign probability below the already observed METAR maximum.

## Current limitation

This first LFPB model does not yet use TAF or forecast-as-issued NWP. It is an operational METAR-based intraday model, not a full weather-model ensemble.

Next steps:

- add immutable forecast logging for LFPB METAR Tmax predictions;
- add Telegram/Railway scheduling policy if LFPB should run operationally alongside EDDM;
- continue/complete LFPB Open-Meteo Single Runs ICON-D2 backfill;
- compare ICON-D2 against ICON-EU if available for Paris;
- add TAF features only if a real historical TAF archive becomes available.
