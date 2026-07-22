# RCSS Taipei METAR Tmax model

## Target and data contract

- Airport: RCSS Taipei Songshan, timezone `Asia/Taipei`.
- Target: maximum whole-degree temperature reported by RCSS METAR during the full local day.
- Period: 2024-02-06 through 2026-07-20.
- Complete target days: 735.
- Intraday states: 11,025 at every local hour from 06:00 through 20:00.
- NWP contract: Open-Meteo Previous Runs `previous_day1`; every NWP value was available before the target local day.
- Sources: JMA MSM, JMA GSM, DWD ICON Global and ECMWF IFS 0.25.
- Spatial candidate: RCTP; wind/advection candidate: RCSS plus RCTP.
- Leakage failures: 0.

The backtest used nine expanding walk-forward folds. Each fold had a 90-day calibration period and a subsequent independent 30-day test period. This produced 270 independent test days and 4,050 test states.

## Raw NWP Tmax

The uncorrected D-1 model Tmax values substantially underforecast the RCSS METAR maximum.

| Source | MAE, C | RMSE, C | Bias, C | Within 1 C |
|---|---:|---:|---:|---:|
| ECMWF IFS | 1.768 | 2.168 | -1.270 | 32.6% |
| ICON Global | 2.307 | 2.812 | -1.990 | 26.3% |
| JMA MSM | 2.582 | 2.969 | -2.418 | 17.0% |
| JMA GSM | 3.238 | 3.638 | -3.071 | 14.4% |

The raw ranking must not be confused with the ranking of the trained intraday models. The trained model learns each source's systematic RCSS bias and combines it with the observed state of the current day.

## Intraday candidates

All PMF candidates below use the same walk-forward dates. `temperature_unimodal_067` is the already validated probability-shape contract used by the other project cities.

| Candidate | MAE, C | RMSE, C | NLL | CRPS | Mode hit | Large mode error | 80% coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| JMA GSM anchor + aggregate NWP | **0.469** | **0.858** | **0.859** | 0.02177 | 67.5% | **10.9%** | 92.9% |
| ECMWF anchor + aggregate NWP | 0.472 | 0.872 | 0.862 | 0.02151 | **67.9%** | 11.4% | 92.9% |
| JMA MSM anchor + aggregate NWP | 0.485 | 0.883 | 0.875 | **0.02126** | 67.1% | 12.0% | 92.5% |
| Multi-NWP plus RCTP spatial/wind | 0.493 | 0.895 | 0.911 | 0.02137 | 67.3% | 11.3% | 92.0% |
| Multi-NWP without RCTP | 0.509 | 0.915 | 0.909 | 0.02192 | 66.3% | 12.4% | 92.4% |
| METAR only | 0.605 | 1.163 | 1.017 | 0.03306 | 64.6% | 15.5% | 92.3% |

For 10:00-17:00 local, the selected candidate achieved MAE 0.341 C, NLL 0.660 and mode hit rate 74.7%.

The paired seven-day block bootstrap comparison between the JMA GSM and ECMWF candidates did not establish a significant difference:

- MAE difference, JMA GSM minus ECMWF: -0.0028 C; 95% interval -0.0225 to +0.0178 C.
- NLL difference: -0.0030; 95% interval -0.0353 to +0.0306.
- CRPS difference: +0.00026; 95% interval -0.00096 to +0.00152.

JMA GSM was selected by the predeclared composite criterion, but ECMWF is statistically tied and should remain an important monitored diagnostic.

## Regression experiment

Ridge, HistGradientBoosting and Extra Trees were trained to correct the residual between NWP Tmax and the final RCSS METAR Tmax. Their equal-weight mean improved point stability, but its probability distribution was inferior to the survival model.

| Candidate | MAE, C | RMSE, C | NLL | CRPS |
|---|---:|---:|---:|---:|
| Equal Ridge + HGB + Extra Trees | **0.516** | **0.875** | 1.133 | 0.05880 |
| HistGradientBoosting | 0.529 | 0.916 | **1.095** | **0.05627** |
| Extra Trees | 0.558 | 0.957 | 1.186 | 0.06328 |
| Ridge | 0.559 | 0.914 | 1.102 | 0.06904 |

The equal mean is useful as a research point forecast, but it is not suitable as production while its NLL and CRPS are materially worse.

## Production decision

Production artifact: `rcss_metar_tmax_multinwp_d1_v1.joblib`.

- Absolute anchor: JMA GSM.
- Supporting aggregate features: normally ICON Global and ECMWF, according to weights fitted only on past data.
- PMF shape: temperature-scaled unimodal, temperature 0.67.
- Live safety gate: JMA GSM and at least one additional NWP source must be available.
- Before 06:00 local: conservative residual mode using the 06:00 profile.
- 06:00-20:00 local: trained hourly intraday mode.
- After 20:00 local: clamped 20:00 profile and observed-METAR lower bound.
- A forecast is emitted after every new RCSS METAR and on the common scheduled forecast cycles.

The model stores every forecast in `data/logs/rcss_forecast_history.jsonl`. Prospective monitoring is required because the statistical tie with ECMWF means a future promotion decision should be based on live outcomes, not on this backtest alone.
