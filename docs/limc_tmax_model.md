# LIMC Milan Malpensa METAR Tmax model

## Operational target

The target is the maximum integer temperature reported by LIMC METAR during
the Europe/Rome local day. Live forecasts are recalculated for every new LIMC
METAR report.

## Historical contract

- METAR source: IEM LIMC archive.
- NWP source: Open-Meteo Previous Runs fixed 24-hour lead values.
- Candidate NWP models: ICON-D2, ICON-EU and Meteo-France ARPEGE Europe.
- Common complete period: 2024-02-05 through 2026-07-20.
- Intraday issue hours: every local hour from 06:00 through 20:00.
- Dataset: 894 days and 13,410 as-of states with zero leakage failures.
- Final training: 11,610 rows; calibration: 1,800 rows.

## Walk-forward selection

Seven folds cover 420 independent test days and 6,300 intraday test states.

| Candidate | MAE C | NLL | CRPS | Mode hit | Mode error >=2 C | Coverage 80 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ICON-D2-centered unimodal | 0.573 | 1.018 | 0.0254 | 60.2% | 9.5% | 94.6% |
| Multi-NWP unimodal | 0.592 | 1.025 | 0.0271 | 59.0% | 12.3% | 94.3% |
| ICON-EU unimodal | 0.637 | 1.040 | 0.0275 | 58.4% | 13.8% | 95.2% |
| ARPEGE unimodal | 0.635 | 1.079 | 0.0260 | 58.3% | 13.8% | 93.9% |
| METAR-only unimodal | 0.909 | 1.475 | 0.0475 | 52.0% | 23.2% | 91.9% |

The selected production variant is `metar_icon_d2_single__unimodal`. Its
residual distribution is centered on ICON-D2. The ML features also retain the
optimized aggregate NWP context, whose fitted daily weights are approximately
62.7% ICON-D2 and 37.3% ICON-EU; ARPEGE receives zero linear Tmax weight.
ARPEGE remains available in the live diagnostic comparison.

The sharper temperature-scaled candidate has lower point MAE but undercovers
the observed target and is therefore not selected for probabilistic use.

## Live behavior

- 06:00-20:59 local: trained intraday model.
- Before 06:00: conservative ICON-D2 residual PMF.
- After 20:59: the trained 20:00 profile with the latest observed METAR Tmax.
- ICON-D2 is required. ICON-EU and ARPEGE failures are tolerated.
- Every forecast is written to `data/logs/limc_forecast_history.jsonl`.
- The latest payload is written to `data/reports/latest_limc_prediction.json`.

The live implementation is isolated in scripts 119 through 121. Existing
airport forecast behavior is not called or modified by those scripts.
