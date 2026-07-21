# LEMD Madrid multi-NWP METAR Tmax model

## Contract

- Target: maximum whole-degree temperature reported by LEMD METAR during the `Europe/Madrid` local day.
- Intraday state: LEMD METAR observations known by the issue time.
- NWP state: Open-Meteo Previous Runs `previous_day1`, a fixed 24-hour lead trajectory known by local day start.
- NWP providers: DWD ICON-EU, ECMWF IFS 0.25, NCEP GFS Global and Meteo-France ARPEGE Europe.
- Supported live issue hours: 06:00 through 20:59 local.
- Runtime requires at least two available NWP providers and renormalizes the trained weights over available providers.

## Data

- Common period: 2024-02-05 through 2026-07-18.
- Quality-controlled target days: 895.
- Intraday rows: 7,160 at 06, 08, 10, 12, 14, 16, 18 and 20 local.
- Walk-forward evaluation: seven expanding-window folds and 420 independent test days.
- Leakage audit failures: 0.

## Candidate comparison

Each single NWP provider, an optimized non-negative NWP blend, METAR-only ML,
single-NWP METAR models, the multi-NWP METAR model and a LETO spatial/wind
candidate were evaluated on identical test states. Raw, unimodal and
temperature-scaled unimodal PMFs were scored separately.

The selected production variant is `metar_multinwp__unimodal`:

| Metric | Result |
| --- | ---: |
| Expected Tmax MAE | 0.661 C |
| Expected Tmax RMSE | 0.978 C |
| Expected Tmax bias | -0.006 C |
| Mean NLL | 1.208 |
| Mean CRPS | 0.02724 |
| 80% interval coverage | 92.4% |
| Most-likely-bin hit rate | 52.9% |
| Most-likely-bin error >= 2 C | 10.7% |

The optimized raw NWP blend reached 0.751 C daily MAE. Individual raw daily
Tmax MAE was 0.858 C for ECMWF, 0.885 C for ICON-EU, 0.927 C for GFS and
1.117 C for ARPEGE. The spatial/wind candidate improved expected-value MAE by
only 0.004 C while slightly worsening NLL, so it was not selected.

Final full-history weights are:

- ICON-EU: 31.7%
- ECMWF IFS: 21.5%
- GFS: 28.0%
- ARPEGE Europe: 18.8%

## Runtime

`scripts/115_predict_lemd_metar_tmax.py` refreshes AWC METAR, obtains the four
D-1 trajectories, creates the live feature row, validates source availability,
predicts the PMF, stores JSON/JSONL history and optionally sends Telegram.
`scripts/117_lemd_metar_event_job.py` runs it once for every new METAR. The
embedded scheduler invokes Madrid through the existing multi-airport jobs.

The Telegram chat defaults to `-1004409683948`. The token lookup order is
`TELEGRAM_BOT_TOKEN_LEMD`, `TELEGRAM_BOT_TOKEN_LFPB`, then
`TELEGRAM_BOT_TOKEN`.
