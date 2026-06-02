# Model disagreement monitoring

Stage 7 adds a diagnostic layer that compares the operational forecast with shadow forecasts.

It does not change the production distribution.

## Compared variants

- `production_champion`: distribution returned to users.
- `shadow_seasonal_intraday`: phase-aware intraday challenger.
- `shadow_intraday_ml`: remaining-upside ML challenger when active.

## Metrics

For every forecast, the audit records:

- expected Tmax spread across available variants;
- most-likely integer-bin spread;
- spread of `P(Tmax >= 25C)`;
- spread of `P(Tmax >= 30C)`;
- per-shadow deltas versus the champion.

## Severity

| Severity | Trigger |
| --- | --- |
| `none` | Variants are close. |
| `watch` | Expected Tmax spread is at least `1.5C`, most-likely bin spread at least `2C`, or threshold probability spread at least `15 pp`. |
| `high` | Expected Tmax spread is at least `3.0C`, most-likely bin spread at least `4C`, or threshold probability spread at least `30 pp`. |

## Forecast effect

Disagreement is caution-only.

- It does not reject forecasts.
- It does not alter the champion distribution.
- It is stored in forecast logs and outcome monitoring.
- It appears in Telegram so humans can recognize difficult forecast situations.

The next promotion/blending stage can use these records to decide whether disagreement regimes should receive different handling.
