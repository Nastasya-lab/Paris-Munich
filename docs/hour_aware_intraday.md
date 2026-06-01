# Hour-aware intraday shadow

This is a shadow-only refinement. The production champion remains unchanged until
live outcome verification proves that the challenger is better.

## Motivation

The same METAR signal means different things at different local hours. A rainy
drop from 18C to 15C at 08:00 local should not imply that the daily maximum has
already passed. The same type of drop at 18:00 local is much stronger evidence.

## Inputs

The shadow challenger uses:

- local issue hour in `Europe/Berlin`;
- warm/cool seasonal weight curves;
- the 2020-2025 METAR first-daily-maximum timing analysis;
- `P(peak still ahead | season, local hour)` as a seasonal survival prior;
- NWP sampled future temperatures, when available;
- drop from observed maximum and current METAR trend.

## Weight Design

The previous shadow profile used broad UTC buckets. The new profile uses a local
hour curve:

| local phase | Expected behavior |
| --- | --- |
| morning | Keep intraday weak, especially if NWP still shows future heating. |
| midday | Increase influence gradually as the day reveals itself. |
| late afternoon | Trust observed max more, but keep exceptions possible. |
| evening | Use seasonal survival prior to reduce unrealistic upside tails. |

The blend weight and the survival-tail correction are intentionally separate:

- blend weight decides how much the intraday distribution contributes;
- survival correction decides how much probability remains above observed max.

## Production Policy

This profile is logged and scored as `shadow_seasonal_intraday`. It should not be
promoted to production until `forecast_variant_monitoring.parquet` has enough
paired champion-vs-shadow outcomes across several regimes.
