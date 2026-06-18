# LFPB truth-source feasibility check

Generated: 2026-06-06T14:58:46.402059Z

## Recommendation

Use Meteo-France daily `TX` from station `95088001 LE BOURGET` as the final target label.
Use Meteo-France hourly `T/TX` for intraday historical features and target timing proxies.
Use Meteo-France 6-minute `RR` as a precipitation feature source, not as temperature truth.

## Key metrics

- Daily TX rows checked: `2347`; non-null TX rows: `2347`.
- Hourly rows checked: `56357` from `2020-01-01T00:00:00+00:00` to `2026-06-06T04:00:00+00:00`.
- Hourly T non-null rate: `1.000`; hourly TX non-null rate: `0.997`.
- Local-day paired days daily TX vs hourly: `2347`.
- Daily TX minus max hourly T: mean `0.28 C`, median `0.30 C`, 95% abs `1.00 C`.
- Daily TX minus max hourly TX: mean `-0.05 C`, median `0.00 C`, 95% abs `0.70 C`.
- Max hourly TX within 0.2 C of daily TX: `91.5%`; within 0.5 C: `94.1%`.
- 6-minute RR rows checked: `563518`; median rows/day `240`.
- IEM METAR quick check 2024-2026: `ok`, rows `42415`.

## Interpretation

The result is strong enough to proceed with an LFPB MVP, but the label should be the official daily TX, not METAR max and not 6-minute RR.
The hourly TX field is a very good bridge between the official daily target and intraday dynamics. It can be used to learn when/how the day evolves, while daily TX remains the final outcome.
The 6-minute dataset is valuable for precipitation regime detection, especially rain onset and intensity, but it does not contain temperature columns for Le Bourget in the checked files.
