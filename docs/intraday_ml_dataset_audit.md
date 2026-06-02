# Intraday ML dataset audit

One row per target local day and issue time. Features remain restricted to as-of knowledge; labels are appended only after the local day closes.

- rows: `15306`
- distinct target days: `2187`
- period: `2020-01-01` to `2025-12-31`
- NWP available rows: `1100` (`7.2%`)
- TAF available rows: `0` (`0.0%`)
- leakage checks passed: `100.0%`

## Labels

- `peak_already_passed`: `32.7%`
- `upside_ge_1c`: `58.5%`
- `upside_ge_2c`: `48.0%`
- `upside_ge_3c`: `41.1%`

## By issue hour

| issue_hour_utc | rows | distinct_days | mean_remaining_upside_c | peak_already_passed_rate | upside_ge_1c_rate | upside_ge_2c_rate | upside_ge_3c_rate | nwp_available_rate | taf_available_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2186 | 2186 | 7.1394327539 | 0.0475754803 | 0.9135407136 | 0.8417200366 | 0.7721866423 | 0.071820677 | 0.0 |
| 3 | 2187 | 2187 | 6.9280749886 | 0.0557841792 | 0.9044352995 | 0.8225880201 | 0.7553726566 | 0.0717878372 | 0.0 |
| 6 | 2187 | 2187 | 6.355829904 | 0.0626428898 | 0.8957475995 | 0.8116140832 | 0.7352537723 | 0.0717878372 | 0.0 |
| 9 | 2186 | 2186 | 3.4994967978 | 0.0800548948 | 0.8577310156 | 0.7319304666 | 0.5786825252 | 0.0713632205 | 0.0 |
| 12 | 2186 | 2186 | 1.034126258 | 0.3096980787 | 0.467063129 | 0.1390667887 | 0.0333943275 | 0.071820677 | 0.0 |
| 15 | 2187 | 2187 | 0.2079103795 | 0.8477366255 | 0.0393232739 | 0.0086877 | 0.003657979 | 0.0722450846 | 0.0 |
| 18 | 2187 | 2187 | 0.1609053498 | 0.8847736626 | 0.0192043896 | 0.0045724737 | 0.0018289895 | 0.0722450846 | 0.0 |

## Limitations

- Historical IEM TAF archive is currently empty, so TAF features gracefully degrade to missing flags during training.
- Forecast-as-issued ICON-D2 overlap starts in late May 2025; the core model is trained to remain usable when NWP is missing.
- Historical rows use scheduled 00/03/06/09/12/15/18 UTC issues. Railway METAR-event timing remains a forward-shadow concern.
