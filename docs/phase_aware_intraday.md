# Phase-aware intraday shadow mode

## Status

The phase-aware intraday logic is a shadow-only challenger. It does not change
the production champion forecast. Its output is logged and shown in Telegram so
that it can be evaluated against DWD truth before promotion.

## Goal

The same METAR temperature trend means different things at different local
times. A morning shower can temporarily lower temperature while substantial
heating remains possible. A similar drop after the normal afternoon peak,
combined with weak NWP upside, is evidence that the daily maximum has probably
already occurred.

The challenger classifies each same-day forecast into one of three phases:

| Phase | Main condition | Effect on shadow intraday weight |
| --- | --- | --- |
| `morning_prior` | Before 11:00 Europe/Berlin | Keep the full-day prior dominant. Cap observed-data weight at 20%, or 8% when NWP still shows at least 3 C of upside. |
| `midday_update` | Main heating window without a late cutoff signal | Allow METAR evidence to update the prior, but cap its weight at 70%. |
| `late_nowcast` | From 16:00, or earlier with a strong late cutoff signal | Give the intraday nowcast at least 65% weight. Existing survival and late-drop logic can make it stronger. |

## Scenario tracking

The shadow output records a compact scenario label:

- `temporary_disruption_possible`: early weather break while NWP still supports
  at least 3 C of future heating.
- `heating_cutoff_likely`: afternoon weather break with little or unknown NWP
  upside.
- `taf_and_metar_adverse`: adverse TAF signal and observed METAR weather break.
- `nwp_still_supports_higher_tmax`: no stronger scenario applies, but NWP still
  supports at least 2 C of upside.
- `near_observed_track`: current observations remain close to the expected
  daily track.

The metadata also preserves:

- `metar_weather_break_signal`
- `taf_adverse_weather_signal`
- `nwp_future_heating_signal`
- `temp_trend_3h_for_phase_c`
- `forecast_phase`
- `phase_reason`
- `phase_season`

## Data used

The challenger uses only information already present in the as-of feature row:

- METAR temperature, observed maximum, three-hour trend, precipitation and
  thunder indicators.
- TAF rain, shower, thunder, fog, snow, PROB30 and PROB40 adverse-weather
  indicators.
- Forecast-as-issued NWP sampled future temperatures from the archived run
  available at issue time.
- Forecast-as-issued NWP future-day remainder aggregates when available:
  precipitation sum, cloud-cover mean, shortwave radiation sum, wind-speed
  maximum, gust maximum and future temperature maximum after model availability.
- Historical METAR peak-survival statistics computed from training-period data.

No future observation or DWD target value is used as a feature.

Older archived NWP extracts may not contain the future-remainder fields. In
that case the challenger degrades gracefully to sampled local temperatures and
keeps the NWP adverse-weather signal as unavailable rather than inventing it
from post-factum data.

## Promotion gate

Promotion must be based on paired forecasts evaluated against DWD truth. At
minimum compare CRPS, NLL, MAE, interval coverage and late-day false-upside
probability by local hour and season. Keep the current production champion until
the challenger improves probabilistic quality without degrading morning
forecasts.
