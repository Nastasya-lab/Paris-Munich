# Operational Ablation Report

This report compares logged model variants and phase-selection policies on completed operational days.

## Recommendations

- Keep production_champion, but log phase_arbitrated_shadow_v1 as the leading promotion candidate.
- Increase safe-blend attention in the 12-16 local window; it is the most useful daytime improvement.
- Keep ML shadow as a late-day signal, but do not promote it globally because morning behavior is unstable.
- Discard base_prior as a standalone operational candidate; keep it only as the NWP prior component.

## all

| variant | rows | days | MAE | median MAE | bias | P(actual bin) | bin ok |
|---|---:|---:|---:|---:|---:|---:|---:|
| policy_dynamic_phase | 255 | 7 | 0.955 | 0.259 | +0.309 | 0.497 | 0.584 |
| policy_conservative_phase | 255 | 7 | 0.958 | 0.262 | +0.312 | 0.494 | 0.584 |
| policy_safe_blend_promoted | 255 | 7 | 0.983 | 0.257 | +0.303 | 0.487 | 0.584 |
| production_champion | 255 | 7 | 0.992 | 0.258 | +0.299 | 0.478 | 0.592 |
| policy_seasonal_promoted | 255 | 7 | 1.041 | 0.379 | +0.538 | 0.512 | 0.580 |
| shadow_seasonal_intraday | 241 | 5 | 1.054 | 0.371 | +0.532 | 0.512 | 0.568 |
| policy_ml_promoted | 255 | 7 | 1.139 | 0.343 | -0.662 | 0.467 | 0.510 |
| shadow_safe_blend | 180 | 4 | 1.167 | 0.260 | +0.457 | 0.519 | 0.633 |
| shadow_intraday_ml | 191 | 4 | 1.338 | 0.474 | -0.875 | 0.476 | 0.534 |
| base_prior | 219 | 5 | 1.403 | 0.685 | +0.907 | 0.347 | 0.493 |

## day_window_10_20

| variant | rows | days | MAE | median MAE | bias | P(actual bin) | bin ok |
|---|---:|---:|---:|---:|---:|---:|---:|
| shadow_safe_blend | 79 | 4 | 0.351 | 0.217 | +0.244 | 0.611 | 0.734 |
| shadow_seasonal_intraday | 108 | 5 | 0.388 | 0.270 | +0.135 | 0.576 | 0.657 |
| policy_dynamic_phase | 118 | 7 | 0.407 | 0.235 | +0.145 | 0.570 | 0.686 |
| policy_seasonal_promoted | 118 | 7 | 0.410 | 0.279 | +0.179 | 0.575 | 0.678 |
| policy_conservative_phase | 118 | 7 | 0.412 | 0.258 | +0.150 | 0.564 | 0.686 |
| policy_safe_blend_promoted | 118 | 7 | 0.455 | 0.235 | +0.093 | 0.552 | 0.686 |
| production_champion | 118 | 7 | 0.462 | 0.245 | +0.098 | 0.546 | 0.703 |
| shadow_intraday_ml | 86 | 4 | 0.473 | 0.224 | +0.210 | 0.573 | 0.651 |
| policy_ml_promoted | 118 | 7 | 0.533 | 0.317 | +0.093 | 0.547 | 0.619 |
| base_prior | 86 | 4 | 0.713 | 0.500 | +0.593 | 0.426 | 0.593 |

## main_heating_12_20

| variant | rows | days | MAE | median MAE | bias | P(actual bin) | bin ok |
|---|---:|---:|---:|---:|---:|---:|---:|
| shadow_safe_blend | 67 | 4 | 0.296 | 0.198 | +0.232 | 0.677 | 0.806 |
| shadow_seasonal_intraday | 88 | 5 | 0.332 | 0.226 | +0.161 | 0.640 | 0.716 |
| policy_dynamic_phase | 98 | 7 | 0.350 | 0.199 | +0.190 | 0.641 | 0.745 |
| policy_conservative_phase | 98 | 7 | 0.356 | 0.235 | +0.195 | 0.634 | 0.745 |
| policy_seasonal_promoted | 98 | 7 | 0.365 | 0.242 | +0.211 | 0.633 | 0.735 |
| shadow_intraday_ml | 70 | 4 | 0.393 | 0.089 | +0.285 | 0.655 | 0.743 |
| policy_safe_blend_promoted | 98 | 7 | 0.408 | 0.218 | +0.131 | 0.618 | 0.745 |
| production_champion | 98 | 7 | 0.416 | 0.213 | +0.133 | 0.612 | 0.765 |
| policy_ml_promoted | 98 | 7 | 0.460 | 0.204 | +0.180 | 0.616 | 0.704 |
| base_prior | 70 | 4 | 0.741 | 0.500 | +0.684 | 0.449 | 0.614 |

## late_16_20

| variant | rows | days | MAE | median MAE | bias | P(actual bin) | bin ok |
|---|---:|---:|---:|---:|---:|---:|---:|
| shadow_intraday_ml | 33 | 4 | 0.056 | 0.043 | +0.056 | 0.960 | 1.000 |
| shadow_seasonal_intraday | 41 | 5 | 0.095 | 0.057 | +0.095 | 0.920 | 1.000 |
| shadow_safe_blend | 33 | 4 | 0.098 | 0.090 | +0.098 | 0.921 | 1.000 |
| policy_ml_promoted | 47 | 7 | 0.123 | 0.051 | +0.123 | 0.911 | 0.979 |
| policy_dynamic_phase | 47 | 7 | 0.132 | 0.056 | +0.132 | 0.902 | 0.979 |
| policy_conservative_phase | 47 | 7 | 0.144 | 0.058 | +0.144 | 0.888 | 0.979 |
| policy_seasonal_promoted | 47 | 7 | 0.144 | 0.058 | +0.144 | 0.888 | 0.979 |
| policy_safe_blend_promoted | 47 | 7 | 0.152 | 0.093 | +0.152 | 0.883 | 0.979 |
| production_champion | 47 | 7 | 0.171 | 0.122 | +0.171 | 0.869 | 0.979 |
| base_prior | 33 | 4 | 0.735 | 0.550 | +0.735 | 0.519 | 0.788 |
