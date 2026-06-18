# LFPB ICON-D2 vs ECMWF IFS diagnostic

- created: `2026-06-09T20:30:38.352972+00:00`
- rows: `2432`
- days: `304`
- period: `2025-07-27` to `2026-05-30`

## Overall

| source | rows | days | mae | rmse | bias | median_abs_error | exact_integer_hit_rate | within_1c_integer_rate | mean_abs_icon_ecmwf_spread |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecmwf | 2432 | 304 | 1.0905 | 1.3310 | -0.8734 | 1.0000 | 0.2615 | 0.7405 | 0.8537 |
| icon | 2432 | 304 | 0.7215 | 0.9491 | -0.3041 | 0.6000 | 0.4437 | 0.8931 | 0.8537 |

## By Phase

| phase | source | rows | days | mae | rmse | bias | median_abs_error | exact_integer_hit_rate | within_1c_integer_rate | mean_abs_icon_ecmwf_spread |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| afternoon | ecmwf | 304 | 304 | 1.1033 | 1.3516 | -0.8783 | 1.0000 | 0.2533 | 0.7401 | 0.9132 |
| afternoon | icon | 304 | 304 | 0.6664 | 0.8839 | -0.1947 | 0.5000 | 0.4836 | 0.9276 | 0.9132 |
| before_work | ecmwf | 608 | 304 | 1.0872 | 1.3169 | -0.8319 | 1.0000 | 0.2615 | 0.7352 | 0.8347 |
| before_work | icon | 608 | 304 | 0.7577 | 0.9414 | -0.2979 | 0.6000 | 0.3882 | 0.8816 | 0.8347 |
| evening | ecmwf | 608 | 304 | 1.0766 | 1.3240 | -0.9336 | 1.0000 | 0.2582 | 0.7632 | 0.8757 |
| evening | icon | 608 | 304 | 0.6316 | 0.9131 | -0.3030 | 0.5000 | 0.5214 | 0.9260 | 0.8757 |
| midday | ecmwf | 608 | 304 | 1.0990 | 1.3415 | -0.8641 | 1.0000 | 0.2632 | 0.7319 | 0.8286 |
| midday | icon | 608 | 304 | 0.7688 | 0.9915 | -0.3428 | 0.6000 | 0.4309 | 0.8717 | 0.8286 |
| morning | ecmwf | 304 | 304 | 1.0947 | 1.3313 | -0.8500 | 1.0000 | 0.2730 | 0.7237 | 0.8385 |
| morning | icon | 304 | 304 | 0.7891 | 1.0094 | -0.3510 | 0.6500 | 0.3849 | 0.8586 | 0.8385 |

## Winners By Phase

| phase | rows | days | icon_wins | ecmwf_wins | ties | icon_win_rate | ecmwf_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| afternoon | 304 | 304 | 208 | 79 | 17 | 0.6842 | 0.2599 |
| before_work | 608 | 304 | 387 | 197 | 24 | 0.6365 | 0.3240 |
| evening | 608 | 304 | 436 | 142 | 30 | 0.7171 | 0.2336 |
| midday | 608 | 304 | 379 | 193 | 36 | 0.6234 | 0.3174 |
| morning | 304 | 304 | 183 | 105 | 16 | 0.6020 | 0.3454 |

## Simple Weight Baseline

| phase | weight_icon | weight_ecmwf | mae | rows | days |
| --- | --- | --- | --- | --- | --- |
| afternoon | 1.0000 | 0.0000 | 0.6664 | 304 | 304 |
| before_work | 0.9500 | 0.0500 | 0.7569 | 608 | 304 |
| evening | 1.0000 | 0.0000 | 0.6316 | 608 | 304 |
| midday | 0.9500 | 0.0500 | 0.7680 | 608 | 304 |
| morning | 0.9500 | 0.0500 | 0.7876 | 304 | 304 |

## Limitations

- This diagnostic compares raw NWP daily Tmax guidance, not a fully trained probabilistic model.
- Rows are included only when both ICON-D2 and ECMWF IFS are available as-of the issue time.
- The comparison uses METAR Tmax target, not official national-climate Tmax.
- Suggested weights are a diagnostic baseline; final production weights should be validated out-of-time.
