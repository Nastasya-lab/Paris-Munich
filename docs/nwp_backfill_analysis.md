# NWP backfill analysis

This report scores Open-Meteo Single Runs ICON-D2 `model_tmax_c` against DWD daily Tmax truth.
It is a source-quality analysis, not yet a promoted production model comparison.

## Coverage

- scored rows: `850`
- first target date: `2025-09-01`
- latest target date: `2025-12-31`
- source IDs: `open_meteo.single_run.icon_d2`

## Overall raw NWP error

- MAE: `0.9790588235294119`
- RMSE: `1.3927543594914191`
- bias: `-0.5461176470588235`

## By issue hour UTC

| value | rows | mae_model_tmax | rmse_model_tmax | bias_model_tmax |
| --- | --- | --- | --- | --- |
| 0 | 122 | 1.0688524590163935 | 1.5229931702805863 | -0.33770491803278685 |
| 3 | 121 | 1.118181818181818 | 1.5751426539095719 | -0.5942148760330578 |
| 6 | 121 | 1.118181818181818 | 1.5751426539095719 | -0.5942148760330578 |
| 9 | 121 | 1.0099173553719007 | 1.3998819312669204 | -0.5801652892561983 |
| 12 | 121 | 1.0099173553719007 | 1.3998819312669204 | -0.5801652892561983 |
| 15 | 122 | 0.7655737704918032 | 1.0923729621903822 | -0.5688524590163933 |
| 18 | 122 | 0.7655737704918032 | 1.0923729621903822 | -0.5688524590163933 |

## By month

| value | rows | mae_model_tmax | rmse_model_tmax | bias_model_tmax |
| --- | --- | --- | --- | --- |
| 9 | 210 | 0.7842857142857141 | 1.1008005744738254 | -0.3957142857142856 |
| 10 | 213 | 0.9586854460093897 | 1.3522368878783655 | -0.4516431924882628 |
| 11 | 210 | 0.9566666666666667 | 1.2424055008617068 | -0.7061904761904761 |
| 12 | 217 | 1.2092165898617513 | 1.7711200861169762 | -0.6294930875576038 |

## Interpretation

NWP rows are now available for NWP-aware experiments, but the active production model is not automatically promoted.
Promotion should wait for a direct probabilistic comparison and at least one clean validation slice where NWP is present in both train and test.