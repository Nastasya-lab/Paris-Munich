# MostlyRight SDK assessment

Reviewed local clone of `https://github.com/mostlyrightmd/mostlyright-sdk` at `ba59d47...` from `main` on 2026-05-28.

## Applicable parts

MostlyRight is useful as an engineering reference for:

- explicit source identity and source mismatch failures;
- temporal-safety primitives such as `KnowledgeView` / `LeakageDetector`;
- local parquet cache layout with versioning and file locks;
- AWC/IEM/GHCNh/NWS CLI observation ingestion patterns;
- Open-Meteo previous/single-run source discipline and leakage regression tests;
- immutable-ish observation ledgers and provenance schemas.

## Direct reuse

Direct reuse is optional. `MostlyRightAdapter` is a thin wrapper and the project runs without the SDK installed.

Safe direct-use candidates, after source coverage is verified:

- METAR/ASOS-like ingestion and cache where international station coverage is proven;
- source metadata normalization;
- Open-Meteo issued-run metadata patterns;
- cache invalidation/version patterns.

## Use as ideas, implement internally

The following are implemented internally because they are load-bearing for this project:

- DWD truth source selection and station metadata;
- DWD 10-minute target construction;
- as-of feature filtering;
- leakage detector;
- source registry;
- forecast log schema;
- probabilistic quantile/CDF/integer-bin model.

## Sources for EDDM

MostlyRight appears strongest around AWC/IEM/ASOS/GHCNh/NWS CLI and Open-Meteo forecast sources. It may help with IEM/AWC aviation observations if EDDM coverage is confirmed, but it is not treated as the authority for EDDM truth.

## International limitations

MostlyRight is weather-market oriented and much of its precision/source logic is US-station oriented. For EDDM, DWD 10-minute observations, DWD station metadata, European aviation TAF coverage, ICON-D2/ICON-EU, MOSMIX, and ECMWF licensing are outside what we can assume from the SDK.

## Coverage checks

- TAF: not assumed covered. This project includes its own TAF parser and source IDs.
- DWD 10-minute observations: not covered as the project truth layer.
- Forecast-as-issued NWP: Open-Meteo previous/single-run ideas are relevant, but DWD ICON/ECMWF issued archives require internal adapters/archiver.
- KnowledgeView/leakage safety: useful pattern, but not used directly as the only guard. Internal tests enforce leakage rules.

## Dependencies

MostlyRight adds optional PyPI dependencies through `mostlyrightmd` / `mostlyrightmd-weather`. Core MVP dependencies do not require it.

## Independence requirement

These parts must work independently from MostlyRight:

- DWD station selection;
- DWD truth target;
- source registry;
- knowledge view;
- leakage detector;
- METAR/TAF fallback parsers;
- NWP operational archiver;
- quantile probabilistic model;
- calibration/backtest;
- CLI/API/forecast logging.

## Explicit non-goals

MostlyRight ML examples or point forecast examples are not the final model. A point model plus fixed Gaussian sigma is not accepted as production probability. Any MostlyRight-style point baseline must be calibrated through empirical residuals and compared against the main quantile/CDF model.
