# Station selection for LFPB

Airport: `LFPB` Paris-Le Bourget Airport.

Selected truth station: Meteo-France station `95088001` `LE BOURGET`, WMO/SYNOP id `07150`.

Station coordinates: 48.967333, 2.427667, elevation 49 m.

LFPB reference coordinates used by the project: 48.969444, 2.441389, elevation about 67 m.

Distance to LFPB reference coordinates: about 1.03 km.

## Decision

Use Meteo-France official climatological observations as the LFPB truth layer:

- Primary target label: `meteofrance.base.daily.air_temperature.95088001`
- Intraday historical observation bridge: `meteofrance.base.hourly.air_temperature.95088001`
- Operational cross-check source: `meteofrance.synop.07150`

The selected station is effectively the airport station and is much closer than the alternatives checked around Paris.

Feasibility check: see [lfpb_truth_feasibility.md](lfpb_truth_feasibility.md).

## Verified public data sources

Meteo-France BASE hourly data:

- Dataset: `Donnees climatologiques de base - horaires`
- Catalog URL: `https://www.data.gouv.fr/api/1/datasets/donnees-climatologiques-de-base-horaires/`
- Department files checked:
- `https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/HOR/H_95_previous-2020-2024.csv.gz`
- `https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/HOR/H_95_latest-2025-2026.csv.gz`
- Checked station rows for `95088001 LE BOURGET`: 2020-01-01 00 UTC through 2026-06-06 04 UTC.
- Relevant fields observed in the file: `T`, `QT`, `TN`, `QTN`, `HTN`, `TX`, `QTX`, `HTX`.

Meteo-France BASE daily data:

- Dataset: `Donnees climatologiques de base - quotidiennes`
- Catalog URL: `https://www.data.gouv.fr/api/1/datasets/donnees-climatologiques-de-base-quotidiennes/`
- Department files checked:
- `https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/QUOT/Q_95_previous-1950-2024_RR-T-Vent.csv.gz`
- `https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/QUOT/Q_95_latest-2025-2026_RR-T-Vent.csv.gz`
- Checked station rows for `95088001 LE BOURGET`: 1950-01-01 through 2026-06-04.
- Relevant fields observed in the file: `TN`, `QTN`, `HTN`, `TX`, `QTX`, `HTX`, `TM`, `QTM`.

Meteo-France SYNOP OMM observations:

- Dataset: `Archive Synop OMM`
- Catalog URL: `https://www.data.gouv.fr/api/1/datasets/observations-synop/`
- Yearly files checked: `synop_2020.csv.gz` through `synop_2026.csv.gz`.
- Station `07150 LE BOURGET` appears in checked SYNOP files from 2025-09-05 onward.
- This is useful for operational cross-checking but too short to support the full 2020-2025 training/backtest history alone.

## Nearby alternatives checked

Distances are approximate, computed against LFPB reference coordinates.

- `95088001` `LE BOURGET`: 1.03 km, Meteo-France hourly and daily, selected.
- `93059001` `PIERREFITTE`: 4.45 km, Meteo-France daily only in checked files.
- `95585001` `SARCELLES`: 5.20 km, Meteo-France daily only in checked files.
- `93005001` `AULNAY`: 5.33 km, Meteo-France daily only in checked files.
- `95492001` `LE PLESSIS GASSOT`: 8.26 km, Meteo-France hourly and daily.
- `95527001` `ROISSY`: 9.43 km, Meteo-France hourly and daily.
- `75114001` / `75114007` `PARIS-MONTSOURIS`: 17.48 km, Meteo-France hourly and daily.

The alternatives are not preferred because the exact Le Bourget station is available in Meteo-France BASE/HOR and BASE/QUOT.

## Target construction recommendation

For LFPB, the closest match to the EDDM truth-layer philosophy is:

1. Use Meteo-France daily `TX` for station `95088001` as the final supervised learning label.
2. Use Meteo-France hourly `T/TX/HTX` for historical intraday features, observed-max-so-far proxies, and Tmax timing analysis.
3. Use `Europe/Paris` local days for model rows; keep a separate audit comparing local-day hourly reconstruction to daily `TX`.
4. If daily `TX` is missing or has bad quality, mark the target row as unavailable rather than silently replacing it with METAR.
5. Do not use SYNOP/METAR as the sole truth target for historical model evaluation.
6. Use Meteo-France 6-minute `RR` as a precipitation-regime feature source, not as temperature truth.

Before writing the final parser, confirm the exact time semantics of `AAAAMMJJHH`, `HTX`, and quality flags against the Meteo-France PDF descriptif:

- `CLIMATOLOGIE_Donnees_horaires_descriptif.pdf`
- `CLIMATOLOGIE_Donnees_quotidiennes_descriptif.pdf`

## Current limitations

- Meteo-France BASE/HOR is hourly, not 10-minute DWD-style data. It is still official and station-specific, but the temporal resolution differs from EDDM.
- Meteo-France BASE/MIN 6-minute files for department 95 include `RR/QRR` for `95088001 LE BOURGET`, but no temperature columns in the checked 2010-2019, 2020-2024, and 2025-2026 files.
- SYNOP `07150` is not sufficient for the full 2020-2025 history based on checked files; it appears from 2025-09-05 onward.
- LFPB NWP source selection must be handled separately. ICON-D2 is Germany-focused; for Paris we should prefer honest issued runs from Meteo-France AROME/ARPEGE if available, otherwise Open-Meteo single/previous runs with explicit run identity and no seamless historical-weather substitution.
