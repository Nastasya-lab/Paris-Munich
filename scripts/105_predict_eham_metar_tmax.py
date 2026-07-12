from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def main() -> None:
    predictor = _load_lfpb_predictor()
    _configure_for_eham(predictor)
    if "--airport" not in sys.argv:
        sys.argv.extend(["--airport", "EHAM"])
    if "--model-path" not in sys.argv:
        sys.argv.extend(["--model-path", "data/models/eham_metar_tmax_icon_d2_v1.joblib"])
    if "--metadata-path" not in sys.argv:
        sys.argv.extend(["--metadata-path", "data/models/eham_metar_tmax_icon_d2_v1.metadata.json"])
    if "--report-path" not in sys.argv:
        sys.argv.extend(["--report-path", "data/reports/latest_eham_icon_d2_metar_tmax_prediction.json"])
    if "--hf-icon-eu-shadow" not in sys.argv and "--no-hf-icon-eu-shadow" not in sys.argv:
        sys.argv.append("--no-hf-icon-eu-shadow")
    predictor.main()


def _load_lfpb_predictor():
    path = Path(__file__).with_name("48_predict_lfpb_metar_tmax.py")
    spec = importlib.util.spec_from_file_location("weather_tmax_lfpb_predictor_for_eham", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load predictor from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure_for_eham(module) -> None:
    module.AIRPORT = "EHAM"
    module.TIMEZONE = "Europe/Amsterdam"
    module.LATITUDE = 52.3086
    module.LONGITUDE = 4.7639
    module.LIVE_NWP_PATH = Path("data/forecasts/open_meteo_archive_EHAM.parquet")
    module.LIVE_ICON_EU_NWP_PATH = Path("data/forecasts/open_meteo_archive_icon_eu_EHAM.parquet")
    module.ENHANCED_ICON_NWP_PATH = Path("data/forecasts/open_meteo_single_runs_icon_d2_EHAM_enhanced.parquet")
    module.HISTORICAL_NWP_PATH = Path("data/forecasts/open_meteo_single_runs_icon_d2_EHAM.parquet")
    module.ECMWF_NWP_PATH = Path("data/forecasts/open_meteo_single_runs_ecmwf_ifs_EHAM.parquet")
    module.AROME_NWP_PATHS = []
    module.SURVIVAL_DATASET_PATH = Path("data/processed/metar_upside_dataset_EHAM_icon_d2.parquet")
    module.SPATIAL_CANDIDATE_MODEL_PATH = Path("data/models/eham_metar_tmax_icon_d2_spatial_wind_advection_v1.joblib")
    module.SPATIAL_CANDIDATE_METADATA_PATH = Path("data/models/eham_metar_tmax_icon_d2_spatial_wind_advection_v1.metadata.json")
    module.HAZARD_SHADOW_MODEL_PATH = Path("data/models/eham_discrete_hazard_spatial_wind_advection_shadow_v1.joblib")
    module.HAZARD_SHADOW_METADATA_PATH = Path("data/models/eham_discrete_hazard_spatial_wind_advection_shadow_v1.metadata.json")
    module.HAZARD_SHADOW_VARIANT = "shadow_discrete_hazard"
    module.LONG_HISTORY_SHADOW_MODEL_PATH = Path(
        "data/models/eham_long_history_candidates/eham_icon_d2_d1_long_all_history_unimodal_candidate_v1.joblib"
    )
    module.LONG_HISTORY_SHADOW_METADATA_PATH = Path(
        "data/models/eham_long_history_candidates/eham_icon_d2_d1_long_all_history_unimodal_candidate_v1.metadata.json"
    )
    module.LONG_HISTORY_SHADOW_ENABLED = True
    module.UNIMODAL_SHADOW_VERSION = "eham_pmf_temperature_unimodal_production_v1"
    module.DEFAULT_SPATIAL_STATIONS = ["EHRD", "EHLE"]
    module.DEFAULT_ADVECTION_STATIONS = ["EHAM", "EHRD", "EHLE"]

    def _no_hf_block(*_args, **_kwargs):
        return []

    module._format_lfpb_hf_icon_eu_compact_block = _no_hf_block
    module._format_lfpb_short_summary = _eham_short_summary

    original_compact = module._format_lfpb_compact_message

    def _eham_compact_message(payload: dict) -> str:
        text = original_compact(payload)
        return (
            text.replace("LFPB Paris Tmax forecast", "EHAM Amsterdam Tmax forecast")
            .replace("РџР°СЂРёР¶Сѓ", "Amsterdam")
            .replace("по Парижу", "по Amsterdam")
        )

    module._format_lfpb_compact_message = _eham_compact_message


def _eham_short_summary(payload: dict) -> list[str]:
    forecast = payload.get("forecast") or {}
    wind = ((payload.get("spatial_candidate") or {}).get("forecast") or {})
    hazard = ((payload.get("hazard_shadow_candidate") or {}).get("forecast") or {})
    long_history = ((payload.get("long_history_shadow_candidate") or {}).get("forecast") or {})
    signal = payload.get("metar_signal") or {}
    lines = [
        "<b>Кратко</b>",
        f"Рабочий прогноз: {_format_bin(forecast.get('most_likely_integer_c'))}",
        f"Wind/advection: {_format_bin(wind.get('most_likely_integer_c'))}",
        f"Discrete hazard: {_format_bin(hazard.get('most_likely_integer_c'))}",
        f"Long history: {_format_bin(long_history.get('most_likely_integer_c'))}",
    ]
    if signal.get("latest_metar_temp_c") is not None:
        lines.append(f"Последний METAR: {float(signal.get('latest_metar_temp_c')):.1f} °C")
    return lines


def _format_bin(value) -> str:
    if value is None:
        return "нет данных"
    return f"{int(value):+d} °C"


if __name__ == "__main__":
    main()
