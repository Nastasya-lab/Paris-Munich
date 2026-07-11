from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STATE_PATH = Path("data/logs/forecast_input_state.json")


def record_update_source(*, airport: str, trigger: str, payload: dict[str, Any]) -> dict[str, Any]:
    current = _snapshot(payload)
    state = _load_state()
    previous = (state.get("airports") or {}).get(airport)
    metar_changed = _changed(previous, current, "metar")
    nwp_changed = _changed(previous, current, "nwp")
    result = {
        "trigger": trigger,
        "metar_changed": metar_changed,
        "nwp_changed": nwp_changed,
        "previous": previous,
        "current": current,
    }
    state.setdefault("airports", {})[airport] = current
    result["state_persisted"] = _save_state(state)
    return result


def format_update_source_lines(update: dict[str, Any] | None) -> list[str]:
    if not update:
        return []
    current = update.get("current") or {}
    previous = update.get("previous") or {}
    trigger = {"scheduled_forecast": "плановый forecast", "new_metar": "новый METAR"}.get(
        update.get("trigger"), str(update.get("trigger") or "не указан")
    )
    lines = ["<b>Источник обновления</b>", f"Триггер: {trigger}"]
    if not previous:
        lines.append("Первый input snapshot для этого аэропорта.")
        return lines
    if update.get("metar_changed"):
        lines.append(f"METAR: {_metar_label(previous.get('metar'))} → {_metar_label(current.get('metar'))}")
    else:
        lines.append(f"METAR: без изменений, {_metar_label(current.get('metar'))}")
    if update.get("nwp_changed"):
        lines.append(f"ICON/NWP Tmax: {_nwp_label(previous.get('nwp'))} → {_nwp_label(current.get('nwp'))}")
    else:
        lines.append(f"ICON/NWP: без изменений, {_nwp_label(current.get('nwp'))}")
    return lines


def _snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    signal = payload.get("metar_signal") or {}
    latest = payload.get("latest_metar_record") or {}
    lineage = payload.get("data_lineage") or {}
    return {
        "metar": {
            "time_utc": signal.get("latest_metar_time_utc") or latest.get("observation_time_utc"),
            "temperature_c": signal.get("latest_metar_temp_c") if signal else latest.get("temperature_c"),
        },
        "nwp": {
            "source": lineage.get("latest_nwp_source_id") or lineage.get("nwp_source_id"),
            "tmax_c": lineage.get("model_tmax_c") or lineage.get("nwp_tmax_c"),
            "knowledge_time_utc": lineage.get("max_nwp_knowledge_time_utc"),
        },
    }


def _changed(previous: dict[str, Any] | None, current: dict[str, Any], key: str) -> bool:
    return bool(previous) and (previous.get(key) or {}) != (current.get(key) or {})


def _load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"airports": {}}
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"airports": {}}
    return state if isinstance(state, dict) else {"airports": {}}


def _save_state(state: dict[str, Any]) -> bool:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    except OSError:
        return False
    return True


def _metar_label(value: dict[str, Any] | None) -> str:
    value = value or {}
    temperature = value.get("temperature_c")
    time_utc = value.get("time_utc") or "нет времени"
    return f"{time_utc}, {float(temperature):.1f} °C" if temperature is not None else str(time_utc)


def _nwp_label(value: dict[str, Any] | None) -> str:
    value = value or {}
    source = value.get("source") or "unknown"
    tmax = value.get("tmax_c")
    return f"{source}, {float(tmax):.1f} °C" if tmax is not None else str(source)
