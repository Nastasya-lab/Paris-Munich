from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts import railway_entrypoint


def _load_embedded_scheduler():
    path = Path(__file__).resolve().parents[1] / "scripts" / "56_embedded_scheduler.py"
    spec = importlib.util.spec_from_file_location("embedded_scheduler", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


embedded_scheduler = _load_embedded_scheduler()


def test_embedded_scheduler_runs_metar_every_minute() -> None:
    jobs = embedded_scheduler.due_jobs(datetime(2026, 6, 18, 12, 5, tzinfo=timezone.utc))

    assert [job.name for job in jobs] == ["metar_event_all_once"]


def test_embedded_scheduler_runs_forecast_on_configured_slots() -> None:
    jobs = embedded_scheduler.due_jobs(datetime(2026, 6, 18, 13, 30, tzinfo=timezone.utc))

    assert [job.name for job in jobs] == ["metar_event_all_once", "forecast_all"]


def test_embedded_scheduler_runs_outcome_and_daily_report() -> None:
    outcome_jobs = embedded_scheduler.due_jobs(datetime(2026, 6, 18, 6, 30, tzinfo=timezone.utc))
    report_jobs = embedded_scheduler.due_jobs(datetime(2026, 6, 18, 18, 15, tzinfo=timezone.utc))
    live_baseline_jobs = embedded_scheduler.due_jobs(datetime(2026, 6, 18, 18, 20, tzinfo=timezone.utc))

    assert "outcome" in [job.name for job in outcome_jobs]
    assert "daily_report" in [job.name for job in report_jobs]
    assert "live_baseline" in [job.name for job in live_baseline_jobs]


def test_railway_entrypoint_defaults_to_api_for_main_service() -> None:
    assert railway_entrypoint.resolve_job("Paris-Munich") == "api"
