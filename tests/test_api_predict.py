from fastapi.testclient import TestClient

from weather_tmax_bot.bot.api import app


def test_api_predict_returns_distribution(tmp_path, monkeypatch):
    monkeypatch.setenv("WEATHER_TMAX_FORECAST_LOG_PATH", str(tmp_path / "forecast_log.jsonl"))
    response = TestClient(app).get("/predict", params={"airport": "EDDM", "target_date": "2026-07-15"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["airport"] == "EDDM"
    assert abs(sum(payload["probabilities_by_integer_c"].values()) - 1.0) < 1e-6
    assert payload["forecast_id"]
    assert "truth_source" in payload["data_lineage"]
    assert "data_freshness" in payload
    assert "forecast_quality" in payload
    assert "source_compatibility" in payload
    assert "forecast_acceptance" in payload


def test_health_and_model_info():
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    scheduler_health = client.post("/scheduler-healthcheck", params={"notify_on_failure": False}).json()
    assert "ready_for_forward_ops" in scheduler_health
    payload = client.get("/model-info").json()
    assert "models" in payload
    assert "active_model" in payload
    assert "model_exists" in payload["active_model"]
    monitoring = client.get("/monitoring-summary").json()
    assert "training_rows" in monitoring
    assert "active_model" in monitoring
    assert "archive_freshness" in monitoring
    registry = client.get("/registry-health").json()
    assert "passed" in registry
    freshness = client.get("/data-freshness-health").json()
    assert "freshness" in freshness
    operational = client.get("/operational-monitoring").json()
    assert "forecast_inventory" in operational
    first_analysis = client.get("/first-analysis").json()
    assert "readiness" in first_analysis
    prepared = client.post(
        "/prepare-operational-run",
        params={"airport": "EDDM", "target_date": "2026-05-29", "skip_awc": True, "skip_nwp": True},
    ).json()
    assert prepared["airport"] == "EDDM"
    pending_truth = client.get("/pending-truth", params={"as_of_date": "2026-05-29"}).json()
    assert "pending_rows" in pending_truth
    pending_cron = client.post("/pending-truth-cron", params={"fetch": False, "as_of_date": "2026-05-29"}).json()
    assert "status" in pending_cron
    cycle = client.post(
        "/operational-cycle",
        params={
            "airport": "EDDM",
            "target_date": "2026-05-29",
            "issue_time": "2026-05-28T20:30:00Z",
            "auto_refresh": False,
            "log": False,
            "update_reports": False,
        },
    ).json()
    assert cycle["airport"] == "EDDM"
    assert "forecast_acceptance" in cycle
    assert "report_summary" in cycle
    operational_prediction = client.post(
        "/predict-operational",
        params={
            "airport": "EDDM",
            "target_date": "2026-05-29",
            "issue_time": "2026-05-28T20:30:00Z",
            "auto_refresh": True,
            "refresh_awc": False,
            "refresh_nwp": False,
            "log": False,
        },
    ).json()
    assert operational_prediction["airport"] == "EDDM"
    assert "refresh_summary" in operational_prediction
    assert "forecast_quality" in operational_prediction
    assert "source_compatibility" in operational_prediction
    assert "forecast_acceptance" in operational_prediction


def test_operational_mutation_endpoints_can_require_api_key(monkeypatch):
    monkeypatch.setenv("OPERATIONAL_API_KEY", "secret")
    client = TestClient(app)

    rejected = client.post(
        "/operational-cycle",
        params={
            "airport": "EDDM",
            "target_date": "2026-05-29",
            "issue_time": "2026-05-28T20:30:00Z",
            "auto_refresh": False,
            "log": False,
            "update_reports": False,
            "notify": False,
        },
    )
    assert rejected.status_code == 401

    rejected_health = client.post("/scheduler-healthcheck", params={"notify_on_failure": False})
    assert rejected_health.status_code == 401

    accepted = client.post(
        "/operational-cycle",
        params={
            "airport": "EDDM",
            "target_date": "2026-05-29",
            "issue_time": "2026-05-28T20:30:00Z",
            "auto_refresh": False,
            "log": False,
            "update_reports": False,
            "notify": False,
        },
        headers={"X-API-Key": "secret"},
    )
    assert accepted.status_code == 200
