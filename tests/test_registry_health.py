import joblib

from weather_tmax_bot.models.model_registry import promote_model, register_artifact
from weather_tmax_bot.models.registry_health import registry_health


def test_registry_health_passes_for_registered_active_model(tmp_path):
    model_path = tmp_path / "candidate.joblib"
    calibrator_path = tmp_path / "candidate.calibrator.joblib"
    joblib.dump({"model": True}, model_path)
    joblib.dump({"calibrator": True}, calibrator_path)
    register_artifact(version="candidate", artifact_type="model", path=model_path, model_dir=tmp_path)
    register_artifact(
        version="candidate.calibrator",
        artifact_type="calibrator",
        path=calibrator_path,
        model_dir=tmp_path,
    )
    promote_model(
        model_version="candidate",
        calibrator_version="candidate.calibrator",
        model_dir=tmp_path,
    )

    health = registry_health(model_dir=tmp_path, fallback_model_path=tmp_path / "missing.joblib")

    assert health["passed"]
    assert health["checks"]["active_version_registered"]


def test_registry_health_uses_fallback_when_no_active_model(tmp_path):
    fallback = tmp_path / "quantile_mvp.joblib"
    joblib.dump({"fallback": True}, fallback)

    health = registry_health(model_dir=tmp_path, fallback_model_path=fallback)

    assert health["passed"]
    assert health["model_path"] == str(fallback)
