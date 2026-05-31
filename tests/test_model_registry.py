from pathlib import Path

import joblib

from weather_tmax_bot.models.model_registry import (
    load_registry,
    promote_model,
    register_artifact,
    resolve_active_artifacts,
    save_model,
)


def test_save_model_registers_and_resolves_active_artifacts(tmp_path):
    model_path = save_model({"kind": "model"}, {"model_version": "candidate"}, model_dir=tmp_path)
    calibrator_path = save_model(
        {"kind": "calibrator"},
        {"model_version": "candidate.calibrator"},
        model_dir=tmp_path,
    )
    register_artifact(
        version="candidate",
        artifact_type="model",
        path=model_path,
        metadata_path=Path(tmp_path) / "candidate.metadata.json",
        model_dir=tmp_path,
    )
    register_artifact(
        version="candidate.calibrator",
        artifact_type="calibrator",
        path=calibrator_path,
        metadata_path=Path(tmp_path) / "candidate.calibrator.metadata.json",
        model_dir=tmp_path,
    )
    promote_model(
        model_version="candidate",
        calibrator_version="candidate.calibrator",
        reason="test",
        model_dir=tmp_path,
    )

    active = resolve_active_artifacts(model_dir=tmp_path, fallback_model_path=tmp_path / "missing.joblib")

    assert active["model_path"] == model_path
    assert active["calibrator_path"] == calibrator_path
    assert joblib.load(active["model_path"]) == {"kind": "model"}
    assert load_registry(tmp_path)["active_model_version"] == "candidate"


def test_resolve_active_artifacts_falls_back_to_quantile_mvp(tmp_path):
    fallback = tmp_path / "quantile_mvp.joblib"
    joblib.dump({"kind": "fallback"}, fallback)

    active = resolve_active_artifacts(model_dir=tmp_path, fallback_model_path=fallback)

    assert active["model_path"] == fallback
    assert active["calibrator_path"] is None


def test_resolve_active_artifacts_normalizes_windows_paths(tmp_path):
    model_path = tmp_path / "data" / "models" / "candidate.joblib"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"x")
    registry = {
        "active_model_version": "candidate",
        "active_calibrator_version": None,
        "entries": [
            {
                "version": "candidate",
                "artifact_type": "model",
                "path": str(model_path).replace("/", "\\"),
            }
        ],
    }
    (tmp_path / "model_registry.json").write_text(__import__("json").dumps(registry), encoding="utf-8")

    active = resolve_active_artifacts(model_dir=tmp_path, fallback_model_path=tmp_path / "missing.joblib")

    assert active["model_path"] == model_path
