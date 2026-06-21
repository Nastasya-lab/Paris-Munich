import pandas as pd

from weather_tmax_bot.evaluation.live_baseline import (
    build_live_baseline_report,
    monitoring_from_telegram_exports,
    write_live_baseline_report,
)


def _monitoring_frame():
    return pd.DataFrame(
        {
            "forecast_id": ["m1", "m2", "p1", "p2", "p3"],
            "airport": ["EDDM", "EDDM", "LFPB", "LFPB", "LFPB"],
            "model_version": ["munich", "munich", "paris", "paris", "paris"],
            "target_date_local": ["2026-06-18", "2026-06-19", "2026-06-18", "2026-06-19", "2026-06-20"],
            "issue_time_utc": [
                "2026-06-18T08:30:00+00:00",
                "2026-06-19T15:00:00+00:00",
                "2026-06-18T08:15:00+00:00",
                "2026-06-19T13:30:00+00:00",
                "2026-06-20T06:30:00+00:00",
            ],
            "actual_tmax_c": [30.0, 33.0, 36.0, 36.0, 25.0],
            "expected_tmax_c": [30.4, 35.7, 35.2, 36.4, 24.0],
            "median_tmax_c": [30.0, 35.0, 35.0, 36.0, 24.0],
            "most_likely_integer_c": [30, 35, 35, 36, 24],
            "error_expected_c": [0.4, 2.7, -0.8, 0.4, -1.0],
            "error_median_c": [0.0, 2.0, -1.0, 0.0, -1.0],
            "nll": [1.0, 2.0, 1.2, 1.1, 1.3],
            "crps": [0.1, 0.2, 0.1, 0.1, 0.2],
            "probability_actual_integer_bin": [0.4, 0.05, 0.3, 0.45, 0.2],
            "coverage_80": [True, False, True, True, True],
            "interval_80_width_c": [2.0, 3.0, 4.0, 3.0, 3.0],
        }
    )


def test_build_live_baseline_filters_local_window_and_summarizes():
    report = build_live_baseline_report(_monitoring_frame())

    assert report.metadata["evaluated_rows"] == 4
    assert set(report.overall["airport"]) == {"EDDM", "LFPB"}

    munich = report.overall[report.overall["airport"] == "EDDM"].iloc[0]
    assert munich["rows"] == 2
    assert munich["mae_expected"] == 1.55
    assert munich["large_error_gt_2c_rate"] == 0.5

    paris = report.overall[report.overall["airport"] == "LFPB"].iloc[0]
    assert paris["rows"] == 2
    assert paris["bias_expected"] == -0.2
    assert paris["coverage_80"] == 1.0

    assert "recommendation" in report.bias_audit.columns
    assert set(report.by_phase["phase"]) == {"morning_10_12", "afternoon_14_16", "late_16_17"}
    assert "mae_expected_gt_1_5c" in set(report.alerts["reason"])
    assert "forecast_error_gt_2c" in set(report.alerts["reason"])


def test_write_live_baseline_report(tmp_path):
    report = build_live_baseline_report(_monitoring_frame())
    paths = write_live_baseline_report(report, output_dir=tmp_path)

    assert (tmp_path / "live_baseline_10_17_overall.csv").exists()
    assert (tmp_path / "live_baseline_10_17.md").exists()
    assert set(paths) == {
        "rows",
        "overall",
        "by_phase",
        "by_hour",
        "by_day",
        "rolling",
        "alerts",
        "bias_audit",
        "metadata",
        "markdown",
    }


def test_monitoring_from_telegram_exports(tmp_path):
    export = tmp_path / "munich.json"
    export.write_text(
        """
{
  "messages": [
    {
      "id": 1,
      "text": "Прогноз готов: EDDM\\nДата: 2026-06-18\\nВыпуск: 18.06.2026 10:30 по Мюнхену\\nТемпературный прогноз\\nОжидаемый максимум: 30.4 °C\\nМедиана: 30.0 °C\\nСамая вероятная корзина: 30 °C\\nИнтервал 80%: 29.0...31.0 °C\\nВероятности по градусам\\n+30 °C: 40.0%\\n+31 °C: 30.0%\\nMETAR-сигнал\\nНаблюдаемый максимум: 30.0 °C\\nТехнические сведения\\nМодель: nwp_residual_icon_d2_20260531"
    },
    {
      "id": 2,
      "text": "METAR-обновление прогноза: EDDM\\nДата: 2026-06-18\\nВыпуск: 18.06.2026 16:30 по Мюнхену\\nЧто изменилось\\nОжидаемый максимум: 30.1 °C\\nMETAR-сигнал\\nНаблюдаемый максимум: 31.0 °C\\nРаспределение по градусам\\n+30 °C: 20.0%\\n+31 °C: 60.0%"
    }
  ]
}
""",
        encoding="utf-8",
    )

    frame = monitoring_from_telegram_exports({"EDDM": export})

    assert len(frame) == 2
    assert frame["actual_tmax_c"].tolist() == [31.0, 31.0]
    assert frame.iloc[0]["error_expected_c"] == -0.6000000000000014
    assert frame.iloc[0]["coverage_80"] == True
