import numpy as np

from weather_tmax_bot.models.calibration import CDFIsotonicCalibrator


def test_isotonic_calibrator_monotone():
    cal = CDFIsotonicCalibrator().fit(np.array([0.1, 0.4, 0.9]), np.array([0, 1, 1]))
    out = cal.transform(np.array([0.2, 0.5, 0.8]))
    assert np.all(np.diff(out) >= 0)
