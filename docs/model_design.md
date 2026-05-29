# Model design

Main model: quantile regression over many quantiles, then monotone CDF and integer-bin discretization.

Steps:

1. Predict quantiles.
2. Fix quantile crossing with monotone accumulation.
3. Interpolate CDF.
4. Compute `P(bin k) = F(k + 0.5) - F(k - 0.5)`.
5. For same-day forecasts, truncate probability below observed maximum so far.
6. Calibrate CDF on validation data using PIT/isotonic calibration.

Climatology is implemented as the first honest baseline and fallback.

Current MVP calibration:

- `DiscreteSpreadCalibrator` is fitted on a validation-style holdout.
- It convolves integer-bin probabilities with a fitted Gaussian-like kernel in bin space.
- This is not a fixed sigma: sigma is selected on validation coverage/NLL.
- `IntegerCDFIsotonicCalibrator` is implemented and reported as an experimental variant.
- Current holdout/rolling reports show isotonic CDF underperforms spread calibration on NLL and interval coverage, so production prediction continues to use spread calibration.
