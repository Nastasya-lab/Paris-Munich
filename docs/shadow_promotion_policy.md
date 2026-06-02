# Shadow promotion policy

## Purpose

The phase-aware intraday challenger is evaluated in shadow mode before it can be
considered for production. The gate is intentionally conservative: it can mark a
candidate as eligible for manual review, but it never switches the production
champion automatically.

## Candidate

Current candidate:

- Forecast variant: `shadow_seasonal_intraday`
- Variant version: `phase_aware_intraday_challenger_v3`
- Additional ML candidate: `shadow_intraday_ml`
- Additional ML version: `intraday_ml_core_challenger_v1`
- Production champion: `production_champion`

Older shadow rows remain in the logs, but the gate filters to the current
candidate version when enough versioned rows exist.

## Required evidence

The gate requires paired champion/shadow forecasts:

- At least 30 paired forecasts.
- At least 14 distinct target days.
- At least 5 morning pairs before 11:00 local time.
- At least 5 late-day pairs from 16:00 local time.

These limits are deliberately modest for MVP monitoring. A real production
promotion should use a larger sample, especially across multiple weather regimes.

## Quality checks

The shadow candidate must not be worse than the champion on core probabilistic
metrics:

- Mean CRPS not worse.
- Mean NLL not worse.
- Mean expected-value MAE not materially worse.
- Late-day false-upside probability not worse.
- Morning CRPS not materially worse.
- 80% interval coverage not materially worse.

False-upside probability is the probability mass assigned to integer bins above
the actual rounded Tmax bin. It is especially important for late-day forecasts,
where the bot should not keep meaningful probability on impossible or highly
unlikely new highs after the daily peak has probably passed.

## Possible statuses

- `pending`: no usable variant monitoring data yet.
- `continue_shadow_monitoring`: sample size is not sufficient.
- `do_not_promote_quality_gate_failed`: sample is sufficient, but quality checks
  failed.
- `eligible_for_manual_promotion_review`: sample and quality gates passed; a
  human review is still required before changing production.

## Reports

Generated reports:

- `data/reports/shadow_promotion_gate.json`
- `docs/shadow_promotion_gate.md`
- The gate is also embedded in `docs/outcome_analysis.md` and monitoring
  summaries.

The intraday ML candidate is evaluated with the same paired-gate logic, but it
is intentionally treated as a separate challenger because it is currently
preliminary and not fully calibrated.
