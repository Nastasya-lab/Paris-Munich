# Safe blended shadow candidate

## Purpose

`blended_shadow_candidate_v1` is a shadow-only promotion candidate. It does not
change the operational forecast returned by the production champion.

The candidate exists to test a safer path toward future promotion:

- preserve the smooth probability shape of the champion and phase-aware shadow;
- use the ML remaining-upside model as a contextual signal only;
- never inject the jagged raw ML integer-bin probabilities into the candidate;
- reduce trust when runtime sources are unverified or data is stale;
- collect independent forward outcomes before any production decision.

## Blend logic

The candidate mixes:

```text
blended = (1 - weight) * production_champion + weight * shadow_seasonal_intraday
```

The initial weight is phase-aware:

| Phase | Initial phase-shadow weight |
| --- | ---: |
| `morning_prior` | 15% |
| `midday_update` | 35% |
| `late_nowcast` | 55% |

The weight can rise to 75% only for a late sharp temperature drop when the
phase-aware model and ML survival signal agree that further upside is unlikely.
High disagreement, stale METAR/NWP data, and source mismatch cap or discount the
weight. Unknown or forbidden runtime sources disable the blend.

## ML safety boundary

The ML shadow can influence the blend strength through:

- `probability_peak_already_passed`;
- `probability_upside_ge_1c`;
- `probability_upside_ge_2c`;
- `probability_upside_ge_3c`.

Its raw integer-bin distribution is deliberately excluded from the blend. This
keeps the blended candidate smooth while the ordinal ML model and its contextual
calibration continue to mature.

## Evaluation

Every logged forecast stores a `shadow_safe_blend` variant. Outcome refresh
scores it with the same NLL, CRPS, Brier, interval coverage, and point-error
metrics as the champion and other shadow models.

`safe_blend_promotion_gate` is advisory. Promotion remains blocked until enough
independent completed-day outcomes have accumulated and a manual review confirms
that the candidate improves probabilistic accuracy without creating false late
upside.
