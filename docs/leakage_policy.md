# Leakage policy

Every feature row is keyed by `airport_icao + issue_time_utc + target_date_local`.

Allowed feature records must satisfy `knowledge_time_utc <= issue_time_utc`.

Specific checks:

- METAR `observation_time_utc <= issue_time_utc`;
- TAF `issue_time_utc <= issue_time_utc`;
- NWP `model_availability_time_utc <= issue_time_utc`;
- target columns are absent from feature matrices;
- target day uses airport local timezone, not UTC day;
- DWD final truth is used only after the event for labels and scoring.

Violation raises `LeakageError`.
