# Railway deploy

This guide deploys the EDDM Tmax probabilistic forecasting MVP to Railway for autonomous forward operation.

## Required Railway resources

- One GitHub-connected Railway project.
- One persistent Volume mounted at `/app/data`.
- One always-on API service.
- Three cron services.

The project expects its writable state under `/app/data`. The Docker image also contains seed data in `/app/seed_data`; `scripts/railway_bootstrap.py` copies missing files into the mounted Volume on first start.

## API service

Use the default Railway service from this repository.

Start command:

```bash
python scripts/railway_bootstrap.py && python scripts/10_start_api.py
```

Environment variables:

```bash
WEATHER_TMAX_DATA_DIR=data
WEATHER_TMAX_SEED_DATA_DIR=seed_data
HOST=0.0.0.0
WEATHER_TMAX_ENABLE_TELEGRAM=1
TELEGRAM_BOT_TOKEN=<set in Railway Variables>
TELEGRAM_CHAT_ID=<set in Railway Variables>
OPERATIONAL_API_KEY=<random secret for cron callers>
```

Railway provides `PORT` automatically.

Health endpoints:

```text
GET /health
GET /model-info
GET /data-freshness-health
GET /first-analysis
GET /operational-monitoring
```

## Cron service: operational forecast

Recommended Railway setup: keep persistent state only on the `Munich` API service Volume. Cron services should call the API instead of writing their own local `data/`.

Create a Railway cron service using the same repo/image.

Command:

```bash
python scripts/33_call_api_job.py forecast
```

Recommended cron schedules in UTC:

```text
0 6 * * *
0 9 * * *
0 12 * * *
0 15 * * *
```

These issue hours are inside the MVP training schedule. You can also add `0 0 * * *`, `0 3 * * *`, and `0 18 * * *` if useful.

## Cron service: outcome update

Command:

```bash
python scripts/33_call_api_job.py outcome
```

Recommended schedule:

```text
30 6 * * *
```

This checks pending forecasts, downloads available DWD truth, rebuilds targets, scores completed forecasts, and refreshes reports.

## Cron service: scheduler healthcheck

Command:

```bash
python scripts/33_call_api_job.py health
```

Recommended schedule:

```text
15 * * * *
```

This asks the main API service to validate the model registry, data freshness, leakage audit, accepted forecasts, and pending truth status on the persistent Volume. It sends Telegram only when the system is blocked.

## Telegram notifications

Set these Railway variables on every service that should send notifications:

```bash
WEATHER_TMAX_ENABLE_TELEGRAM=1
TELEGRAM_BOT_TOKEN=<your bot token>
TELEGRAM_CHAT_ID=<your chat id>
```

Do not commit the token to GitHub.

Manual test command:

```bash
python scripts/railway_bootstrap.py && python scripts/32_send_telegram_test.py
```

Forecast and outcome cron jobs notify by default. Healthcheck only notifies failures by default; add `--notify-on-success` if you want hourly success messages.

Cron service variables:

```bash
MUNICH_API_BASE_URL=https://<your-munich-domain>
OPERATIONAL_API_KEY=<same value as API service>
```

## Manual Railway checks

Run these commands from a Railway shell or one-off job:

```bash
python scripts/28_check_launch_readiness.py --require-forward-ready
python scripts/18_check_data_freshness.py --no-fail-on-missing
python scripts/16_check_model_registry.py
python scripts/13_run_leakage_audit.py
python -m pytest -q
```

## Operational interpretation

`ready_for_forward_ops=true` means the service can issue and log forecasts.

`ready_for_outcome_monitoring=true` only becomes true after at least one logged forecast has a completed DWD truth outcome.

Do not expect outcome monitoring immediately after deployment; the system must first log forecasts, wait for local target days to finish, then fetch DWD truth.

## Volume warning

Mount the same Railway Volume at `/app/data` for all services and cron jobs. If each service gets a different Volume, forecast logs, NWP archives, and outcomes will diverge.
