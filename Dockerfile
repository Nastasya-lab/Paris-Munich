FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV WEATHER_TMAX_DATA_DIR=data
ENV WEATHER_TMAX_SEED_DATA_DIR=seed_data

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e .

COPY config ./config
COPY docs ./docs
COPY scripts ./scripts
COPY data ./seed_data

RUN python scripts/railway_bootstrap.py

EXPOSE 8000

CMD ["sh", "-c", "python scripts/railway_bootstrap.py && python scripts/10_start_api.py"]
