# ---------- build stage ----------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# System deps for compiling wheels (asyncpg, numpy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip wheel --no-cache-dir --wheel-dir /wheels .

# ---------- runtime stage ----------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MEDHAS_LOG_FORMAT=json \
    MEDHAS_LOG_FILE=/app/logs/medhas.log

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl postgresql-client tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r medhas && useradd -r -g medhas medhas

COPY --from=builder /wheels /wheels
COPY pyproject.toml ./
# Install the prebuilt wheels (no build toolchain in runtime image).
RUN pip install --no-cache-dir --find-links=/wheels medhas \
    && rm -rf /wheels

# Application source (the medhas package + entrypoints).
COPY medhas ./medhas
COPY config ./config
COPY server.py main.py ./
COPY .env.example ./.env.example

RUN mkdir -p /app/logs && chown -R medhas:medhas /app
USER medhas

EXPOSE 8000

# tini reaps zombies; healthcheck hits the FastAPI health route.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["medhas-server"]
